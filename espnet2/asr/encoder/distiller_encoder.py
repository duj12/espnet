import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import contextlib
import copy
import logging
import os, sys
from pathlib import Path
from typing import Optional, Tuple

import yaml
from filelock import FileLock
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.asr.specaug.specaug import SpecAug
from espnet.nets.pytorch_backend.transformer.subsampling_without_posenc import (
    Conv2dSubsamplingWOPosEnc,
)
from espnet.nets.pytorch_backend.transformer.subsampling import (
    TooShortUttError,
)
import s3prl

def check_short_utt(sub, size):
    """Check if the utterance is too short for subsampling."""
    if sub == 2 and size < 2:
        return True, 2  # for 2-times subsampling, min length is 2.
    if sub==4 and size < 7:
        return True, 7  # for 4-times subsampling, min length is 7.
    elif sub==6 and size < 6:
        return True, 6
    elif sub==8 and size < 8:
        return True, 8
    return False, -1

from fairseq.utils import get_activation_fn
from fairseq.models.wav2vec.wav2vec2 import (
    MultiheadAttention,
    SamePad,
    ConvFeatureExtractionModel,
    GradMultiply,
)

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)

def index_put(tensor, indices, value):
    tensor = tensor.clone()
    tensor[indices] = value
    return tensor

class SplitLinear(nn.Module):
    """Split Linear Layer"""

    def __init__(self, in_dim, in_split, out_dim):
        super().__init__()

        self.in_dim = in_dim  # Din
        self.in_split = in_split  # N
        self.out_dim = out_dim  # Dout

        if in_split > 1:
            # weight = torch.zeros((1, 1, self.in_split, self.in_dim, self.out_dim))
            weight = torch.zeros((self.in_split, self.in_dim, self.out_dim))
            self.weight = nn.Parameter(weight, requires_grad=True)
            nn.init.uniform_(self.weight, -(self.in_dim ** -0.5), self.in_dim ** -0.5)

            bias = torch.zeros((1, 1, self.in_split, self.out_dim))
            self.bias = nn.Parameter(bias, requires_grad=True)
            nn.init.uniform_(self.bias, -(self.in_dim ** -0.5), self.in_dim ** -0.5)
        else:
            self.layer = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x: torch.Tensor):
        # x: shape = B x T x NDin

        if self.in_split == 1:
            return self.layer(x)
        else:
            x = x.reshape(x.shape[0], x.shape[1], self.in_split, 1, self.in_dim)
            # x: B x T x N x 1 x Din

            out = torch.einsum("...klm,kmn->...kln", x, self.weight).squeeze(3)
            # out: B x T x N x Dout
            out = out + self.bias

            return out.reshape(x.shape[0], x.shape[1], -1)

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        print(f"[TransformerEncoder] - Attention type = {args.attention_type}")
        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    attention_type=args.attention_type,
                )
                for _ in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, attn_mask=None, get_hidden=False):
        x, layer_results = self.extract_features(
            x, padding_mask, attn_mask, get_hidden=get_hidden
        )

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, attn_mask=None, get_hidden=False):

        if padding_mask is not None:
            #x[padding_mask] = 0  # avoid inplace value assignment
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                    self_attn_mask=attn_mask,
                )
                if get_hidden:
                    layer_results.append(x.transpose(0, 1))

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        attention_type: str = "original",
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.attention_type = attention_type
        if attention_type == "original":
            self.self_attn = MultiheadAttention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                self_attention=True,
            )
        elif attention_type == "sparse":
            from fairseq.modules.sparse_multihead_attention import (
                SparseMultiheadAttention,
            )

            self.self_attn = SparseMultiheadAttention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                self_attention=True,
                stride=32,
                expressivity=16,
            )
        elif attention_type == "dynamic":
            from fairseq.modules import DynamicConv

            self.self_attn = DynamicConv(
                self.embedding_dim,
                kernel_size=31,
                padding_l=15,
                num_heads=num_attention_heads,
                weight_dropout=0.0,
                weight_softmax=True,
                bias=True,
            )
        else:
            raise NotImplementedError(f"Unknown attention type {attention_type}")

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward_self_attn(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
    ):
        if self.attention_type in ["original", "sparse"]:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
            )
        elif self.attention_type == "dynamic":
            x = self.self_attn(x)
            attn = None

        return x, attn

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.forward_self_attn(
                x,
                self_attn_mask=self_attn_mask,
                need_weights=False,
                self_attn_padding_mask=self_attn_padding_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.forward_self_attn(
                x,
                self_attn_mask=self_attn_mask,
                need_weights=need_weights,
                self_attn_padding_mask=self_attn_padding_mask,
            )

            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn

class DistillerConfig:
    """
    Configuration class
    """

    def __init__(self, config: dict):
        # Feature extractor
        self.extractor_mode = str(config.get("extractor_mode", "default"))
        self.extractor_conv_feature_layers = str(
            config.get(
                "extractor_conv_feature_layers",
                "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
            )
        )
        self.extractor_dropout = float(config.get("extractor_dropout", 0.0))
        self.feature_grad_mult = float(config.get("feature_grad_mult", 1.0))

        # Convolutional relative positional encoding
        self.conv_pos = int(config.get("conv_pos", 128))
        self.conv_pos_groups = int(config.get("conv_pos_groups", 16))

        # Transformer encoder
        self.encoder_layers = int(config.get("encoder_layers", 1))
        self.layer_type = str(config.get("layer_type", "transformer"))
        self.encoder_embed_dim = int(config.get("encoder_embed_dim", 768))
        self.encoder_ffn_embed_dim = int(config.get("encoder_ffn_embed_dim", 3072))
        self.encoder_attention_heads = int(config.get("encoder_attention_heads", 12))
        self.activation_fn = str(config.get("activation_fn", "gelu"))
        self.layer_norm_first = bool(config.get("layer_norm_first", False))
        self.attention_type = str(config.get("attention_type", "original"))

        # Dropout
        self.dropout = float(config.get("dropout", 0.1))
        self.attention_dropout = float(config.get("attention_dropout", 0.1))
        self.activation_dropout = float(config.get("activation_dropout", 0.1))
        self.encoder_layerdrop = float(config.get("encoder_layerdrop", 0.0))

        # Output
        self.final_dim = int(config.get("final_dim", 768))
        self.out_layer_type = str(config.get("out_layer_type", "expand-last"))
        self.out_layer_inter_dim = int(config.get("out_layer_inter_dim", -1))

        # Task & loss
        self.n_tasks = int(config.get("n_tasks", 12))
        self.task_emb_type = str(config.get("task_emb_type", "expand-last"))
        self.task_emb_size = int(config.get("task_emb_size", 0))
        self.layer_emb_size = int(config.get("layer_emb_size", 0))
        self.loss_type = str(config.get("loss_type", "l1"))
        self.feat_pen_loss = float(config.get("feat_pen_loss", 0.0))
        self.cosine_loss = float(config.get("cosine_loss", 0.0))

        # When task_emb_type == 'expand-last' only
        self.pred_layer_id = list(
            config.get("pred_layer_id", range(1, self.n_tasks + 1))
        )

        # Initialization
        self.init_teacher_conv_layers = bool(
            config.get("init_teacher_conv_layers", False)
        )
        self.init_teacher_encoder_layers = bool(
            config.get("init_teacher_encoder_layers", False)
        )

class DistillerModel(nn.Module):
    """
    Distiller Model
    """

    def __init__(self, config: DistillerConfig):
        super().__init__()

        self.config = config

        self.conv_layers = eval(config.extractor_conv_feature_layers)
        feat_emb_dim = self.conv_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(
            self.conv_layers,
            dropout=config.extractor_dropout,
            mode=config.extractor_mode,
            conv_bias=False,
        )
        self.feature_grad_mult = config.feature_grad_mult

        self.n_tasks = config.n_tasks
        self.task_emb_type = config.task_emb_type

        final_emb_size = config.encoder_embed_dim
        if self.task_emb_type == "add":
            self.task_embedding = nn.Embedding(config.n_tasks, config.encoder_embed_dim)
            nn.init.normal_(self.task_embedding.weight, 0.0, 0.1)
        elif self.task_emb_type == "concat":
            assert config.task_emb_size > 0
            feat_emb_dim += config.task_emb_size
            self.task_embedding = nn.Embedding(config.n_tasks, config.task_emb_size)
        elif self.task_emb_type == "concat-last":
            assert config.task_emb_size > 0
            self.task_embedding = nn.Embedding(config.n_tasks, config.task_emb_size)
            final_emb_size += config.task_emb_size
        elif self.task_emb_type == "expand-last":
            self.pred_layer_id = config.pred_layer_id
            assert self.n_tasks == len(self.pred_layer_id)
            print(
                f"[DistillerModel] - Expands the output dimension by {self.n_tasks} times"
            )
            print(f"[DistillerModel] - Pred layers: {self.pred_layer_id}")
        elif self.task_emb_type == "self-hidden":
            self.pred_layer_id = config.pred_layer_id
            assert self.n_tasks == len(self.pred_layer_id)
            assert self.n_tasks == config.encoder_layers + 1
            print("[DistillerModel] - Predicting with self-hidden layers")
            print(f"[DistillerModel] - Pred layers: {self.pred_layer_id}")
        elif self.task_emb_type == "none":
            print(
                f"[DistillerModel] - Disabled task embedding (predicts only layer {self.n_tasks})"
            )
        else:
            raise NotImplementedError(f"Unknown task emb type {self.task_emb_type}")

        self.post_extract_proj = (
            nn.Linear(feat_emb_dim, config.encoder_embed_dim)
            if feat_emb_dim != config.encoder_embed_dim
            else None
        )

        if config.encoder_layers > 0:
            if config.layer_type == "transformer":
                self.encoder = TransformerEncoder(config)
            else: #TODO: add conformer encoder
                raise NotImplementedError(f"{config.layer_type} is not implemented")
        else:
            self.encoder = nn.GELU()

        final_dim = config.final_dim * (
            1 if self.task_emb_type != "expand-last" else self.n_tasks
        )

        inter_dim = config.out_layer_inter_dim
        inter_dim = inter_dim if inter_dim > 0 else final_emb_size

        print(f"[DistillerModel] - Out layer type: {config.out_layer_type}")
        if config.out_layer_type == "expand-last":
            assert self.task_emb_type == "expand-last"
            print(f"[DistillerModel] - Inter dim = {inter_dim}")
            self.output_layer = nn.Sequential(
                nn.Linear(final_emb_size, inter_dim * self.n_tasks),
                nn.GELU(),
                SplitLinear(inter_dim, self.n_tasks, config.final_dim),
            )
        elif config.out_layer_type in {"none", "self-hidden"}:
            self.output_layer = None
        else:
            raise NotImplementedError(f"Unknown out layer type {config.out_layer_type}")

    def forward_feature(self, wave, pad_mask):
        """Forward feature extractor"""

        if self.feature_grad_mult > 0:
            feat = self.feature_extractor(wave)
            if self.feature_grad_mult != 1.0:
                feat = GradMultiply.apply(feat, self.feature_grad_mult)
        else:
            with torch.no_grad():
                feat = self.feature_extractor(wave)

        feat = feat.transpose(1, 2)  # B x T x D
        pad_mask = self.cal_pad_mask(pad_mask, feat.shape[1])

        return feat, pad_mask

    def forward(self, wave, pad_mask, task_id=None, get_hidden=True, no_pred=False):
        """
        Forward function
        Input:
            wave (FloatTensor): B x T_wave
            pad_mask (BoolTensor): B x T_wave
            task_id (LongTensor): N >= 1
        """

        feat, pad_mask = self.forward_feature(wave, pad_mask)

        if self.task_emb_type not in ["none", "expand-last", "self-hidden"]:
            if task_id is None:
                task_id = self.generate_task_id(feat.device)
            elif isinstance(task_id, list):
                task_id = torch.LongTensor(task_id).to(feat.device)
            task_embs = self.task_embedding(task_id)
            # N x D
            n_sz = len(task_id)
        else:
            n_sz = 1
        b_sz, t_sz, _ = feat.shape

        if self.task_emb_type == "add":
            # Add embs to feature
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat)
            else:
                feat_final = feat
            feat_final = feat_final.unsqueeze(1) + task_embs.unsqueeze(0).unsqueeze(2)
        elif self.task_emb_type == "concat":
            # Concatenates embs to feature
            feat_final = torch.cat(
                [
                    feat.unsqueeze(1).expand(-1, n_sz, -1, -1),
                    task_embs.unsqueeze(0).unsqueeze(2).expand(b_sz, -1, t_sz, -1),
                ],
                dim=-1,
            )
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat_final)
        else:
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat)
            else:
                feat_final = feat
            feat_final = feat_final.unsqueeze(1)
        # feat_final: B x N x T x D or B x 1 x T x D

        pad_mask = pad_mask.unsqueeze(1).expand(-1, n_sz, -1).reshape(b_sz * n_sz, t_sz)
        # BN x T
        feat_final = feat_final.reshape(b_sz * n_sz, t_sz, -1)
        # BN x T x D

        layer_hiddens = []
        if self.config.encoder_layers > 0:
            get_hidden_tmp = (
                True if (self.task_emb_type == "self-hidden") else get_hidden
            )
            hidden, layer_hiddens = self.encoder(
                feat_final, pad_mask.bool(), get_hidden=get_hidden_tmp
            )
        else:
            hidden = self.encoder(feat_final)

        if not no_pred:
            if self.task_emb_type == "self-hidden":
                pred = torch.stack([feat_final] + layer_hiddens, dim=1)
            else:
                pred = self.output_layer(hidden).reshape(b_sz, n_sz, t_sz, -1)
            # B x N x T x D
        else:
            pred = None

        if (not no_pred) and self.task_emb_type == "expand-last":
            assert n_sz == 1, n_sz
            pred = (
                pred.squeeze(1)
                .reshape(b_sz, t_sz, self.n_tasks, -1)
                .permute(0, 2, 1, 3)
            )
            # B x N x T x D

        return {'features': feat_final,  # 512 -> 256, 和 transformer匹配
                    'x': hidden,         # 最后一层transformer
                    'prediction': pred,  # 最后的Linear,用来蒸馏Hubert-base, 768维
                    'padding_mask': pad_mask,
                    'hiddens': layer_hiddens  #中间的transformer层
                }

    def extract_features(self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(source, padding_mask)
        # pred: B x N x T x D
        prediction_feats = res['prediction'].transpose(0, 1).split(1, 0)
        prediction_feats = [hid.squeeze(0) for hid in prediction_feats]

        # 返回CNN+Transformer+Linear
        if 'output_intermediate_layers' in kwargs:
            if kwargs['output_intermediate_layers']:
                hidden_feats = [res['features']] + res['hiddens'] + prediction_feats
                return hidden_feats, res["padding_mask"]

        if ret_conv:   # 直接返回CNN最后一层
            return res['features'], res['padding_mask']

        if output_layer is not None:  #返回transformer某一层
            return res['hiddens'][output_layer - 1], res["padding_mask"]

        #默认返回最后一层Linear
        return prediction_feats[-1], res['padding_mask']


    def cal_pad_mask(self, pad_mask, max_len):
        """Calculates pad mask after conv."""
        pad_len = (pad_mask > 0).sum(1).long()
        for conv_l in self.conv_layers:
            if len(conv_l) == 3:
                _, k_size, s_size = conv_l
            elif len(conv_l) == 4:
                _, k_size, s_size, _ = conv_l
            else:
                raise NotImplementedError(f"conv_layer config  {conv_l} is not allowed.")
            pad_len = torch.div((pad_len - k_size), s_size, rounding_mode="trunc") + 1

        new_pad_mask = torch.ones(
            (pad_mask.shape[0], max_len), dtype=pad_mask.dtype, device=pad_mask.device
        )

        for idx in range(pad_len.shape[0]):
            new_pad_mask[idx, pad_len[idx] :] = 0

        return new_pad_mask

    def generate_task_id(self, device):
        return torch.arange(self.n_tasks, device=device, dtype=torch.long)

def load_distiller_model(ckpt, device='cpu'):
    # Since some old checkpoints contained pickled scheduler which needs 'optimizers'
    # module which is now moved into s3prl package.
    original_optimizer = sys.modules.get("optimizers")
    sys.modules["optimizers"] = s3prl.optimizers


    all_states = torch.load(ckpt, map_location="cpu")
    config = all_states["Config"]

    del sys.modules["optimizers"]
    if original_optimizer is not None:
        sys.modules["optimizers"] = original_optimizer
    # Set model config
    model_config = DistillerConfig(config["distiller"])
    hidden_size = model_config.encoder_embed_dim

    # Build model
    model = DistillerModel(model_config)

    # Load from a PyTorch state_dict
    model.load_state_dict(all_states["Distiller"])
    model = model.to(device)
    return model


class DistillerEncoder(AbsEncoder):
    def __init__(
        self,
        input_size: int,
        distiller_ckpt_path: str = "./",
        output_size: int = 256,
        normalize_before: bool = False,
        freeze_finetune_updates: int = 0,
        dropout_rate: float = 0.0,
        activation_dropout: float = 0.1,
        attention_dropout: float = 0.0,
        mask_length: int = 10,
        mask_prob: float = 0.75,
        mask_selection: str = "static",
        mask_other: int = 0,
        apply_mask: bool = True,
        mask_channel_length: int = 64,
        mask_channel_prob: float = 0.5,
        mask_channel_other: int = 0,
        mask_channel_selection: str = "static",
        layerdrop: float = 0.1,
        feature_grad_mult: float = 0.0,
        specaug: Optional[str] = None,
        output_layer: Optional[str] = None,
        **args,
    ):
        assert check_argument_types()
        super().__init__()
        self.apply_mask = apply_mask

        model = load_distiller_model(distiller_ckpt_path)

        d = model.config.final_dim
        self._output_size = output_size

        self.encoders = model

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        # 2. Data augmentation for spectrogram
        if specaug is not None:
            self.specaug = SpecAug(**args['specaug_conf'])
        else:
            self.specaug = None
        if output_layer and output_size:
            if output_layer == "linear":
                self.output_layer = torch.nn.Sequential(
                    torch.nn.Linear(d, output_size),
                    torch.nn.LayerNorm(output_size),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.ReLU(),
                )
                self.subsample = 1
            elif output_layer == "conv2d2":
                self.output_layer = Conv2dSubsamplingWOPosEnc(
                    d, output_size, dropout_rate, kernels=[2], strides=[2]
                )
                self.subsample = 2
            elif output_layer == "conv2d":
                self.output_layer = Conv2dSubsamplingWOPosEnc(
                    d, output_size, dropout_rate, kernels=[3, 3], strides=[2, 2]
                )
                self.subsample = 4
            elif output_layer == "conv2d6":
                self.output_layer = Conv2dSubsamplingWOPosEnc(
                    d, output_size, dropout_rate, kernels=[2, 3], strides=[2, 3]
                )
                self.subsample = 6
            elif output_layer == "conv2d8":
                self.output_layer = Conv2dSubsamplingWOPosEnc(
                    d,
                    output_size,
                    dropout_rate,
                    kernels=[2, 2, 2],
                    strides=[2, 2, 2],
                )
                self.subsample = 8
            else:
                raise ValueError("unknown output_layer: " + output_layer)
        else:
            if output_size and output_size != d:
                self.output_layer = torch.nn.Sequential(
                    torch.nn.Linear(d, output_size),
                )
            else:
                self.output_layer = None

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward Hubert ASR Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)

        ft = self.freeze_finetune_updates <= self.num_updates

        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning hubert parameters!")
        else:
            self.num_updates += 1
        with torch.no_grad() if not ft else contextlib.nullcontext():
            xs_pad, masks = self.encoders.extract_features(
                xs_pad,
                padding_mask=masks,
            )
        #xs_pad = enc_outputs["x"]  # (B,T,C),
        #masks = enc_outputs["padding_mask"]  # (B, T)

        # # save gpu memory
        # del enc_outputs
        feats_lengths = (~masks).sum(dim=1)
        if self.specaug is not None and self.training:
            xs_pad, feats_lengths = self.specaug(xs_pad, feats_lengths)

        if isinstance(self.output_layer, Conv2dSubsamplingWOPosEnc):
            short_status, limit_size = check_short_utt(self.subsample, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.output_layer(xs_pad, masks.unsqueeze(1))
            masks = masks.squeeze(1)
        elif self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        olens = (~masks).sum(dim=1)
        # if self.output_layer is not None:
        #     xs_pad = self.output_layer(xs_pad)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        return xs_pad, olens, None
