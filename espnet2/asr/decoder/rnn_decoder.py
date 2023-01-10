import random

import numpy as np
import six
import torch
import torch.nn.functional as F
from typeguard import check_argument_types
from typing import Any
from typing import List

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, pad_list, mask_by_length
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.pytorch_backend.rnn.attentions import initial_att
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet.nets.pytorch_backend.attentions import AttMoChAstable
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet2.asr.decoder.abs_decoder import AbsDecoder, AbsFirstPassDecoder, first_pass_params, second_pass_encoder_params
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder

MAX_DECODER_OUTPUT = 5
CTC_SCORING_RATIO = 1.5

def build_attention_list(
    eprojs: int,
    dunits: int,
    atype: str = "location",
    num_att: int = 1,
    num_encs: int = 1,
    aheads: int = 4,
    adim: int = 320,
    awin: int = 5,
    aconv_chans: int = 10,
    aconv_filts: int = 100,
    han_mode: bool = False,
    han_type=None,
    han_heads: int = 4,
    han_dim: int = 320,
    han_conv_chans: int = -1,
    han_conv_filts: int = 100,
    han_win: int = 5,
):

    att_list = torch.nn.ModuleList()
    if num_encs == 1:
        for i in range(num_att):
            att = initial_att(
                atype, eprojs, dunits, aheads, adim, awin, aconv_chans, aconv_filts,
            )
            att_list.append(att)
    elif num_encs > 1:  # no multi-speaker mode
        if han_mode:
            att = initial_att(
                han_type,
                eprojs,
                dunits,
                han_heads,
                han_dim,
                han_win,
                han_conv_chans,
                han_conv_filts,
                han_mode=True,
            )
            return att
        else:
            att_list = torch.nn.ModuleList()
            for idx in range(num_encs):
                att = initial_att(
                    atype,
                    eprojs,
                    dunits,
                    aheads,
                    adim,
                    awin,
                    aconv_chans,
                    aconv_filts,
                )
                att_list.append(att)
    else:
        raise ValueError(
            "Number of encoders needs to be more than one. {}".format(num_encs)
        )
    return att_list


class RNNDecoder(AbsDecoder, AbsFirstPassDecoder, BatchScorerInterface):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        hidden_size: int = 320,
        embed_dim: int = 300,
        sampling_probability: float = 0.0,
        dropout: float = 0.0,
        context_residual: bool = False,
        replace_sos: bool = False,
        num_encs: int = 1,
        multi_task_dim: int = 0,
        att_conf: dict = get_default_kwargs(build_attention_list),
    ):
        # FIXME(kamo): The parts of num_spk should be refactored more more more
        assert check_argument_types()
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported: rnn_type={rnn_type}")

        super().__init__()
        eprojs = encoder_output_size
        self.dtype = rnn_type
        self.dunits = hidden_size
        self.dlayers = num_layers
        self.context_residual = context_residual
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.odim = vocab_size
        self.sampling_probability = sampling_probability
        self.dropout = dropout
        self.num_encs = num_encs

        # for multilingual translation
        self.replace_sos = replace_sos

        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.dropout_emb = torch.nn.Dropout(p=dropout)

        self.decoder = torch.nn.ModuleList()
        self.dropout_dec = torch.nn.ModuleList()
        self.decoder += [
            torch.nn.LSTMCell(embed_dim + eprojs, hidden_size)
            if self.dtype == "lstm"
            else torch.nn.GRUCell(embed_dim + eprojs, hidden_size)
        ]
        self.dropout_dec += [torch.nn.Dropout(p=dropout)]
        for _ in range(1, self.dlayers):
            self.decoder += [
                torch.nn.LSTMCell(hidden_size, hidden_size)
                if self.dtype == "lstm"
                else torch.nn.GRUCell(hidden_size, hidden_size)
            ]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]
            # NOTE: dropout is applied only for the vertical connections
            # see https://arxiv.org/pdf/1409.2329.pdf

        if context_residual:
            self.output = torch.nn.Linear(hidden_size + eprojs, vocab_size)
        else:
            self.output = torch.nn.Linear(hidden_size, vocab_size)

        if multi_task_dim > 0:
            self.multi_task_output = torch.nn.Linear(hidden_size, multi_task_dim)
        else:
            self.multi_task_output = None

        self.att_list = build_attention_list(
            eprojs=eprojs, dunits=hidden_size, **att_conf
        )
        self.logzero = -10000000000.0

    @staticmethod
    def _get_last_yseq(exp_yseq):
        last = []
        for y_seq in exp_yseq:
            last.append(y_seq[-1])
        return last

    @staticmethod
    def _append_ids(yseq, ids):
        if isinstance(ids, list):
            for i, j in enumerate(ids):
                yseq[i].append(j)
        else:
            for i in range(len(yseq)):
                yseq[i].append(ids)
        return yseq

    @staticmethod
    def _index_select_list(yseq, lst):
        new_yseq = []
        for i in lst:
            new_yseq.append(yseq[i][:])
        return new_yseq

    @staticmethod
    def _index_select_lm_state(rnnlm_state, dim, vidx):
        if isinstance(rnnlm_state, dict):
            new_state = {}
            for k, v in rnnlm_state.items():
                new_state[k] = [torch.index_select(vi, dim, vidx) for vi in v]
        elif isinstance(rnnlm_state, list):
            new_state = []
            for i in vidx:
                new_state.append(rnnlm_state[int(i)][:])
        return new_state

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        if self.dtype == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for l in range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]), (z_prev[l], c_prev[l]),
                )
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])
            for l in range(1, self.dlayers):
                z_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]), z_prev[l]
                )
        return z_list, c_list

    def forward(self, hs_pad, hlens, ys_in_pad, ys_in_lens, strm_idx=0, **kwargs):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            hs_pad = [hs_pad]
            hlens = [hlens]

        multi_task = kwargs.get("multi_task", False)
        att_ws = []

        # attention index for the attention module
        # in SPA (speaker parallel attention),
        # att_idx is used to select attention module. In other cases, it is 0.
        att_idx = min(strm_idx, len(self.att_list) - 1)

        # hlens should be list of list of integer
        hlens = [list(map(int, hlens[idx])) for idx in range(self.num_encs)]

        # get dim, length info
        olength = ys_in_pad.size(1)

        # initialization
        c_list = [self.zero_state(hs_pad[0])]
        z_list = [self.zero_state(hs_pad[0])]
        for _ in range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad[0]))
            z_list.append(self.zero_state(hs_pad[0]))
        z_all = []
        if self.num_encs == 1:
            att_w = None
            self.att_list[att_idx].reset()  # reset pre-computation of h
        else:
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * self.num_encs  # atts
            for idx in range(self.num_encs + 1):
                # reset pre-computation of h in atts and han
                self.att_list[idx].reset()

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim
        # loop for an output sequence
        for i in range(olength):
            if self.num_encs == 1:
                att_c, att_w = self.att_list[att_idx](
                    hs_pad[0], hlens[0], self.dropout_dec[0](z_list[0]), att_w,
                )
                att_ws.append(att_w.unsqueeze(1))
            else:
                for idx in range(self.num_encs):
                    att_c_list[idx], att_w_list[idx] = self.att_list[idx](
                        hs_pad[idx],
                        hlens[idx],
                        self.dropout_dec[0](z_list[0]),
                        att_w_list[idx],
                    )
                hs_pad_han = torch.stack(att_c_list, dim=1)
                hlens_han = [self.num_encs] * len(ys_in_pad)
                att_c, att_w_list[self.num_encs] = self.att_list[self.num_encs](
                    hs_pad_han,
                    hlens_han,
                    self.dropout_dec[0](z_list[0]),
                    att_w_list[self.num_encs],
                )
            if i > 0 and random.random() < self.sampling_probability:
                z_out = self.output(z_all[-1])
                z_out = np.argmax(z_out.detach().cpu(), axis=1)
                z_out = self.dropout_emb(self.embed(to_device(self, z_out)))
                ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
            else:
                # utt x (zdim + hdim)
                ey = torch.cat((eys[:, i, :], att_c), dim=1)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
            if self.context_residual:
                z_all.append(
                    torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
                )  # utt x (zdim + hdim)
            else:
                z_all.append(self.dropout_dec[-1](z_list[-1]))  # utt x (zdim)

        z_all = torch.stack(z_all, dim=1)
        z_all = self.output(z_all) if multi_task is False else self.multi_task_output(z_all)
        z_all.masked_fill_(
            make_pad_mask(ys_in_lens, z_all, 1), 0,
        )

        self.att_ws = torch.cat(att_ws, dim=1)[:, :-1]
        return z_all, ys_in_lens

    def init_state(self, x, trigger_point=None):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        c_list = [self.zero_state(x[0].unsqueeze(0))]
        z_list = [self.zero_state(x[0].unsqueeze(0))]
        for _ in range(1, self.dlayers):
            c_list.append(self.zero_state(x[0].unsqueeze(0)))
            z_list.append(self.zero_state(x[0].unsqueeze(0)))
        # TODO(karita): support strm_index for `asr_mix`
        strm_index = 0
        att_idx = min(strm_index, len(self.att_list) - 1)
        if self.num_encs == 1:
            a = None
            self.att_list[att_idx].reset(trigger_point)  # reset pre-computation of h
        else:
            a = [None] * (self.num_encs + 1)  # atts + han
            for idx in range(self.num_encs + 1):
                # reset pre-computation of h in atts and han
                self.att_list[idx].reset(trigger_point)
        return dict(
            c_prev=c_list[:],
            z_prev=z_list[:],
            a_prev=a,
            ctc_index=0,
            workspace=(att_idx, z_list, c_list),
        )

    def score(self, yseq, state, x):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        att_idx, z_list, c_list = state["workspace"]
        vy = yseq[-1].unsqueeze(0)
        ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
        if self.num_encs == 1:
            attn = self.att_list[att_idx]
            if isinstance(attn, AttMoChAstable):
                att_c, ctc_index, att_w = self.att_list[att_idx](
                    x[0].unsqueeze(0),
                    [x[0].size(0)],
                    self.dropout_dec[0](state["z_prev"][0]),
                    state["a_prev"],
                    soft=False,
                    order=1,
                    ctc_index=state["ctc_index"]
                )
            else:
                ctc_index = None
                att_c, att_w = self.att_list[att_idx](
                    x[0].unsqueeze(0),
                    [x[0].size(0)],
                    self.dropout_dec[0](state["z_prev"][0]),
                    state["a_prev"],
                )
        else:
            att_w = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * self.num_encs  # atts
            for idx in range(self.num_encs):
                att_c_list[idx], att_w[idx] = self.att_list[idx](
                    x[idx].unsqueeze(0),
                    [x[idx].size(0)],
                    self.dropout_dec[0](state["z_prev"][0]),
                    state["a_prev"][idx],
                )
            h_han = torch.stack(att_c_list, dim=1)
            att_c, att_w[self.num_encs] = self.att_list[self.num_encs](
                h_han,
                [self.num_encs],
                self.dropout_dec[0](state["z_prev"][0]),
                state["a_prev"][self.num_encs],
            )
        ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
        z_list, c_list = self.rnn_forward(
            ey, z_list, c_list, state["z_prev"], state["c_prev"]
        )
        if self.context_residual:
            logits = self.output(
                torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
            )
        else:
            logits = self.output(self.dropout_dec[-1](z_list[-1]))
        logp = F.log_softmax(logits, dim=1).squeeze(0)

        return (
            logp,
            dict(
                c_prev=c_list[:],
                z_prev=z_list[:],
                a_prev=att_w,
                ctc_index=ctc_index,
                workspace=(att_idx, z_list, c_list),
            ),
        )

    def batch_init_state(self, x, **kwargs):
        trigger_point = kwargs.get("trigger_point", None)
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        c_list = [self.zero_state(x[0].unsqueeze(0)).squeeze(0)]
        z_list = [self.zero_state(x[0].unsqueeze(0)).squeeze(0)]
        for _ in range(1, self.dlayers):
            c_list.append(self.zero_state(x[0].unsqueeze(0)).squeeze(0))
            z_list.append(self.zero_state(x[0].unsqueeze(0)).squeeze(0))
        # TODO(karita): support strm_index for `asr_mix`
        strm_index = 0
        att_idx = min(strm_index, len(self.att_list) - 1)
        if self.num_encs == 1:
            a = None
            self.att_list[att_idx].reset(trigger_point)  # reset pre-computation of h
        else:
            a = [None] * (self.num_encs)  # atts + han
            for idx in range(self.num_encs):
                # reset pre-computation of h in atts and han
                self.att_list[idx].reset()
        state = [dict(c_prev=c_list[i],
                      z_prev=z_list[i],
                      a_prev=a if i == 0 else None,
                      ctc_index=0 if i == 0 else None,
                      a_list=[] if i == 0 else None,
                      att_idx=att_idx) for i in range(self.dlayers)]
        return state

    def batch_score(
            self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor):
        n_batch = len(ys)
        if states[0] is None:
            batch_z_list = None
            batch_c_list = None
            batch_ctc_index = [0] * n_batch
            a_list = [[]]
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_z_list = [
                torch.stack([states[b][i]["z_prev"] for b in range(n_batch)])
                for i in range(self.dlayers)
            ]
            batch_c_list = [
                torch.stack([states[b][i]["c_prev"] for b in range(n_batch)])
                for i in range(self.dlayers)
            ]
            batch_ctc_index = [states[b][0]["ctc_index"] for b in range(n_batch)]
            a_list = [states[b][0]["a_list"] for b in range(n_batch)]
        if states[0][0]["a_prev"] is None:
            batch_a_prev = None
        else:
            batch_a_prev = torch.stack([states[b][0]["a_prev"] for b in range(n_batch)])

        att_idx = states[0][0]["att_idx"]
        vy = ys[:, -1]
        ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
        if self.num_encs == 1:
            attn = self.att_list[att_idx]
            if isinstance(attn, AttMoChAstable):
                if hasattr(attn, 'trigger_point') and not attn.trigger_point is None:
                    att_c, ctc_index, att_w = self.att_list[att_idx](
                        xs[0].unsqueeze(0),
                        [xs[0].size(0)],
                        self.dropout_dec[0](batch_z_list[0]),
                        batch_a_prev,
                        soft=False,
                        order=1,
                        ctc_index=batch_ctc_index,
                    )
                else:
                    ctc_index = [None] * n_batch
                    att_c, att_w = self.att_list[att_idx](
                        xs[0].unsqueeze(0),
                        [xs[0].size(0)],
                        self.dropout_dec[0](batch_z_list[0]),
                        batch_a_prev,
                        soft=False,
                        order=1,
                        ctc_index=batch_ctc_index,
                    )

                attend_ids = att_w.max(dim=1)[1].tolist()
                for attend_his, new_id in zip(a_list, attend_ids):
                    attend_his.append(new_id)
            else:
                ctc_index = [None] * n_batch
                att_c, att_w = attn(
                    xs[0].unsqueeze(0),
                    [xs[0].size(0)],
                    self.dropout_dec[0](batch_z_list[0]),
                    batch_a_prev,
                )

        ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
        batch_z_list, batch_c_list = self.rnn_forward(
            ey, batch_z_list, batch_c_list, batch_z_list, batch_c_list
        )
        if self.context_residual:
            logits = self.output(
                torch.cat((self.dropout_dec[-1](batch_z_list[-1]), att_c), dim=-1)
            )
        else:
            logits = self.output(self.dropout_dec[-1](batch_z_list[-1]))
        logp = F.log_softmax(logits, dim=1)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[dict(c_prev=batch_c_list[i][b],
                            z_prev=batch_z_list[i][b],
                            a_prev=att_w[b] if i == 0 else None,
                            a_list=a_list[b] if i == 0 else None,
                            ctc_index=ctc_index[b] if i == 0 else None,
                            att_idx=att_idx) for i in range(self.dlayers)] for b in range(n_batch)]

        return (
            logp,
            state_list,
        )

    def greedy(
        self,
        hs_pad: torch.Tensor,
        max_len_ratio: float=0.5,
    ):
        ymax = int(hs_pad.shape[1] * max_len_ratio)
        y = hs_pad.new_zeros((1), dtype=torch.int64).fill_(self.eos)
        state = self.init_state(hs_pad, trigger_point=None)
        hyp = []
        for i in range(ymax):
            logp, state = self.score(y, state, hs_pad[0])
            y = logp.argmax(dim=0, keepdims=True)
            hyp.append(y)
            if y.item() == self.eos:
                break
        hyp = torch.cat(hyp).unsqueeze(0)
        return hyp

    def recognize_beam_batch(self, h, hlens, lpz, recog_args, char_list, rnnlm=None,
                             normalize_score=True, strm_idx=0, lang_ids=None):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            h = [h]
            hlens = [hlens]
            lpz = [lpz]
        if self.num_encs > 1 and lpz is None:
            lpz = [lpz] * self.num_encs

        att_idx = min(strm_idx, len(self.att_list) - 1)
        for idx in range(self.num_encs):
            h[idx] = mask_by_length(h[idx], hlens[idx], 0.0)

        # search params
        batch = len(hlens[0])
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = getattr(recog_args, "ctc_weight", 0)  # for NMT
        att_weight = 1.0 - ctc_weight
        ctc_margin = getattr(
            recog_args, "ctc_window_margin", 0
        )  # use getattr to keep compatibility
        # weights-ctc,
        # e.g. ctc_loss = w_1*ctc_1_loss + w_2 * ctc_2_loss + w_N * ctc_N_loss
        if lpz[0] is not None and self.num_encs > 1:
            weights_ctc_dec = recog_args.weights_ctc_dec / np.sum(
                recog_args.weights_ctc_dec
            )  # normalize
        else:
            weights_ctc_dec = [1.0]

        n_bb = batch * beam
        pad_b = to_device(h[0], torch.arange(batch) * beam).view(-1, 1)

        max_hlen = np.amin([max(hlens[idx]) for idx in range(self.num_encs)])
        if recog_args.maxlenratio == 0:
            maxlen = max_hlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
        minlen = int(recog_args.minlenratio * max_hlen)

        # initialization
        c_prev = [
            to_device(h[0], torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)
        ]
        z_prev = [
            to_device(h[0], torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)
        ]
        c_list = [
            to_device(h[0], torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)
        ]
        z_list = [
            to_device(h[0], torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)
        ]
        vscores = to_device(h[0], torch.zeros(batch, beam))

        rnnlm_state = None
        if self.num_encs == 1:
            a_prev = [None]
            att_w_list, ctc_scorer, ctc_state = [None], [None], [None]
            self.att_list[att_idx].reset()  # reset pre-computation of h
        else:
            a_prev = [None] * (self.num_encs + 1)  # atts + han
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            ctc_scorer, ctc_state = [None] * (self.num_encs), [None] * (self.num_encs)
            for idx in range(self.num_encs + 1):
                self.att_list[idx].reset()  # reset pre-computation of h in atts and han

        if self.replace_sos and recog_args.tgt_lang:
            yseq = [
                [char_list.index(recog_args.tgt_lang)] for _ in six.moves.range(n_bb)
            ]
        elif lang_ids is not None:
            # NOTE: used for evaluation during training
            yseq = [
                [lang_ids[b // recog_args.beam_size]] for b in six.moves.range(n_bb)
            ]
        else:
            yseq = [[self.sos] for _ in six.moves.range(n_bb)]

        accum_odim_ids = [self.sos for _ in six.moves.range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in range(batch)]

        exp_hlens = [
            hlens[idx].repeat(beam).view(beam, batch).transpose(0, 1).contiguous()
            for idx in range(self.num_encs)
        ]
        exp_hlens = [exp_hlens[idx].view(-1).tolist() for idx in range(self.num_encs)]
        exp_h = [
            h[idx].unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
            for idx in range(self.num_encs)
        ]
        exp_h = [
            exp_h[idx].view(n_bb, h[idx].size()[1], h[idx].size()[2])
            for idx in range(self.num_encs)
        ]

        if lpz[0] is not None:
            scoring_num = min(
                int(beam * CTC_SCORING_RATIO)
                if att_weight > 0.0 and not lpz[0].is_cuda
                else 0,
                lpz[0].size(-1),
            )
            ctc_scorer = [
                CTCPrefixScoreTH(
                    lpz[idx],
                    hlens[idx],
                    0,
                    self.eos,
                    margin=ctc_margin,
                )
                for idx in range(self.num_encs)
            ]

        for i in six.moves.range(maxlen):
            vy = to_device(h[0], torch.LongTensor(self._get_last_yseq(yseq)))
            ey = self.dropout_emb(self.embed(vy))
            if self.num_encs == 1:
                att_c, att_w = self.att_list[att_idx](
                    exp_h[0], exp_hlens[0], self.dropout_dec[0](z_prev[0]), a_prev[0]
                )
                att_w_list = [att_w]
            else:
                for idx in range(self.num_encs):
                    att_c_list[idx], att_w_list[idx] = self.att_list[idx](
                        exp_h[idx],
                        exp_hlens[idx],
                        self.dropout_dec[0](z_prev[0]),
                        a_prev[idx],
                    )
                exp_h_han = torch.stack(att_c_list, dim=1)
                att_c, att_w_list[self.num_encs] = self.att[self.num_encs](
                    exp_h_han,
                    [self.num_encs] * n_bb,
                    self.dropout_dec[0](z_prev[0]),
                    a_prev[self.num_encs],
                )
            ey = torch.cat((ey, att_c), dim=1)

            # attention decoder
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_prev, c_prev)
            if self.context_residual:
                logits = self.output(
                    torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
                )
            else:
                logits = self.output(self.dropout_dec[-1](z_list[-1]))
            local_scores = att_weight * F.log_softmax(logits, dim=1)

            # rnnlm
            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_state, vy, n_bb)
                local_scores = local_scores + recog_args.lm_weight * local_lm_scores

            # ctc
            if ctc_scorer[0]:
                local_scores[:, 0] = self.logzero  # avoid choosing blank
                part_ids = (
                    torch.topk(local_scores, scoring_num, dim=-1)[1]
                    if scoring_num > 0
                    else None
                )
                for idx in range(self.num_encs):
                    att_w = att_w_list[idx]
                    att_w_ = att_w if isinstance(att_w, torch.Tensor) else att_w[0]
                    local_ctc_scores, ctc_state[idx] = ctc_scorer[idx](
                        yseq, ctc_state[idx], part_ids, att_w_
                    )
                    local_scores = (
                        local_scores
                        + ctc_weight * weights_ctc_dec[idx] * local_ctc_scores
                    )

            local_scores = local_scores.view(batch, beam, self.odim)
            if i == 0:
                local_scores[:, 1:, :] = self.logzero

            # accumulate scores
            eos_vscores = local_scores[:, :, self.eos] + vscores
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.odim)
            vscores[:, :, self.eos] = self.logzero
            vscores = (vscores + local_scores).view(batch, -1)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1)
            accum_odim_ids = (
                torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist()
            )
            accum_padded_beam_ids = (
                (accum_best_ids // self.odim + pad_b).view(-1).data.cpu().tolist()
            )

            y_prev = yseq[:][:]
            yseq = self._index_select_list(yseq, accum_padded_beam_ids)
            yseq = self._append_ids(yseq, accum_odim_ids)
            vscores = accum_best_scores
            vidx = to_device(h[0], torch.LongTensor(accum_padded_beam_ids))

            a_prev = []
            num_atts = self.num_encs if self.num_encs == 1 else self.num_encs + 1
            for idx in range(num_atts):
                if isinstance(att_w_list[idx], torch.Tensor):
                    _a_prev = torch.index_select(
                        att_w_list[idx].view(n_bb, *att_w_list[idx].shape[1:]), 0, vidx
                    )
                elif isinstance(att_w_list[idx], list):
                    # handle the case of multi-head attention
                    _a_prev = [
                        torch.index_select(att_w_one.view(n_bb, -1), 0, vidx)
                        for att_w_one in att_w_list[idx]
                    ]
                else:
                    # handle the case of location_recurrent when return is a tuple
                    _a_prev_ = torch.index_select(
                        att_w_list[idx][0].view(n_bb, -1), 0, vidx
                    )
                    _h_prev_ = torch.index_select(
                        att_w_list[idx][1][0].view(n_bb, -1), 0, vidx
                    )
                    _c_prev_ = torch.index_select(
                        att_w_list[idx][1][1].view(n_bb, -1), 0, vidx
                    )
                    _a_prev = (_a_prev_, (_h_prev_, _c_prev_))
                a_prev.append(_a_prev)
            z_prev = [
                torch.index_select(z_list[li].view(n_bb, -1), 0, vidx)
                for li in range(self.dlayers)
            ]
            c_prev = [
                torch.index_select(c_list[li].view(n_bb, -1), 0, vidx)
                for li in range(self.dlayers)
            ]

            # pick ended hyps
            if i >= minlen:
                k = 0
                penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1]
                for samp_i in six.moves.range(batch):
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in six.moves.range(beam):
                        _vscore = None
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            if len(yk) <= min(
                                hlens[idx][samp_i] for idx in range(self.num_encs)
                            ):
                                _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                        elif i == maxlen - 1:
                            yk = yseq[k][:]
                            _vscore = vscores[samp_i][beam_j] + penalty_i
                        if _vscore:
                            yk.append(self.eos)
                            if rnnlm:
                                _vscore += recog_args.lm_weight * rnnlm.final(
                                    rnnlm_state, index=k
                                )
                            _score = _vscore.data.cpu().numpy()
                            ended_hyps[samp_i].append(
                                {"yseq": yk, "vscore": _vscore, "score": _score}
                            )
                        k = k + 1

            # end detection
            stop_search = [
                stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                for samp_i in six.moves.range(batch)
            ]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

            if rnnlm:
                rnnlm_state = self._index_select_lm_state(rnnlm_state, 0, vidx)
            if ctc_scorer[0]:
                for idx in range(self.num_encs):
                    ctc_state[idx] = ctc_scorer[idx].index_select_state(
                        ctc_state[idx], accum_best_ids
                    )

        torch.cuda.empty_cache()

        dummy_hyps = [
            {"yseq": [self.sos, self.eos], "score": np.array([-float("inf")])}
        ]
        ended_hyps = [
            ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
            for samp_i in six.moves.range(batch)
        ]
        if normalize_score:
            for samp_i in six.moves.range(batch):
                for x in ended_hyps[samp_i]:
                    x["score"] /= len(x["yseq"])

        nbest_hyps = [
            sorted(ended_hyps[samp_i], key=lambda x: x["score"], reverse=True)[
                : min(len(ended_hyps[samp_i]), recog_args.nbest)
            ]
            for samp_i in six.moves.range(batch)
        ]

        return nbest_hyps

class SecondPassRNNDecoder(RNNDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        hidden_size: int = 320,
        embed_dim: int = 300,
        sampling_probability: float = 0.0,
        dropout: float = 0.0,
        context_residual: bool = False,
        replace_sos: bool = False,
        num_encs: int = 1,
        token_list: List[str] = None,
        first_pass_decoder: AbsFirstPassDecoder = None,
        first_pass_conf: dict = first_pass_params,
        second_pass_encoder_conf: dict = second_pass_encoder_params,
        att_conf: dict = get_default_kwargs(build_attention_list),
    ):
        # FIXME(kamo): The parts of num_spk should be refactored more more more
        assert check_argument_types()
        super().__init__(
            vocab_size,
            encoder_output_size,
            rnn_type,
            num_layers,
            hidden_size,
            embed_dim,
            sampling_probability,
            dropout,
            context_residual,
            replace_sos,
            num_encs,
            att_conf,
        )
        assert first_pass_decoder is not None, f"First pass decoder is required"
        assert first_pass_conf["beams"] >= 1, first_pass_conf["beams"]
        self.first_pass_decoding_mode = "greedy_search" if first_pass_conf["beams"] == 1 else "beam_search"
        self.use_first_pass = first_pass_conf["apply"]
        if self.first_pass_decoding_mode == "beam_search" and self.use_first_pass:
            first_pass_conf.update(token_list=token_list)
            first_pass_decoder.init_beam_decoder(**first_pass_conf)
        assert num_encs == 2, num_encs
        eprojs = encoder_output_size
        self.first_pass_decoder = first_pass_decoder
        self.token_list = token_list

        # decoder
        if self.use_first_pass:
            self.decoder[0] = torch.nn.LSTMCell(embed_dim + second_pass_encoder_conf["output_size"] * 2, hidden_size) \
            if self.dtype == "lstm" else torch.nn.GRUCell(embed_dim + eprojs, hidden_size)
        else:
            self.decoder[0] = torch.nn.LSTMCell(embed_dim + second_pass_encoder_conf["output_size"], hidden_size) \
                if self.dtype == "lstm" else torch.nn.GRUCell(embed_dim + eprojs, hidden_size)

        self.enc_subsample = Conv2dSubsampling(80, encoder_output_size, dropout_rate=dropout)
        if second_pass_encoder_conf["type"] == "blstm":
            # encode first-pass hyp
            self.hyp_encoder = RNN(embed_dim,
                                   second_pass_encoder_conf["num_blocks"],
                                   second_pass_encoder_conf["hidden_size"],
                                   second_pass_encoder_conf["output_size"],
                                   dropout,
                                   typ="blstm")
            if self.use_first_pass:
                # encode first-pass encoder out
                self.enc_encoder = RNN(encoder_output_size,
                                       second_pass_encoder_conf["num_blocks"],
                                       second_pass_encoder_conf["hidden_size"],
                                       second_pass_encoder_conf["output_size"],
                                       dropout,
                                       typ="blstm")
        elif second_pass_encoder_conf["type"] == "transformer":
            transformer_params = {k:v for k,v in second_pass_encoder_conf.items() if k != "type"}
            self.enc_encoder = TransformerEncoder(
                input_size=encoder_output_size,
                **transformer_params,
            )
            if self.use_first_pass:
                # reduce hyp encoder layer to 1, avoid overfitting
                transformer_params["num_blocks"] = 1
                self.hyp_encoder = TransformerEncoder(
                    input_size=embed_dim,
                    **transformer_params,
                )
        elif second_pass_encoder_conf["type"] == "conformer":
            conformer_params = {k:v for k,v in second_pass_encoder_conf.items() if k != "type"}
            self.enc_encoder = ConformerEncoder(
                input_size=encoder_output_size,
                **conformer_params,
            )
            # reduce hyp encoder layer to 1, avoid overfitting
            if self.use_first_pass:
                conformer_params["num_blocks"] = 1
                self.hyp_encoder = ConformerEncoder(
                    input_size=embed_dim,
                    **conformer_params,
            )
        else:
            raise NotImplementedError(f"{second_pass_encoder_conf['type']} is not supported!")

        self.att_list = build_attention_list(
            eprojs=second_pass_encoder_conf["output_size"], dunits=hidden_size, num_encs=num_encs,
            **att_conf
        )

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        if self.dtype == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for l in range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]), (z_prev[l], c_prev[l]),
                )
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])
            for l in range(1, self.dlayers):
                z_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]), z_prev[l]
                )
        return z_list, c_list

    def first_pass(self, hs_pad, hlens):
        with torch.no_grad():
            # beam search may be slow
            if self.first_pass_decoding_mode == "beam_search":
                best_hyps, hyp_lens = self.first_pass_decoder.beam_search(hs_pad, hlens)
            else:
                best_hyps, hyp_lens = self.first_pass_decoder.greedy(hs_pad, hlens)
            y_out = self.embed(best_hyps)

        # ctc_hyp = "".join([self.token_list[x] for x in best_hyps[0, :hyp_lens[0]]])
        # logging.info(f"ctc hyp: {ctc_hyp}")
        if self.hyp_encoder is not None:
            y_out, hyp_lens, _ = self.hyp_encoder(y_out, hyp_lens)

        return y_out, hyp_lens

    def forward(self, hs_pad, hlens, ys_in_pad, ys_in_lens, feats_si=None):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.use_first_pass:
            first_pass_y_out, first_pass_y_lens = self.first_pass(hs_pad, hlens)
        if feats_si is not None:
            feats_si, feats_si_lens = feats_si
            masks = (~make_pad_mask(feats_si_lens)[:, None, :]).to(hs_pad.device)
            feats_si, feats_si_lens = self.enc_subsample(feats_si, masks)
            hs_pad = hs_pad + feats_si
        hs_pad, hlens, _ = self.enc_encoder(hs_pad, hlens)

        if self.use_first_pass:
            hs_pad = [hs_pad, first_pass_y_out]
            hlens = [hlens, first_pass_y_lens]
        else:
            hs_pad = [hs_pad, hs_pad]
            hlens = [hlens] * self.num_encs

        # hlens should be list of list of integer
        hlens = [list(map(int, hlens[idx])) for idx in range(self.num_encs)]

        # get dim, length info
        olength = ys_in_pad.size(1)

        # initialization
        c_list = [self.zero_state(hs_pad[0])]
        z_list = [self.zero_state(hs_pad[0])]
        for _ in range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad[0]))
            z_list.append(self.zero_state(hs_pad[0]))
        z_all = []

        enc_att_w = None  # atts + han
        dec_att_w = None
        for att in self.att_list:
            # reset pre-computation of h in atts
            att.reset()

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim
        # loop for an output sequence
        for i in range(olength):
            # enc att
            enc_att_c, enc_att_w = self.att_list[0](
                    hs_pad[0], hlens[0], self.dropout_dec[0](z_list[0]), enc_att_w,
                )
            if self.use_first_pass:
                # dec att
                dec_att_c, dec_att_w = self.att_list[1](
                        first_pass_y_out, hlens[1], self.dropout_dec[0](z_list[0]), dec_att_w,
                    )
                att_c = torch.cat([enc_att_c, dec_att_c], dim=1)
            else:
                att_c = enc_att_c

            if i > 0 and random.random() < self.sampling_probability:
                z_out = self.output(z_all[-1])
                z_out = np.argmax(z_out.detach().cpu(), axis=1)
                z_out = self.dropout_emb(self.embed(to_device(self, z_out)))
                ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
            else:
                # utt x (zdim + hdim)
                ey = torch.cat((eys[:, i, :], att_c), dim=1)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
            if self.context_residual:
                z_all.append(
                    torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
                )  # utt x (zdim + hdim)
            else:
                z_all.append(self.dropout_dec[-1](z_list[-1]))  # utt x (zdim)

        z_all = torch.stack(z_all, dim=1)
        z_all = self.output(z_all)
        z_all.masked_fill_(
            make_pad_mask(ys_in_lens, z_all, 1), 0,
        )
        return z_all, ys_in_lens

    def init_state(self, x, trigger_point=None):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        x = [x]
        c_list = [self.zero_state(x[0].unsqueeze(0))]
        z_list = [self.zero_state(x[0].unsqueeze(0))]
        for _ in range(1, self.dlayers):
            c_list.append(self.zero_state(x[0].unsqueeze(0)))
            z_list.append(self.zero_state(x[0].unsqueeze(0)))

        a = [None] * self.num_encs
        self.enc_enc = None
        self.hyp_enc = None
        for idx in range(self.num_encs):
            # reset pre-computation of h in atts and han
            self.att_list[idx].reset()
        return dict(
            c_prev=c_list[:],
            z_prev=z_list[:],
            a_prev=a[:],
            ctc_index=0,
            workspace=(z_list, c_list),
        )

    def batch_init_state(self, x, trigger_point=None):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        x = [x]

        c_list = [self.zero_state(x[0].unsqueeze(0)).squeeze(0)]
        z_list = [self.zero_state(x[0].unsqueeze(0)).squeeze(0)]
        for _ in range(1, self.dlayers):
            c_list.append(self.zero_state(x[0].unsqueeze(0)).squeeze(0))
            z_list.append(self.zero_state(x[0].unsqueeze(0)).squeeze(0))

        for idx in range(self.num_encs):
            # reset pre-computation of h in atts and han
            self.att_list[idx].reset()
        state = [dict(c_prev=c_list[i],
                      z_prev=z_list[i],
                      enc_a_prev=None,
                      hyp_a_prev=None
                      ) for i in range(self.dlayers)]
        self.enc_enc = None
        self.hyp_enc = None
        return state

    def score(self, yseq, state, x):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        z_list, c_list = state["workspace"]
        enc_a_prev, hyp_a_prev = state["a_prev"]

        if self.enc_enc is None:
            hlens = torch.IntTensor([x.shape[0]])
            self.hyp_enc, first_pass_y_lens = self.first_pass(x.unsqueeze(0), hlens)
            self.enc_enc, hlens, _ = self.enc_encoder(x.unsqueeze(0), hlens)
            # hlens should be list of list of integer
            hlens = [hlens, first_pass_y_lens]
            self.hlens = [list(map(int, hlens[idx])) for idx in range(self.num_encs)]

        vy = yseq[-1].unsqueeze(0)
        ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim

        # dual-attention
        enc_att_c, enc_att_w = self.att_list[0](
            self.enc_enc, self.hlens[0], self.dropout_dec[0](z_list[0]), enc_a_prev,
        )
        dec_att_c, dec_att_w = self.att_list[1](
            self.hyp_enc, self.hlens[1], self.dropout_dec[0](z_list[0]), hyp_a_prev,
        )
        att_c = torch.cat([enc_att_c, dec_att_c], dim=1)

        # rnn forward
        ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)

        # rnn forward
        z_list, c_list = self.rnn_forward(
            ey, z_list, c_list, state["z_prev"], state["c_prev"]
        )
        if self.context_residual:
            logits = self.output(
                torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
            )
        else:
            logits = self.output(self.dropout_dec[-1](z_list[-1]))
        logp = F.log_softmax(logits, dim=1).squeeze(0)
        return (
            logp,
            dict(
                c_prev=c_list[:],
                z_prev=z_list[:],
                a_prev=[enc_att_w, dec_att_w],
                workspace=(z_list[:], c_list[:]),
            ),
        )

    def batch_score(
            self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor):
        n_batch = len(ys)
        if states[0] is None:
            batch_z_list = None
            batch_c_list = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_z_list = [
                torch.stack([states[b][i]["z_prev"] for b in range(n_batch)])
                for i in range(self.dlayers)
            ]
            batch_c_list = [
                torch.stack([states[b][i]["c_prev"] for b in range(n_batch)])
                for i in range(self.dlayers)
            ]
        if states[0][0]["enc_a_prev"] is None:
            batch_enc_a_prev = None
            batch_hyp_a_prev = None
        else:
            batch_enc_a_prev = torch.stack([states[b][0]["enc_a_prev"] for b in range(n_batch)])
            batch_hyp_a_prev = torch.stack([states[b][0]["hyp_a_prev"] for b in range(n_batch)])

        if self.enc_enc is None:
            hlens = torch.IntTensor([xs.shape[1]])
            self.hyp_enc, first_pass_y_lens = self.first_pass(xs, hlens)
            self.enc_enc, hlens, _ = self.enc_encoder(xs, hlens)
            # hlens should be list of list of integer
            hlens = [hlens, first_pass_y_lens]
            self.hlens = [list(map(int, hlens[idx])) for idx in range(self.num_encs)]

        vy = ys[:, -1]
        ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim

        # dual-attention
        enc_att_c, enc_att_w = self.att_list[0](
            self.enc_enc, self.hlens[0], self.dropout_dec[0](batch_z_list[0]), batch_enc_a_prev,
        )
        if self.use_first_pass:
            dec_att_c, dec_att_w = self.att_list[1](
                self.hyp_enc, self.hlens[1], self.dropout_dec[0](batch_z_list[0]), batch_hyp_a_prev,
            )
            att_c = torch.cat([enc_att_c, dec_att_c], dim=1)
        else:
            dec_att_w = enc_att_c.new_zeros((n_batch, 1)) # dummy
            att_c = enc_att_c

        # rnn forward
        ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
        batch_z_list, batch_c_list = self.rnn_forward(
            ey, batch_z_list, batch_c_list, batch_z_list, batch_c_list
        )
        if self.context_residual:
            logits = self.output(
                torch.cat((self.dropout_dec[-1](batch_z_list[-1]), att_c), dim=-1)
            )
        else:
            logits = self.output(self.dropout_dec[-1](batch_z_list[-1]))
        logp = F.log_softmax(logits, dim=1)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[dict(c_prev=batch_c_list[i][b],
                            z_prev=batch_z_list[i][b],
                            enc_a_prev=enc_att_w[b] if i == 0 else None,
                            hyp_a_prev=dec_att_w[b] if i == 0 else None)
                       for i in range(self.dlayers)] for b in range(n_batch)]

        return (
            logp,
            state_list,
        )
