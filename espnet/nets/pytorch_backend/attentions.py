import math
import six

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device
from torch import nn
import logging
import numpy as np


class Energy(nn.Module):
    def __init__(self, enc_dim=10, dec_dim=10, att_dim=10, init_r=-4, is_monotonic_energy=False):
        """
        [Modified Bahdahnau attention] from
        "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
        http://arxiv.org/abs/1704.00784

        Used for Monotonic Attention and Chunk Attention
        """
        super().__init__()
        self.tanh = nn.Tanh()
        self.W = nn.Linear(enc_dim, att_dim, bias=False)
        self.V = nn.Linear(dec_dim, att_dim, bias=False)
        self.b = nn.Parameter(torch.Tensor(att_dim).normal_())
        self.is_monotonic_energy = is_monotonic_energy
        if self.is_monotonic_energy:
            self.v = nn.utils.weight_norm(nn.Linear(att_dim, 1))
            self.v.weight_g.data = torch.Tensor([1 / att_dim]).sqrt()
            self.r = nn.Parameter(torch.Tensor([init_r]))
        else:
            self.v = nn.Linear(att_dim, 1)

    def forward(self, encoder_outputs, decoder_h):
        """
        Args:
            encoder_outputs: [batch_size, sequence_length, enc_dim]
            decoder_h: [batch_size, dec_dim]
        Return:
            Energy [batch_size, sequence_length]
        """
        batch_size, sequence_length, enc_dim = encoder_outputs.size()
        encoder_outputs = encoder_outputs.view(-1, enc_dim)
        energy = self.tanh(self.W(encoder_outputs) +
                           self.V(decoder_h).repeat(1, sequence_length).reshape(batch_size * sequence_length, -1) +
                           self.b)
        if self.is_monotonic_energy:
            energy = self.v(energy).squeeze(-1) + self.r
        else:
            energy = self.v(energy).squeeze(-1)

        return energy.view(batch_size, sequence_length)


class MonotonicAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, att_dim):
        """
        [Monotonic Attention] from
        "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
        http://arxiv.org/abs/1704.00784
        """
        super().__init__()

        self.monotonic_energy = Energy(enc_dim, dec_dim, att_dim, is_monotonic_energy=True)
        self.sigmoid = nn.Sigmoid()

    def gaussian_noise(self, *size):
        """Additive gaussian nosie to encourage discreteness"""
        if torch.cuda.is_available():
            return torch.cuda.FloatTensor(*size).normal_()
        else:
            return torch.Tensor(*size).normal_()

    def safe_cumprod_old(self, x):
        """Numerically stable cumulative product by cumulative sum in log-space"""
        return torch.exp(torch.cumsum(torch.log(torch.clamp(x, min=1e-10, max=1)), dim=1))

    def safe_cumprod(self, x, exclusive=False):
        """Numerically stable cumulative product by cumulative sum in log-space"""
        bsz = x.size(0)
        logsum = torch.cumsum(torch.log(torch.clamp(x, min=1e-10, max=1)), dim=1)
        if exclusive:
            logsum = torch.cat([torch.zeros(bsz, 1).to(logsum), logsum], dim=1)[:, :-1]
        return torch.exp(logsum)

    def exclusive_cumprod(self, x):
        """Exclusive cumulative product [a, b, c] => [1, a, a * b]
        * TensorFlow: https://www.tensorflow.org/api_docs/python/tf/cumprod
        * PyTorch: https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614
        """
        batch_size, sequence_length = x.size()
        if torch.cuda.is_available():
            one_x = torch.cat([torch.ones(batch_size, 1).cuda(), x], dim=1)[:, :-1]
        else:
            one_x = torch.cat([torch.ones(batch_size, 1), x], dim=1)[:, :-1]
        return torch.cumprod(one_x, dim=1)

    def soft(self, encoder_outputs, decoder_h, previous_alpha=None, simple=False):
        """
        Soft monotonic attention (Train)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_alpha [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
        """
        batch_size, sequence_length, enc_dim = encoder_outputs.size()

        monotonic_energy = self.monotonic_energy(encoder_outputs, decoder_h)
        p_select = self.sigmoid(monotonic_energy + self.gaussian_noise(monotonic_energy.size()))
        cumprod_1_minus_p = self.safe_cumprod(1 - p_select, exclusive=True)

        if previous_alpha is None:
            # First iteration => alpha = [1, 0, 0 ... 0]
            previous_alpha = torch.zeros(batch_size, sequence_length)
            previous_alpha[:, 0] = torch.ones(batch_size)
            if torch.cuda.is_available():
                previous_alpha = previous_alpha.cuda()

        if simple is True:
            # logging.info('## using simplified form ##')
            alpha = p_select * cumprod_1_minus_p
        else:
            alpha = p_select * cumprod_1_minus_p * \
                    torch.cumsum(previous_alpha / torch.clamp(cumprod_1_minus_p, min=1e-10), dim=1)
        # logging.debug('## mono attention p_select:\n {}\n cumprod:\n {}\n alpha:\n {} ##'.format(p_select, torch.clamp(cumprod_1_minus_p, min=1e-10), alpha))
        # logging.debug('## mono attention and indices:\n {} ##'.format(alpha.max(1)))
        return alpha

    def hard(self, encoder_outputs, decoder_h, previous_attention=None):
        """
        Hard monotonic attention (Test)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_attention [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
        """
        batch_size, sequence_length, enc_dim = encoder_outputs.size()

        if previous_attention is None:
            # First iteration => alpha = [1, 0, 0 ... 0]
            previous_attention = torch.zeros(batch_size, sequence_length)
            previous_attention[:, 0] = torch.ones(batch_size)
            if torch.cuda.is_available():
                previous_attention = previous_attention.cuda()
        if True:
            # TODO: Linear Time Decoding
            # It's not clear if authors' TF implementation decodes in linear time.
            # https://github.com/craffel/mad/blob/master/example_decoder.py#L235
            # They calculate energies for whole encoder outputs
            # instead of scanning from previous attended encoder output.
            monotonic_energy = self.monotonic_energy(encoder_outputs, decoder_h)

            # Hard Sigmoid
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
            above_threshold = (monotonic_energy > -3.75).float()

            p_select = above_threshold * torch.cumsum(previous_attention, dim=1)
            attention = p_select * self.exclusive_cumprod(1 - p_select)

            # Not attended => attend at last encoder output
            # Assume that encoder outputs are not padded
            not_attend = []
            attended = attention.sum(dim=1)
            for batch_i in range(batch_size):
                if not attended[batch_i]:
                    # logging.info('## monotonic hard attention: not attend ##')
                    not_attend.append(batch_i)
                    attention[batch_i, -1] = 1
            # logging.info('## monotonic hard attention: energy {}\n, previous_att {}\n, att {}\n ##'.format(monotonic_energy, previous_attention, attention))
            # Ex)
            # p_select                        = [0, 0, 0, 1, 1, 0, 1, 1]
            # 1 - p_select                    = [1, 1, 1, 0, 0, 1, 0, 0]
            # exclusive_cumprod(1 - p_select) = [1, 1, 1, 1, 0, 0, 0, 0]
            # attention: product of above     = [0, 0, 0, 1, 0, 0, 0, 0]
        return attention, not_attend


class MoChA(MonotonicAttention):
    def __init__(self, enc_dim, dec_dim, att_dim, chunk_size=3):
        """
        [Monotonic Chunkwise Attention] from
        "Monotonic Chunkwise Attention" (ICLR 2018)
        https://openreview.net/forum?id=Hko85plCW
        """
        super().__init__(enc_dim, dec_dim, att_dim)
        self.chunk_size = chunk_size
        self.chunk_energy = Energy(enc_dim, dec_dim, att_dim)
        self.softmax = nn.Softmax(dim=1)

    def moving_max(self, x, window=2):
        x_padded = F.pad(x, pad=[window - 1, 0], value=-np.inf)
        x_padded = x_padded.unsqueeze(1)
        results = F.max_pool1d(x_padded, window, stride=1)
        return results[:, 0, :]

    def frame(self, x, window, pad_value=-np.inf):
        if window >= 0:
            x_padded = F.pad(x, pad=[window - 1, 0], value=pad_value)
        else:
            window = -window
            x_padded = F.pad(x, pad=[0, window - 1], value=pad_value)
        n, l = x.shape
        results = torch.zeros((n, window, l))
        if torch.cuda.is_available():
            results = results.cuda()
        for i in range(window):
            results[:, i, :] = x_padded[:, i:l + i]
        return results

    def moving_sum(self, x, back, forward):
        """Parallel moving sum with 1D Convolution"""
        # Pad window before applying convolution
        # [batch_size,    back + sequence_length + forward]
        x_padded = F.pad(x, pad=[back, forward])

        # Fake channel dimension for conv1d
        # [batch_size, 1, back + sequence_length + forward]
        x_padded = x_padded.unsqueeze(1)

        # Apply conv1d with filter of all ones for moving sum
        filters = torch.ones(1, 1, back + forward + 1)
        if torch.cuda.is_available():
            filters = filters.cuda()
        x_sum = F.conv1d(x_padded, filters)

        # Remove fake channel dimension
        # [batch_size, sequence_length]
        return x_sum.squeeze(1)

    def chunkwise_attention_soft(self, alpha, u):
        """
        Args:
            alpha [batch_size, sequence_length]: emission probability in monotonic attention
            u [batch_size, sequence_length]: chunk energy
            chunk_size (int): window size of chunk
        Return
            beta [batch_size, sequence_length]: MoChA weights
        """

        # Numerical stability
        # Divide by same exponent => doesn't affect softmax
        u -= torch.max(u, dim=1, keepdim=True)[0]
        exp_u = torch.exp(u)
        # Limit range of logit
        exp_u = torch.clamp(exp_u, min=1e-5)

        # Moving sum:
        # Zero-pad (chunk size - 1) on the left + 1D conv with filters of 1s.
        # [batch_size, sequence_length]
        denominators = self.moving_sum(exp_u,
                                       back=self.chunk_size - 1, forward=0)

        # Compute beta (MoChA weights)
        beta = exp_u * self.moving_sum(alpha / denominators,
                                       back=0, forward=self.chunk_size - 1)
        return beta

    def stable_chunkwise_attention_soft(self, emit_probs, softmax_logits):
        """Compute chunkwise attention distribution stably by subtracting logit max."""
        # Shift logits to avoid overflow    
        logits_max = self.moving_max(softmax_logits, self.chunk_size)
        framed_logits = self.frame(softmax_logits, self.chunk_size)
        
        # Normalize each logit subsequence by the max in that subsequence
        framed_logits -= logits_max.unsqueeze(1)
        
        # Compute softmax denominators (d)
        softmax_denominators = torch.sum(torch.exp(framed_logits), 1)
        
        # Construct matrix of framed denominators, padding at the end so the final
        # frame is [softmax_denominators[-1], inf, inf, ..., inf] (E)
        framed_denominators = self.frame(softmax_denominators, -self.chunk_size, pad_value=np.inf)
        
        # Create matrix of copied logits so that column j is softmax_logits[j] copied
        # chunk_size times (N)
        copied_logits = softmax_logits.unsqueeze(1).repeat(1, self.chunk_size, 1)
        
        # Subtract the max over subsequences(M) from each logit
        framed_max = self.frame(logits_max, -self.chunk_size, pad_value=np.inf)
        copied_logits -= framed_max
        
        # Take exp() to get softmax numerators
        softmax_numerators = torch.exp(copied_logits)

        # Create matrix with length-chunk_size frames of emit_probs, padded so that
        # the last frame is [emit_probs[-1], 0, 0, ..., 0] (A)
        framed_probs = self.frame(emit_probs, -self.chunk_size, pad_value=0)
        
        beta = torch.sum(framed_probs * softmax_numerators / framed_denominators, 1)
        zeros = (softmax_denominators == 0).nonzero().sum()
        logging.warning('## stabel chunkwise attention checking zero : {} ##'.format(zeros))
        logging.warning('## stabel chunkwise attention checking  softmax_denominators {}\n framed loggits {}\n softmax_logits {}\n##'.format(softmax_denominators, framed_logits, softmax_logits))
        # logging.warning('## stabel chunkwise attention ##\n \
        # framed_logits {}\n softmax_denominators {}\n framed_denominators {}\n \
            # framed_probs {}\n beta {}'.format(framed_logits, softmax_denominators, framed_denominators, framed_probs, beta))
        return beta

    def chunkwise_attention_hard(self, monotonic_attention, chunk_energy):
        """
        Mask non-attended area with '-inf'
        Args:
            monotonic_attention [batch_size, sequence_length]
            chunk_energy [batch_size, sequence_length]
        Return:
            masked_energy [batch_size, sequence_length]
        """
        batch_size, sequence_length = monotonic_attention.size()

        # [batch_size]
        attended_indices = monotonic_attention.nonzero().cpu().data[:, 1].tolist()
        # logging.info('## chunkwise attention hard: attended indices {} ##'.format(attended_indices))
        i = [[], []]
        total_i = 0
        # logging.warning('chunk size {}'.format(self.chunk_size))
        for batch_i, attended_idx in enumerate(attended_indices):
            for window in range(self.chunk_size):
                if attended_idx - window >= 0:
                    i[0].append(batch_i)
                    i[1].append(attended_idx - window)
                    total_i += 1
            # for window in range(1, 2):
            #     if attended_idx + window < sequence_length:
            #         i[0].append(batch_i)
            #         i[1].append(attended_idx + window)
            #         total_i += 1
        i = torch.LongTensor(i)
        v = torch.FloatTensor([1] * total_i)
        mask = torch.sparse.FloatTensor(i, v, monotonic_attention.size())
        if torch.cuda.is_available():
            mask = ~mask.to_dense().cuda().bool()
        else:
            mask = ~mask.to_dense().bool()

        # mask '-inf' energy before softmax
        masked_energy = chunk_energy.masked_fill_(mask, -float('inf'))
        return masked_energy

    def soft(self, encoder_outputs, decoder_h, previous_alpha=None, mask=None, simple=False):
        """
        Soft monotonic chunkwise attention (Train)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_alpha [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
            beta [batch_size, sequence_length]
        """
        alpha = super().soft(encoder_outputs, decoder_h, previous_alpha, simple)
        chunk_energy = self.chunk_energy(encoder_outputs, decoder_h)
        if mask is not None:
            chunk_energy.masked_fill_(mask, -float('inf'))
        beta = self.chunkwise_attention_soft(alpha, chunk_energy)
        return alpha, beta

    def real_hard(self, encoder_outputs, decoder_h, previous_attention=None):
        assert encoder_outputs.shape[0] == 1 and decoder_h.shape[0] == 1, "support one utterance only"
        if previous_attention is None:
            # First iteration => alpha = [1, 0, 0 ... 0]
            previous_attention = torch.zeros(batch_size, sequence_length)
            previous_attention[:, 0] = torch.ones(batch_size)
            if torch.cuda.is_available():
                previous_attention = previous_attention.cuda()
            
            # TODO: Linear Time Decoding
            # It's not clear if authors' TF implementation decodes in linear time.
            # https://github.com/craffel/mad/blob/master/example_decoder.py#L235
            # They calculate energies for whole encoder outputs
            # instead of scanning from previous attended encoder output.
        monotonic_energy = self.monotonic_energy(encoder_outputs, decoder_h)

            # Hard Sigmoid
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
        above_threshold = (monotonic_energy > 0).float()

        p_select = above_threshold * torch.cumsum(previous_attention, dim=1)
        attention = p_select * self.exclusive_cumprod(1 - p_select)


    def hard(self, encoder_outputs, decoder_h, previous_attention=None):
        """
        Hard monotonic chunkwise attention (Test)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_attention [batch_size, sequence_length]
        Return:
            monotonic_attention [batch_size, sequence_length]: hard alpha
            chunkwise_attention [batch_size, sequence_length]: hard beta
        """
        # hard attention (one-hot)
        # [batch_size, sequence_length]
        monotonic_attention, not_attend = super().hard(encoder_outputs, decoder_h, previous_attention)
        chunk_energy = self.chunk_energy(encoder_outputs, decoder_h)
        masked_energy = self.chunkwise_attention_hard(monotonic_attention, chunk_energy)
        chunkwise_attention = self.softmax(masked_energy)
        # for i in not_attend:
            # chunkwise_attention[i] = torch.zeros_like(chunkwise_attention[i])
        return monotonic_attention, chunkwise_attention


def moving_sum(x, back, forward):
    """Compute the moving sum of x over a window with the provided bounds.
    x is expected to be of shape (B x T_max).
    The returned tensor x_sum is computed as
    x_sum[i, j] = x[i, j - back] + ... + x[i, j + forward]
    """
    x_pad = F.pad(x, (back,forward,0,0), "constant", 0.0)
    kernel = torch.ones(1,1,back+forward+1, dtype=x.dtype)
    if torch.cuda.is_available() and x.is_cuda:
        kernel = kernel.cuda()
    return F.conv1d(x_pad.unsqueeze(1), kernel).squeeze(1)


def exclusive_cumprod(x):
    """Exclusive cumulative product [a, b, c] => [1, a, a * b]
    * TensorFlow: https://www.tensorflow.org/api_docs/python/tf/cumprod
    * PyTorch: https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614
    """
    batch_size, sequence_length = x.size()
    if torch.cuda.is_available() and x.is_cuda:
        one_x = torch.cat([torch.ones(batch_size, 1).cuda(), x], dim=1)[:, :-1]
    else:
        one_x = torch.cat([torch.ones(batch_size, 1), x], dim=1)[:, :-1]
    return torch.cumprod(one_x, dim=1)


def safe_cumprod(x, *args, **kwargs):
    """Computes cumprod of x in logspace using cumsum to avoid underflow.
    The cumprod function and its gradient can result in numerical instabilities
    when its argument has very small and/or zero values.  As long as the argument
    is all positive, we can instead compute the cumulative product as
    exp(cumsum(log(x))).  This function can be called identically to tf.cumprod.
    Args:
        x: Tensor to take the cumulative product of.
        *args: Passed on to cumsum; these are identical to those in cumprod.
        **kwargs: Passed on to cumsum; these are identical to those in cumprod.
    Returns:
        Cumulative product of x, the first element is 1.
    """
    cumprod = torch.exp(torch.cumsum(torch.log(torch.clamp(x[:, :-1], 1e-10, 1.)), *args, **kwargs))
    exclusive_cumprod = cumprod.new_ones(x.shape)
    exclusive_cumprod[:, 1:] = cumprod
    return exclusive_cumprod


def consume_ctc_stamp(ctc_stamps, ctc_idx, mocha_stamp, last_stamp, tolerance=2, offset=0):
    while ctc_idx < len(ctc_stamps) and last_stamp + offset >= ctc_stamps[ctc_idx]:
        ctc_idx += 1
    curr_ctc = ctc_stamps[ctc_idx] if ctc_idx < len(ctc_stamps) else None
    curr_mocha = mocha_stamp if mocha_stamp is not None else (curr_ctc + tolerance + 1 if curr_ctc is not None else None)
    new_ctc_idx = ctc_idx
    stamp = -1
    if mocha_stamp is not None or ctc_idx < len(ctc_stamps):
        if ctc_idx < len(ctc_stamps) and ((curr_mocha - curr_ctc) > tolerance or (curr_mocha is None)):
            stamp = curr_ctc + offset
            new_ctc_idx = ctc_idx + 1
        elif ctc_idx >= len(ctc_stamps) or (curr_ctc - curr_mocha) > tolerance:
            stamp = curr_mocha
        else:
            stamp = curr_mocha
            new_ctc_idx = ctc_idx + 1
        if stamp != curr_mocha:
            logging.info(
                f"ctc_trigger_point {curr_ctc if ctc_idx < len(ctc_stamps) else None},"
                f"last_attend_idx {last_stamp}, mocha_attend_idx {curr_mocha}, ret {stamp}")

    return stamp, new_ctc_idx


class AttMoChAstable(nn.Module):
    '''Monotonic chunkwise attention dropping prev attention distribution
        which is slightly different from Google's formulation but more stable during training
    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int win: chunk width for MoChA
    :param float scaling: scaling parameter before applying softmax
    :param float sigmoid_noise: Standard deviation of pre-sigmoid noise.
    :param float score_bias_init: Initial value for score bias scalar.
                                  It's recommended to initialize this to a negative value
                                  (e.g. -4.0) when the length of the memory is large.'''
    def __init__(self, eprojs, dunits, att_dim, att_win,
                 sigmoid_noise=1.0, score_bias_init=-4.0):
        super(AttMoChAstable, self).__init__()
        
        self.monotonic_mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.monotonic_mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.monotonic_gvec = torch.nn.Linear(att_dim, 1, bias=False)
        # don't forget to initialize this to 1.0 / math.sqrt(att_dim)
        self.monotonic_factor = torch.nn.Parameter(torch.Tensor(1, 1).fill_(1.0 / math.sqrt(att_dim)))
        # don't forget to initialize this to a negative value (e.g. -4.0)
        self.monotonic_bias = torch.nn.Parameter(torch.Tensor(1, 1).fill_(score_bias_init))
        assert att_win > 0
        if att_win > 1:  # Hard Monotonic Attention for att_win = 1
            self.chunk_mlp_enc = torch.nn.Linear(eprojs, att_dim)
            self.chunk_mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
            self.chunk_gvec = torch.nn.Linear(att_dim, 1, bias=False)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.att_win = att_win
        self.sigmoid_noise = sigmoid_noise
        self.score_bias_init = score_bias_init
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.pre_compute_chunk_enc_h = None

    def reset(self, trigger_point=None):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.pre_compute_chunk_enc_h = None
        self.mask = None
        self.last_offset = 0
        self.trigger_point = trigger_point

    def batch_stream(self, dec_z, dec_z_tiled, aw_prev, scaling=2.0, offset=2, ctc_index=0):
        '''MoChA forward in online scenario, only support one utterance
                :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
                :param list enc_h_len: padded encoder hidden state lenght (B)
                :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
                :param int aw_prev: previous end-point of MoChA
                :param int offset: the first index of new coming encoder hidden states
                                   designed for streaming encoder
                :return: attentioin weighted encoder state (B, D_enc)
                :rtype: torch.Tensor
                :return: previous end-point (B x T_max)
                :rtype: torch.Tensor
                '''
        batch = len(dec_z_tiled)
        # utt x frame x att_dim -> utt x frame
        e = self.monotonic_factor / torch.norm(self.monotonic_gvec.weight, p=2) \
            * self.monotonic_gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled)).squeeze(2) \
            + self.monotonic_bias

        if aw_prev is None:
            aw_prev = e.new_zeros(batch, self.enc_h.shape[1])
            aw_prev[:, 0] = 1
        p_choose_i = (e > -2.0).float()
        prev_choice = torch.cumsum(aw_prev, dim=-1)
        p_choose_i *= prev_choice
        alpha = p_choose_i * exclusive_cumprod(1 - p_choose_i)
        mocha_not_attend = torch.where(alpha.sum(dim=1) < 1)[0].tolist()
        alpha_prev = torch.where(aw_prev > 0.9)[1].tolist()
        mocha_attend_idx = torch.where(alpha > 0.9)[1].tolist()
        for idx in mocha_not_attend:
            mocha_attend_idx.insert(idx, None)
        assert len(mocha_attend_idx) == len(ctc_index) == len(alpha_prev), f"{len(ctc_index)}, {len(mocha_attend_idx)}, {len(alpha_prev)}"
        tolerance = 6
        offset = 2
        attend_ret = [consume_ctc_stamp(self.trigger_point,
                                        ctc_index[b],
                                        mocha_attend_idx[b], alpha_prev[b],
                                        tolerance=tolerance,
                                        offset=offset) for b in range(batch)]
        attend_ids = [min(x[0], self.enc_h.shape[1] -1) for x in attend_ret]
        new_ctc_idx = [x[1] for x in attend_ret]
        mask = dec_z.new_ones(batch, self.enc_h.shape[1]).bool()
        not_attend = []
        alpha = dec_z.new_zeros(batch, self.enc_h.shape[1])
        for i, stop_id in enumerate(attend_ids):
            if stop_id < 0:
                not_attend.append(i)
                alpha[i, -1] = 1.0
            else:
                alpha[i, stop_id] = 1.0
            start_id = max(0, stop_id - self.att_win + 1)
            mask[i, start_id: stop_id + 1] = False

        if self.att_win == 1:
            c = self.enc_h[:, attend_ids]
        else:
            # dec_z_chunk_tiled: utt x frame x att_dim
            dec_z_chunk_tiled = self.chunk_mlp_dec(dec_z).view(batch, 1, self.att_dim)
            # dot with gvec
            # utt x frame x att_dim -> utt x frame
            u = self.chunk_gvec(
                    torch.tanh(self.pre_compute_chunk_enc_h + dec_z_chunk_tiled)).squeeze(2)
            u = u.masked_fill_(mask, -float("inf"));

            w = F.softmax(scaling * u, dim=1)
            c = torch.sum(self.enc_h * w.view(batch, self.enc_h.shape[1] , 1), dim=1)

        if not_attend:
            c[not_attend] = 0.0

        return c, new_ctc_idx, alpha


    def stream(self, enc_hs_pad, enc_hs_len, dec_z, end_point, scaling=2.0, offset=2, order=1, ctc_index=0):
        '''MoChA forward in online scenario, only support one utterance
        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param int end_point: previous end-point of MoChA
        :param int offset: the first index of new coming encoder hidden states
                           designed for streaming encoder
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous end-point (B x T_max)
        :rtype: torch.Tensor
        '''
        assert len(enc_hs_pad) == 1
        batch = 1
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad
            # self.h_length += self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.monotonic_mlp_enc(self.enc_h)
            if self.att_win > 1:
                self.pre_compute_chunk_enc_h = self.chunk_mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.monotonic_mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # utt x frame x att_dim -> utt x frame
        e = self.monotonic_factor / torch.norm(self.monotonic_gvec.weight, p=2) \
            * self.monotonic_gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled)).squeeze(2) \
            + self.monotonic_bias

        last_stamp = end_point
        if end_point is None:
            end_point = 0
            last_stamp = -1
        flag = False
        for z in range(end_point, e.size(1)):
            if e[0, z] > -2.0:
                flag = True
                break

        tolerance = 6
        offset = 2
        if isinstance(ctc_index, list):
            ctc_index = ctc_index[0]
        while ctc_index < len(self.trigger_point) and last_stamp + offset >= self.trigger_point[ctc_index]:
            ctc_index += 1
            # logging.info("### increase ctc index by 1 ###")
        new_ctc_index = ctc_index
        curr_ctc = self.trigger_point[ctc_index] if ctc_index < len(self.trigger_point) else None
        if flag or ctc_index < len(self.trigger_point):
            if ctc_index < len(self.trigger_point) and ((z - curr_ctc) > tolerance or (not flag)):
                end_point = curr_ctc + offset
                new_ctc_index = ctc_index + 1
            elif ctc_index >= len(self.trigger_point) or (curr_ctc - z) > tolerance:
                end_point = z
            else:
                end_point = z
                new_ctc_index = ctc_index + 1
            if end_point != z:
                logging.info(
                    f"### ctc_curr {curr_ctc if ctc_index < len(self.trigger_point) else None} last_stamp {last_stamp}, mocha stamp {z},  ret {end_point}")
        # if flag:
            end_point = min(end_point, e.shape[1] - 1)
            # end_point = z
            order = min(order, max(end_point - self.att_win + 2, 1))
            chunk_length = min(self.att_win + order - 1, end_point + 1)
            chunk_scores = torch.sigmoid(e[0, end_point + 1 - order: end_point + 1])
            chunk_scores /= chunk_scores.sum()
            if self.att_win == 1:
                c = self.enc_h[:, end_point]
                # todo: implement high order mode
            else:
                # dec_z_chunk_tiled: utt x frame x att_dim
                dec_z_chunk_tiled = self.chunk_mlp_dec(dec_z).view(batch, 1, self.att_dim)
                # dot with gvec
                # utt x frame x att_dim -> utt x frame
                w = dec_z_chunk_tiled.new_zeros(batch, chunk_length)
                for i, weight in enumerate(torch.flip(chunk_scores, (0,))):
                    start_point = max(0, end_point - self.att_win + 1 - i)
                    u = self.chunk_gvec(torch.tanh(
                        self.pre_compute_chunk_enc_h[:, start_point:end_point + 1 - i] + dec_z_chunk_tiled)).squeeze(2)
                    w_ = F.softmax(scaling * u, dim=1) * weight
                    w[:, order - i - 1: order - i - 1 + self.att_win] += w_
                start_point = max(0, end_point - self.att_win - order + 2)
                c = torch.sum(self.enc_h[:, start_point:end_point + 1] * w.view(batch, w.shape[1], 1), dim=1)
        else:
            w = enc_hs_pad.new_zeros(batch, len(enc_hs_pad))
            c = enc_hs_pad.new_zeros(batch, self.eprojs)
            # end_point = None
        return c, new_ctc_index, end_point

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0, **kwargs):
        '''MoChA forward
        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attetion weight (B x T_max)
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous a (B x T_max)
        :rtype: torch.Tensor
        '''
        # if not kwargs.get('soft', True):
        #     return self.stream(enc_hs_pad, enc_hs_len, dec_z, att_prev, ctc_index=kwargs.get("ctc_index", None))
        batch = len(dec_z)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.monotonic_mlp_enc(self.enc_h)
            if self.att_win > 1:
                self.pre_compute_chunk_enc_h = self.chunk_mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)
        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.monotonic_mlp_dec(dec_z).view(batch, 1, self.att_dim)

        if not self.trigger_point is None and not kwargs.get('soft', True):
            return self.batch_stream(dec_z, dec_z_tiled, att_prev, ctc_index=kwargs.get("ctc_index", None))

        if att_prev is None:
            att_prev = enc_hs_pad.new_zeros(batch, self.h_length)
            att_prev[:, 0] = 1.0  # initialize attention weights

        # Implements additive energy function to compute pre-sigmoid activation e.
        # Sigmoid is used to compute selection probability p, than its expectation value a.
        # To mitigate saturating and sensitivity to offset,
        # monotonic_factor and monotonic_bias are added here as learnable scalars
        # utt x frame x att_dim -> utt x frame
        e = self.monotonic_factor / torch.norm(self.monotonic_gvec.weight, p=2) \
            * self.monotonic_gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled)).squeeze(2) \
            + self.monotonic_bias

        # NOTE consider zero padding when compute p and a
        # a: utt x frame
        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len, xs=e))
        e.masked_fill_(self.mask, -float('inf'))
        # Optionally add pre-sigmoid noise to the scores
        e += self.sigmoid_noise * to_device(self, torch.normal(mean=torch.zeros(e.shape), std=1))
        p = torch.sigmoid(e)
        # safe_cumprod computes cumprod in logspace with numeric checks
        cumprod_1mp = safe_cumprod(1 - p, dim=1)
        # Google's formulation:
        # a = p * cumprod_1mp * torch.cumsum(
        #    att_prev / torch.clamp(cumprod_1mp, 1e-10, 1.), dim=1)
        # or an approximation:
        # a = p * cumprod_1mp * torch.cumsum(att_prev, dim=1)
        # Stable MoChA:
        a = p * cumprod_1mp

        if self.att_win == 1:
            w = a.masked_fill(self.mask, 0)
        else:
            # dec_z_chunk_tiled: utt x frame x att_dim
            dec_z_chunk_tiled = self.chunk_mlp_dec(dec_z).view(batch, 1, self.att_dim)
            # dot with gvec
            # utt x frame x att_dim -> utt x frame
            u = self.chunk_gvec(torch.tanh(self.pre_compute_chunk_enc_h + dec_z_chunk_tiled)).squeeze(2)

            # NOTE consider zero padding when compute w.
            u.masked_fill_(self.mask, -float('inf'))
            exp_u = torch.exp(u * scaling)
            w = exp_u * moving_sum(a / torch.clamp(moving_sum(exp_u, self.att_win - 1, 0),
                                                   1e-10, float('inf')), 0, self.att_win - 1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
        return c, a

class AttMoChA(torch.nn.Module):
    """Dot product attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    """

    def __init__(self, eprojs, dunits, att_dim, chunk_size=10, simple=False):
        super(AttMoChA, self).__init__()
        # self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        # self.mlp_dec = torch.nn.Linear(dunits, att_dim)

        self.mocha = MoChA(enc_dim=eprojs, dec_dim=dunits, att_dim=eprojs, chunk_size=chunk_size)
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.pre_alpha = None
        self.mask = None
        self.simple_alpha = simple

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.pre_alpha = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0, soft=True):
        """AttDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weight (B x T_max)
        :rtype: torch.Tensor
        """
        batch = enc_hs_pad.size(0)
        self.h_length = enc_hs_pad.size(1)
        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))

        # logging.info('## MoChA before hard attention att_prev:\n {}\n ##'.format(att_prev))
        self.enc_h = enc_hs_pad
        if soft is True:
            self.pre_alpha, beta = self.mocha.soft(enc_hs_pad, dec_z, self.pre_alpha, self.mask, self.simple_alpha)
        else:
            att_prev, beta = self.mocha.hard(enc_hs_pad, dec_z, att_prev)
        # logging.info('## MoChA att:\n {}\n beta: \n{} \n##'.format(att_prev, beta))
        # # pre-compute all h outside the decoder loop
        # if self.pre_compute_enc_h is None:
        #     self.enc_h = enc_hs_pad  # utt x frame x hdim
        #     self.h_length = self.enc_h.size(1)
        #     # utt x frame x att_dim
        #     self.pre_compute_enc_h = torch.tanh(self.mlp_enc(self.enc_h))

        # e = torch.sum(self.pre_compute_enc_h * torch.tanh(self.mlp_dec(dec_z)).view(batch, 1, self.att_dim),
        #               dim=2)  # utt x frame

        # # NOTE consider zero padding when compute w.
        # e.masked_fill_(self.mask, -float('inf'))
        # w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * beta.view(batch, self.h_length, 1), dim=1)
        if soft is True:
            return c, beta
        else:
            return c, beta, att_prev


class AttProjMoChA(torch.nn.Module):
    """Dot product attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    """

    def __init__(self, eprojs, dunits, att_dim, chunk_size=3, simple=False):
        super(AttProjMoChA, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, eprojs)
        # self.mlp_dec = torch.nn.Linear(dunits, att_dim)

        self.mocha = MoChA(enc_dim=eprojs, dec_dim=dunits, att_dim=att_dim, chunk_size=chunk_size)
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.pre_alpha = None
        self.mask = None
        self.simple_alpha = simple

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.pre_alpha = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0, soft=True):
        """AttDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weight (B x T_max)
        :rtype: torch.Tensor
        """
        batch = enc_hs_pad.size(0)
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = torch.tanh(self.mlp_enc(self.enc_h))
        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))

        # logging.info('## MoChA before hard attention att_prev:\n {}\n ##'.format(att_prev))
        if soft is True:
            self.pre_alpha, beta = self.mocha.soft(self.pre_compute_enc_h, dec_z,
                                                   self.pre_alpha, self.mask, self.simple_alpha)
        else:
            att_prev, beta = self.mocha.hard(self.pre_compute_enc_h, dec_z, att_prev)
        # logging.info('## MoChA att:\n {}\n beta: \n{} \n##'.format(att_prev, beta))
        # # pre-compute all h outside the decoder loop

        # e = torch.sum(self.pre_compute_enc_h * torch.tanh(self.mlp_dec(dec_z)).view(batch, 1, self.att_dim),
        #               dim=2)  # utt x frame

        # # NOTE consider zero padding when compute w.
        # e.masked_fill_(self.mask, -float('inf'))
        # w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * beta.view(batch, self.h_length, 1), dim=1)
        if soft is True:
            return c, beta
        else:
            return c, beta, att_prev


class AttConcatMultiHeadMoChA(torch.nn.Module):
    """Dot product attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    """

    def __init__(self, eprojs, dunits, att_dim, chunk_size=3, aheads=1):
        super(AttConcatMultiHeadMoChA, self).__init__()
        # logging.warning('## AttMultiHeadMoChA: aheads {}, chunk_size {}, eprojs {} ##'.format(aheads, chunk_size, eprojs))
        self.mocha = MoChA(enc_dim=att_dim, dec_dim=dunits, att_dim=att_dim, chunk_size=chunk_size)
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim * aheads)
        # self.mlp_o = torch.nn.Linear(aheads * att_dim, eprojs, bias=False)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.aheads = aheads
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.pre_alpha = None
        self.mask = None
        self.batch_hs_len = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.pre_alpha = None
        self.batch_hs_len = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0, soft=True):
        """AttDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weight (B x T_max)
        :rtype: torch.Tensor
        """
        batch = enc_hs_pad.size(0)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            self.batch_hs_len = enc_hs_len * self.aheads
            # utt x frame x (nheads x att_dim) -> (nheads x utt) x frame x attdim
            self.pre_compute_enc_h = self.split_and_concat(
                torch.tanh(self.mlp_enc(self.enc_h)), split_dim=2, concat_dim=0)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        dec_z = dec_z.repeat(self.aheads, 1)

        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(self.batch_hs_len))
        # logging.info('## MoChA before hard attention att_prev:\n {}\n ##'.format(att_prev))

        # TODO faster version
        if soft is True:
            self.pre_alpha, beta = self.mocha.soft(self.pre_compute_enc_h, dec_z, self.pre_alpha, self.mask)
        else:
            att_prev, beta = self.mocha.hard(self.pre_compute_enc_h, dec_z, att_prev)

        # pre_compute_enc_h (nhead x batch) x T x att_dim, beta (nhead x batch) x T
        # c (nhead x batch) x att_dim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.pre_compute_enc_h * beta.view(batch * self.aheads, self.h_length, 1), dim=1)

        # (nhead x batch) x att_dim -> batch x (nheads x att_dim)
        c = self.split_and_concat(c, split_dim=0, concat_dim=1)
        # logging.warning('## faster MultiMoChA, pre_h {} dec_z {} c {}##'.format(self.pre_compute_enc_h.shape, dec_z.shape, c.shape))

        # logging.warning('## beta\n{} c\n{} ##'.format(beta, c))
        # logging.info('## MoChA att:\n {}\n beta: \n{} \n##'.format(att_prev, beta))
        if soft is True:
            return c, beta.split(batch, 0)
        else:
            return c, beta.split(batch, 0), att_prev

    def split_and_concat(self, h, split_dim, concat_dim):
        x = torch.split(h, h.shape[split_dim] // self.aheads, split_dim)
        x = torch.cat(x, concat_dim)
        return x


class AttBatchMultiHeadMoChA(torch.nn.Module):
    """Dot product attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    """

    def __init__(self, eprojs, dunits, att_dim, chunk_size=3, aheads=1):
        super(AttBatchMultiHeadMoChA, self).__init__()
        # logging.warning('## AttMultiHeadMoChA: aheads {}, chunk_size {}, eprojs {} ##'.format(aheads, chunk_size, eprojs))
        self.mocha = MoChA(enc_dim=att_dim, dec_dim=dunits, att_dim=att_dim, chunk_size=chunk_size)
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim * aheads)
        # self.mlp_o = torch.nn.Linear(aheads * att_dim, eprojs, bias=False)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.aheads = aheads
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.pre_alpha = None
        self.mask = None
        self.batch_hs_len = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.pre_alpha = None
        self.batch_hs_len = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0, soft=True):
        """AttDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weight (B x T_max)
        :rtype: torch.Tensor
        """
        batch = enc_hs_pad.size(0)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            self.batch_hs_len = enc_hs_len * self.aheads
            # utt x frame x (nheads x att_dim) -> (nheads x utt) x frame x attdim
            self.pre_compute_enc_h = self.split_and_concat(
                torch.tanh(self.mlp_enc(self.enc_h)), split_dim=2, concat_dim=0)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        dec_z = dec_z.repeat(self.aheads, 1)

        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(self.batch_hs_len))
        # logging.info('## MoChA before hard attention att_prev:\n {}\n ##'.format(att_prev))

        # TODO faster version
        if soft is True:
            self.pre_alpha, beta = self.mocha.soft(self.pre_compute_enc_h, dec_z,
                                                   self.pre_alpha, self.mask, simple=True)
        else:
            att_prev, beta = self.mocha.hard(self.pre_compute_enc_h, dec_z, att_prev)
            # att_prev, beta = self.mocha.soft(self.pre_compute_enc_h, dec_z, att_prev)

        # pre_compute_enc_h (nhead x batch) x T x att_dim, beta (nhead x batch) x T
        # c (nhead x batch) x att_dim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.pre_compute_enc_h * beta.view(batch * self.aheads, self.h_length, 1), dim=1)

        # logging.warning('## faster MultiMoChA, pre_h {} dec_z {} c {}##'.format(self.pre_compute_enc_h.shape, dec_z.shape, c.shape))

        # logging.warning('## att\n{} ##'.format(att_prev)
        # logging.info('## MoChA att:\n {}\n beta: \n{} \n##'.format(att_prev, beta))
        # if soft is False:
        #     # (nhead x batch) x att_dim -> batch x (nheads x att_dim)
        #     logging.info('att\n {} \n beta\n {} \n c \n{}\n'.format(att_prev, beta, c))
        #     c = self.split_and_concat(c, split_dim=0, concat_dim=1)
        #     logging.info('c after split \n{}\n'.format(c))
        #     c = self.mlp_o(c)
        #     logging.info('c after map \n{}\n'.format(c))
        # else:
        c = torch.mean(c.reshape(self.aheads, batch, self.att_dim), 0)

        if soft is True:
            return c, beta.split(batch, 0)
        else:
            return c, beta.split(batch, 0), att_prev

    def split_and_concat(self, h, split_dim, concat_dim):
        x = torch.split(h, h.shape[split_dim] // self.aheads, split_dim)
        x = torch.cat(x, concat_dim)
        return x


class AttMultiHeadMoChA(torch.nn.Module):
    """Dot product attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    """

    def __init__(self, eprojs, dunits, att_dim, chunk_size=3, aheads=1, att2=True):
        super(AttMultiHeadMoChA, self).__init__()
        logging.warning(
            '## AttMultiHeadMoChA: aheads {}, chunk_size {}, eprojs {} ##'.format(aheads, chunk_size, eprojs))
        self.mocha_list = torch.nn.ModuleList()
        self.mlp_o = torch.nn.Linear(aheads * att_dim, eprojs, bias=False)
        self.mlp_enc = torch.nn.ModuleList()
        self.att2 = att2
        if att2:
            self.mlp_q = torch.nn.Linear(dunits, aheads * 4)
            self.mlp_k = torch.nn.Linear(eprojs * aheads, aheads * 4, bias=False)
            self.gvec = torch.nn.Linear(aheads * 4, aheads)
        for i in six.moves.range(aheads):
            self.mocha_list.append(MoChA(enc_dim=eprojs, dec_dim=dunits, att_dim=att_dim, chunk_size=chunk_size))
            self.mlp_enc.append(torch.nn.Linear(eprojs, eprojs))
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.aheads = aheads
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.pre_alpha = [None] * self.aheads
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.pre_alpha = [None] * self.aheads

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0, soft=True):
        """AttDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weight (B x T_max)
        :rtype: torch.Tensor
        """
        batch = enc_hs_pad.size(0)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = [torch.tanh(self.mlp_enc[i](self.enc_h)) for i in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))
        # logging.info('## MoChA before hard attention att_prev:\n {}\n ##'.format(att_prev))
        beta = []
        c = []
        c_s, c_h = [], []
        if soft is True:
            for i in six.moves.range(self.aheads):
                self.pre_alpha[i], b = self.mocha_list[i].soft(
                    self.pre_compute_enc_h[i], dec_z, self.pre_alpha[i], self.mask, simple=True)
                beta.append(b)
                c += [torch.sum(self.pre_compute_enc_h[i] * beta[i].view(batch, self.h_length, 1), dim=1)]
        else:
            if att_prev is None:
                att_prev = [None] * self.aheads
            for i in six.moves.range(self.aheads):
                # a, b = self.mocha_list[i].soft(self.pre_compute_enc_h[i], dec_z, att_prev[i])
                a, b = self.mocha_list[i].hard(self.pre_compute_enc_h[i], dec_z, att_prev[i])
                beta.append(b)
                att_prev[i] = a
                # weighted sum over flames
                # c_s += [torch.sum(self.pre_compute_enc_h[i] * b_s.view(batch, self.h_length, 1), dim=1)]
                c += [torch.sum(self.pre_compute_enc_h[i] * b.view(batch, self.h_length, 1), dim=1)]
            # logging.warning('## multi mocha att\n{}\n beta\n{}\n c\n{}\n'.format(att_prev, beta, c))
            # logging.warning('## multi mocha soft_c_final_ave\n{}\n hard_c_final_ave\n{}'.format(sum(c_s), sum(c_h)))
            # c_s = self.mlp_o(torch.cat(c_s, dim=1))
            # c_h = self.mlp_o(torch.cat(c_h, dim=1))

            # logging.warning('## multi mocha soft_c_final\n{}\n hard_c_final\n{}'.format(c_s, c_h))
        
        # attend to attended multihead info
        if self.att2:
            k = self.mlp_k(torch.cat(c, dim=1))  #batch x dim
            q = self.mlp_q(dec_z)  # batch x dim
            e = self.gvec(torch.tanh(k + q)).unsqueeze(2) # batch  x aheads x 1
            w = F.softmax(e, dim=1)
            c = torch.cat([c_.unsqueeze(1) for c_ in c], dim=1) #batch x aheads x eprojs
            c = torch.sum(c * w, dim=1)
            
        else:
            c = self.mlp_o(torch.cat(c, dim=1))

        # NOTE use bmm instead of sum(*)
        # concat all of c
        if soft is True:
            return c, beta
        else:
            return c, beta, att_prev


class NoAtt(torch.nn.Module):
    """No attention"""

    def __init__(self):
        super(NoAtt, self).__init__()
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        """NoAtt forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B, T_max, D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights
        :rtype: torch.Tensor
        """
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            # if no bias, 0 0-pad goes 0
            mask = 1. - make_pad_mask(enc_hs_len).float()
            att_prev = mask / mask.new(enc_hs_len).unsqueeze(-1)
            att_prev = att_prev.to(self.enc_h)
            self.c = torch.sum(self.enc_h * att_prev.view(batch, self.h_length, 1), dim=1)

        return self.c, att_prev


class AttDot(torch.nn.Module):
    """Dot product attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    """

    def __init__(self, eprojs, dunits, att_dim):
        super(AttDot, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        """AttDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weight (B x T_max)
        :rtype: torch.Tensor
        """

        batch = enc_hs_pad.size(0)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = torch.tanh(self.mlp_enc(self.enc_h))

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        e = torch.sum(self.pre_compute_enc_h * torch.tanh(self.mlp_dec(dec_z)).view(batch, 1, self.att_dim),
                      dim=2)  # utt x frame

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
        return c, w


class AttAdd(torch.nn.Module):
    """Additive attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    """

    def __init__(self, eprojs, dunits, att_dim):
        super(AttAdd, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        """AttLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights (B x T_max)
        :rtype: torch.Tensor
        """

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, w


class AttLoc(torch.nn.Module):
    """location-aware attention

    Reference: Attention-Based Models for Speech Recognition
        (https://arxiv.org/pdf/1506.07503.pdf)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    """

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        """AttLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attention weight (B x T_max)
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights (B x T_max)
        :rtype: torch.Tensor
        """
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            # if no bias, 0 0-pad goes 0
            att_prev = to_device(self, (1. - make_pad_mask(enc_hs_len).float()))
            att_prev = att_prev / att_prev.new(enc_hs_len).unsqueeze(-1)

        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, w


class AttCov(torch.nn.Module):
    """Coverage mechanism attention

    Reference: Get To The Point: Summarization with Pointer-Generator Network
       (https://arxiv.org/abs/1704.04368)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    """

    def __init__(self, eprojs, dunits, att_dim):
        super(AttCov, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.wvec = torch.nn.Linear(1, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_list, scaling=2.0):
        """AttCov forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param list att_prev_list: list of previous attention weight
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attention weights
        :rtype: list
        """

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev_list is None:
            # if no bias, 0 0-pad goes 0
            att_prev_list = to_device(self, (1. - make_pad_mask(enc_hs_len).float()))
            att_prev_list = [att_prev_list / att_prev_list.new(enc_hs_len).unsqueeze(-1)]

        # att_prev_list: L' * [B x T] => cov_vec B x T
        cov_vec = sum(att_prev_list)
        # cov_vec: B x T => B x T x 1 => B x T x att_dim
        cov_vec = self.wvec(cov_vec.unsqueeze(-1))

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(cov_vec + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)
        att_prev_list += [w]

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, att_prev_list


class AttLoc2D(torch.nn.Module):
    """2D location-aware attention

    This attention is an extended version of location aware attention.
    It take not only one frame before attention weights, but also earlier frames into account.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param int att_win: attention window size (default=5)
    """

    def __init__(self, eprojs, dunits, att_dim, att_win, aconv_chans, aconv_filts):
        super(AttLoc2D, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (att_win, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.aconv_chans = aconv_chans
        self.att_win = att_win
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        """AttLoc2D forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attention weight (B x att_win x T_max)
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights (B x att_win x T_max)
        :rtype: torch.Tensor
        """

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            # B * [Li x att_win]
            # if no bias, 0 0-pad goes 0
            att_prev = to_device(self, (1. - make_pad_mask(enc_hs_len).float()))
            att_prev = att_prev / att_prev.new(enc_hs_len).unsqueeze(-1)
            att_prev = att_prev.unsqueeze(1).expand(-1, self.att_win, -1)

        # att_prev: B x att_win x Tmax -> B x 1 x att_win x Tmax -> B x C x 1 x Tmax
        att_conv = self.loc_conv(att_prev.unsqueeze(1))
        # att_conv: B x C x 1 x Tmax -> B x Tmax x C
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        # update att_prev: B x att_win x Tmax -> B x att_win+1 x Tmax -> B x att_win x Tmax
        att_prev = torch.cat([att_prev, w.unsqueeze(1)], dim=1)
        att_prev = att_prev[:, 1:]

        return c, att_prev


class AttLocRec(torch.nn.Module):
    """location-aware recurrent attention

    This attention is an extended version of location aware attention.
    With the use of RNN, it take the effect of the history of attention weights into account.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    """

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttLocRec, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.att_lstm = torch.nn.LSTMCell(aconv_chans, att_dim, bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_states, scaling=2.0):
        """AttLocRec forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param tuple att_prev_states: previous attention weight and lstm states
                                      ((B, T_max), ((B, att_dim), (B, att_dim)))
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights and lstm states (w, (hx, cx))
                 ((B, T_max), ((B, att_dim), (B, att_dim)))
        :rtype: tuple
        """

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev_states is None:
            # initialize attention weight with uniform dist.
            # if no bias, 0 0-pad goes 0
            att_prev = to_device(self, (1. - make_pad_mask(enc_hs_len).float()))
            att_prev = att_prev / att_prev.new(enc_hs_len).unsqueeze(-1)

            # initialize lstm states
            att_h = enc_hs_pad.new_zeros(batch, self.att_dim)
            att_c = enc_hs_pad.new_zeros(batch, self.att_dim)
            att_states = (att_h, att_c)
        else:
            att_prev = att_prev_states[0]
            att_states = att_prev_states[1]

        # B x 1 x 1 x T -> B x C x 1 x T
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # apply non-linear
        att_conv = F.relu(att_conv)
        # B x C x 1 x T -> B x C x 1 x 1 -> B x C
        att_conv = F.max_pool2d(att_conv, (1, att_conv.size(3))).view(batch, -1)

        att_h, att_c = self.att_lstm(att_conv, att_states)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_h.unsqueeze(1) + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, (w, (att_h, att_c))


class AttCovLoc(torch.nn.Module):
    """Coverage mechanism location aware attention

    This attention is a combination of coverage and location-aware attentions.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    """

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttCovLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.aconv_chans = aconv_chans
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_list, scaling=2.0):
        """AttCovLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param list att_prev_list: list of previous attention weight
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attention weights
        :rtype: list
        """

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev_list is None:
            # if no bias, 0 0-pad goes 0
            mask = 1. - make_pad_mask(enc_hs_len).float()
            att_prev_list = [to_device(self, mask / mask.new(enc_hs_len).unsqueeze(-1))]

        # att_prev_list: L' * [B x T] => cov_vec B x T
        cov_vec = sum(att_prev_list)

        # cov_vec: B x T -> B x 1 x 1 x T -> B x C x 1 x T
        att_conv = self.loc_conv(cov_vec.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)
        att_prev_list += [w]

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, att_prev_list


class AttMultiHeadDot(torch.nn.Module):
    """Multi head dot product attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    """

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v):
        super(AttMultiHeadDot, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        for _ in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        """AttMultiHeadDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :return: attention weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attention weight (B x T_max) * aheads
        :rtype: list
        """

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                torch.tanh(self.mlp_k[h](self.enc_h)) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            e = torch.sum(self.pre_compute_k[h] * torch.tanh(self.mlp_q[h](dec_z)).view(
                batch, 1, self.att_dim_k), dim=2)  # utt x frame

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_device(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadAdd(torch.nn.Module):
    """Multi head additive attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using additive attention for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    """

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v):
        super(AttMultiHeadAdd, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        for _ in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        """AttMultiHeadAdd forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attention weight (B x T_max) * aheads
        :rtype: list
        """

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            e = self.gvec[h](torch.tanh(
                self.pre_compute_k[h] + self.mlp_q[h](dec_z).view(batch, 1, self.att_dim_k))).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_device(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadLoc(torch.nn.Module):
    """Multi head location based attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using location-aware attention for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    """

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v, aconv_chans, aconv_filts):
        super(AttMultiHeadLoc, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        self.loc_conv = torch.nn.ModuleList()
        self.mlp_att = torch.nn.ModuleList()
        for _ in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
            self.loc_conv += [torch.nn.Conv2d(
                1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)]
            self.mlp_att += [torch.nn.Linear(aconv_chans, att_dim_k, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        """AttMultiHeadLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: list of previous attention weight (B x T_max) * aheads
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attention weight (B x T_max) * aheads
        :rtype: list
        """

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            att_prev = []
            for _ in six.moves.range(self.aheads):
                # if no bias, 0 0-pad goes 0
                mask = 1. - make_pad_mask(enc_hs_len).float()
                att_prev += [to_device(self, mask / mask.new(enc_hs_len).unsqueeze(-1))]

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            att_conv = self.loc_conv[h](att_prev[h].view(batch, 1, 1, self.h_length))
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            att_conv = self.mlp_att[h](att_conv)

            e = self.gvec[h](torch.tanh(
                self.pre_compute_k[h] + att_conv + self.mlp_q[h](dec_z).view(
                    batch, 1, self.att_dim_k))).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_device(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMoChAMultiHeadLoc(torch.nn.Module):
    """Multi head location based attention support online decoding

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using location-aware attention for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    """

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v, aconv_chans, aconv_filts):
        super(AttMultiHeadLoc, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        self.loc_conv = torch.nn.ModuleList()
        self.mlp_att = torch.nn.ModuleList()
        for _ in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
            self.loc_conv += [torch.nn.Conv2d(
                1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)]
            self.mlp_att += [torch.nn.Linear(aconv_chans, att_dim_k, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        """AttMultiHeadLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: list of previous attention weight (B x T_max) * aheads
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attention weight (B x T_max) * aheads
        :rtype: list
        """

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            att_prev = []
            for _ in six.moves.range(self.aheads):
                # if no bias, 0 0-pad goes 0
                mask = 1. - make_pad_mask(enc_hs_len).float()
                att_prev += [to_device(self, mask / mask.new(enc_hs_len).unsqueeze(-1))]

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            att_conv = self.loc_conv[h](att_prev[h].view(batch, 1, 1, self.h_length))
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            att_conv = self.mlp_att[h](att_conv)

            e = self.gvec[h](torch.tanh(
                self.pre_compute_k[h] + att_conv + self.mlp_q[h](dec_z).view(
                    batch, 1, self.att_dim_k))).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_device(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadMultiResLoc(torch.nn.Module):
    """Multi head multi resolution location based attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using location-aware attention for each head.
    Furthermore, it uses different filter size for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    :param int aconv_chans: maximum # channels of attention convolution
        each head use #ch = aconv_chans * (head + 1) / aheads
        e.g. aheads=4, aconv_chans=100 => filter size = 25, 50, 75, 100
    :param int aconv_filts: filter size of attention convolution
    """

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v, aconv_chans, aconv_filts):
        super(AttMultiHeadMultiResLoc, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        self.loc_conv = torch.nn.ModuleList()
        self.mlp_att = torch.nn.ModuleList()
        for h in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
            afilts = aconv_filts * (h + 1) // aheads
            self.loc_conv += [torch.nn.Conv2d(
                1, aconv_chans, (1, 2 * afilts + 1), padding=(0, afilts), bias=False)]
            self.mlp_att += [torch.nn.Linear(aconv_chans, att_dim_k, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        """AttMultiHeadMultiResLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: list of previous attention weight (B x T_max) * aheads
        :return: attention weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attention weight (B x T_max) * aheads
        :rtype: list
        """

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            att_prev = []
            for _ in six.moves.range(self.aheads):
                # if no bias, 0 0-pad goes 0
                mask = 1. - make_pad_mask(enc_hs_len).float()
                att_prev += [to_device(self, mask / mask.new(enc_hs_len).unsqueeze(-1))]

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            att_conv = self.loc_conv[h](att_prev[h].view(batch, 1, 1, self.h_length))
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            att_conv = self.mlp_att[h](att_conv)

            e = self.gvec[h](torch.tanh(
                self.pre_compute_k[h] + att_conv + self.mlp_q[h](dec_z).view(
                    batch, 1, self.att_dim_k))).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_device(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttForward(torch.nn.Module):
    """Forward attention

    Reference: Forward attention in sequence-to-sequence acoustic modeling for speech synthesis
        (https://arxiv.org/pdf/1807.06736.pdf)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    """

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttForward, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=1.0):
        """AttForward forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: attention weights of previous step
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights (B x T_max)
        :rtype: torch.Tensor
        """
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            # initial attention will be [1, 0, 0, ...]
            att_prev = enc_hs_pad.new_zeros(*enc_hs_pad.size()[:2])
            att_prev[:, 0] = 1.0

        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).unsqueeze(1)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled + att_conv)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # forward attention
        att_prev_shift = F.pad(att_prev, (1, 0))[:, :-1]
        w = (att_prev + att_prev_shift) * w
        # NOTE: clamp is needed to avoid nan gradient
        w = F.normalize(torch.clamp(w, 1e-6), p=1, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.unsqueeze(-1), dim=1)

        return c, w


class AttForwardTA(torch.nn.Module):
    """Forward attention with transition agent

    Reference: Forward attention in sequence-to-sequence acoustic modeling for speech synthesis
        (https://arxiv.org/pdf/1807.06736.pdf)

    :param int eunits: # units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param int odim: output dimension
    """

    def __init__(self, eunits, dunits, att_dim, aconv_chans, aconv_filts, odim):
        super(AttForwardTA, self).__init__()
        self.mlp_enc = torch.nn.Linear(eunits, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_ta = torch.nn.Linear(eunits + dunits + odim, 1)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eunits = eunits
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.trans_agent_prob = 0.5

    def reset(self):
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.trans_agent_prob = 0.5

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, out_prev, scaling=1.0):
        """AttForwardTA forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B, Tmax, eunits)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B, dunits)
        :param torch.Tensor att_prev: attention weights of previous step
        :param torch.Tensor out_prev: decoder outputs of previous step (B, odim)
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, dunits)
        :rtype: torch.Tensor
        :return: previous attention weights (B, Tmax)
        :rtype: torch.Tensor
        """
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            # initial attention will be [1, 0, 0, ...]
            att_prev = enc_hs_pad.new_zeros(*enc_hs_pad.size()[:2])
            att_prev[:, 0] = 1.0

        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # forward attention
        att_prev_shift = F.pad(att_prev, (1, 0))[:, :-1]
        w = (self.trans_agent_prob * att_prev + (1 - self.trans_agent_prob) * att_prev_shift) * w
        # NOTE: clamp is needed to avoid nan gradient
        w = F.normalize(torch.clamp(w, 1e-6), p=1, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        # update transition agent prob
        self.trans_agent_prob = torch.sigmoid(
            self.mlp_ta(torch.cat([c, out_prev, dec_z], dim=1)))

        return c, w


def att_for(args, num_att=1):
    """Instantiates an attention module given the program arguments

    :param Namespace args: The arguments
    :param int num_att: number of attention modules (in multi-speaker case, it can be 2 or more)
    :rtype torch.nn.Module
    :return: The attention module
    """
    if args.lm_fusion:
        logging.info('## double dunits for att ##')
        dunits = args.dunits * 2
    else:
        dunits = args.dunits
    att_list = torch.nn.ModuleList()
    for i in range(num_att):
        att = None
        if args.atype == 'noatt':
            att = NoAtt()
        elif args.atype == 'dot':
            att = AttDot(args.eprojs, dunits, args.adim)
        elif args.atype == 'add':
            att = AttAdd(args.eprojs, dunits, args.adim)
        elif args.atype == 'location':
            att = AttLoc(args.eprojs, dunits,
                         args.adim, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'location2d':
            att = AttLoc2D(args.eprojs, dunits,
                           args.adim, args.awin, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'location_recurrent':
            att = AttLocRec(args.eprojs, dunits,
                            args.adim, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'coverage':
            att = AttCov(args.eprojs, dunits, args.adim)
        elif args.atype == 'coverage_location':
            att = AttCovLoc(args.eprojs, dunits, args.adim,
                            args.aconv_chans, args.aconv_filts)
        elif args.atype == 'multi_head_dot':
            att = AttMultiHeadDot(args.eprojs, dunits,
                                  args.aheads, args.adim, args.adim)
        elif args.atype == 'multi_head_add':
            att = AttMultiHeadAdd(args.eprojs, dunits,
                                  args.aheads, args.adim, args.adim)
        elif args.atype == 'multi_head_loc':
            att = AttMultiHeadLoc(args.eprojs, dunits,
                                  args.aheads, args.adim, args.adim,
                                  args.aconv_chans, args.aconv_filts)
        elif args.atype == 'multi_head_multi_res_loc':
            att = AttMultiHeadMultiResLoc(args.eprojs, dunits,
                                          args.aheads, args.adim, args.adim,
                                          args.aconv_chans, args.aconv_filts)
        elif args.atype == 'MoChA':
            att = AttMoChA(args.eprojs, dunits, args.adim, chunk_size=10, simple=args.att_mocha_simple)
        elif args.atype == 'batch_multi_MoChA':
            att = AttBatchMultiHeadMoChA(args.eprojs, dunits, args.adim, args.att_chunk_size, args.aheads)
        elif args.atype == 'concat_multi_MoChA':
            att = AttConcatMultiHeadMoChA(args.eprojs, dunits, args.adim, args.att_chunk_size, args.aheads)
        elif args.atype == 'multi_MoChA':
            att = AttMultiHeadMoChA(args.eprojs, dunits, args.adim, args.att_chunk_size, args.aheads)
        elif args.atype == 'ProjMoChA':
            att = AttProjMoChA(args.eprojs, dunits, args.adim, args.att_chunk_size)
        elif args.atype == 'MoChAstable':
            att = AttMoChAstable(args.eprojs, dunits, args.adim, args.att_chunk_size)
        att_list.append(att)
    return att_list


def att_to_numpy(att_ws, att):
    """Converts attention weights to a numpy array given the attention

    :param list att_ws: The attention weights
    :param torch.nn.Module att: The attention
    :rtype: np.ndarray
    :return: The numpy array of the attention weights
    """
    # convert to numpy array with the shape (B, Lmax, Tmax)
    if isinstance(att, AttLoc2D):
        # att_ws => list of previous concate attentions
        att_ws = torch.stack([aw[:, -1] for aw in att_ws], dim=1).cpu().numpy()
    elif isinstance(att, (AttCov, AttCovLoc)):
        # att_ws => list of list of previous attentions
        att_ws = torch.stack([aw[-1] for aw in att_ws], dim=1).cpu().numpy()
    elif isinstance(att, AttLocRec):
        # att_ws => list of tuple of attention and hidden states
        att_ws = torch.stack([aw[0] for aw in att_ws], dim=1).cpu().numpy()
    elif isinstance(att, (AttMultiHeadDot, AttMultiHeadAdd, AttMultiHeadLoc, AttMultiHeadMultiResLoc, AttMultiHeadMoChA, AttBatchMultiHeadMoChA, AttConcatMultiHeadMoChA)):
        # att_ws => list of list of each head attention
        n_heads = len(att_ws[0])
        att_ws_sorted_by_head = []
        for h in six.moves.range(n_heads):
            att_ws_head = torch.stack([aw[h] for aw in att_ws], dim=1)
            att_ws_sorted_by_head += [att_ws_head]
        att_ws = torch.stack(att_ws_sorted_by_head, dim=1).cpu().numpy()
    else:
        # att_ws => list of attentions
        att_ws = torch.stack(att_ws, dim=1).cpu().numpy()
    return att_ws
