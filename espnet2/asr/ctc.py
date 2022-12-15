import logging

import torch
import torch.nn.functional as F
from typeguard import check_argument_types


class CTC(torch.nn.Module):
    """CTC module.

    Args:
        odim: dimension of outputs
        encoder_output_size: number of encoder projection units
        dropout_rate: dropout rate (0.0 ~ 1.0)
        ctc_type: builtin or warpctc
        reduce: reduce the CTC loss into a scalar
        ignore_nan_grad: Same as zero_infinity (keeping for backward compatiblity)
        zero_infinity:  Whether to zero infinite losses and the associated gradients.
    """

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: bool = None,
        zero_infinity: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.ctc_type = ctc_type
        if ignore_nan_grad is not None:
            zero_infinity = ignore_nan_grad

        if self.ctc_type == "builtin":
            self.ctc_loss = torch.nn.CTCLoss(
                reduction="none", zero_infinity=zero_infinity
            )
        elif self.ctc_type == "warpctc":
            import warpctc_pytorch as warp_ctc

            if zero_infinity:
                logging.warning("zero_infinity option is not supported for warp_ctc")
            self.ctc_loss = warp_ctc.CTCLoss(size_average=True, reduce=reduce)

        elif self.ctc_type == "gtnctc":
            from espnet.nets.pytorch_backend.gtn_ctc import GTNCTCLossFunction

            self.ctc_loss = GTNCTCLossFunction.apply
        else:
            raise ValueError(
                f'ctc_type must be "builtin" or "warpctc": {self.ctc_type}'
            )

        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        if self.ctc_type == "builtin":
            th_pred = th_pred.log_softmax(2)
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            size = th_pred.size(1)

            if self.reduce:
                # Batch-size average
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        elif self.ctc_type == "warpctc":
            # warpctc only supports float32
            th_pred = th_pred.to(dtype=torch.float32)

            th_target = th_target.cpu().int()
            th_ilen = th_ilen.cpu().int()
            th_olen = th_olen.cpu().int()
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            if self.reduce:
                # NOTE: sum() is needed to keep consistency since warpctc
                # return as tensor w/ shape (1,)
                # but builtin return as tensor w/o shape (scalar).
                loss = loss.sum()
            return loss

        elif self.ctc_type == "gtnctc":
            log_probs = torch.nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, th_target, th_ilen, 0, "none")

        else:
            raise NotImplementedError

    def forward(self, hs_pad, hlens, ys_pad, ys_lens):
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))

        if self.ctc_type == "gtnctc":
            # gtn expects list form for ys
            ys_true = [y[y != -1] for y in ys_pad]  # parse padded ys
        else:
            # ys_hat: (B, L, D) -> (L, B, D)
            ys_hat = ys_hat.transpose(0, 1)
            # (B, L) -> (BxL,)
            ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])

        loss = self.loss_fn(ys_hat, ys_true, hlens, ys_lens).to(
            device=hs_pad.device, dtype=hs_pad.dtype
        )

        return loss

    def softmax(self, hs_pad):
        """softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.softmax(self.ctc_lo(hs_pad), dim=2)

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)

    def forced_align(self, h, y, blank_id=0, ignore_id=-1):
        """forced alignment.

        :param torch.Tensor h: hidden state sequence, 2d tensor (T, D)
        :param torch.Tensor y: id sequence tensor 1d tensor (L)
        :param int y: blank symbol index
        :return: best alignment results
        :rtype: list
        """

        def interpolate_blank(label, blank_id=0):
            """Insert blank token between every two label token."""
            label = np.expand_dims(label, 1)
            blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
            label = np.concatenate([blanks, label], axis=1)
            label = label.reshape(-1)
            label = np.append(label, label[0])
            return label

        import numpy as np

        # logits is the probo f output of each phoneme at time t (L, T) labels x time
        # sequence is the index of the phoneme expected in the output sequence.

        # http://deeplearning.cs.cmu.edu/slides.spring19/lec14.CTC.pdf
        # slide 158
        def viterbi_align(logits, sequence):
            blank_id = 0
            sequence = list(sequence)
            for i in range(len(sequence)):
                sequence.insert(2 * i, blank_id)
            sequence.append(blank_id)

            T = logits.shape[1]
            N = len(sequence)
            # print(N, T)
            if (N > T):
                raise Exception("Number of expected symbols more than the time stamps")

            s = np.zeros((T, N))
            bp = np.zeros((T, N), dtype=np.int)
            bp.fill(-1)
            bscr = np.zeros((T, N))
            aligned_seq1 = np.zeros((T), dtype=np.int)
            aligned_seq2 = np.zeros((T), dtype=np.int)

            # filling S
            # print(sequence)
            for i in range(N):
                s[:, i] = logits[sequence[i]]

            # s = np.log(np.array([[0.1, 0.5, 0.4, 0.1, 0.2], [0.2, 0.2, 0.2, 0.3, 0.7], [0.4, 0.1, 0.1, 0.2, 0.6],\
            # 	 [0.2, 0.3, 0.3, 0.1, 0.6], [0.3, 0.1, 0.4, 0.4, 0.7]])).T

            # base case
            bp[0, 0] = 0  # made this 0 instead of -1.
            bp[0, 1] = 1
            bscr[0, 0] = s[0, 0]
            bscr[0, 1] = s[0, 1]
            bscr[0, 2:] = np.NINF

            # filling over the rest time stamps
            for t in range(1, T):
                bp[t, 0] = bp[t - 1, 0]
                bscr[t, 0] = bscr[t - 1, 0] + s[t, 0]
                bp[t, 1] = 1 if bscr[t - 1, 1] > bscr[t - 1, 0] else 0
                bscr[t, 1] = bscr[t - 1, bp[t, 1]] + s[t, 1]

                for i in range(2, N):
                    # print("going in")
                    if (i % 2 == 0):  # blank
                        bp[t, i] = i if bscr[t - 1, i] > bscr[t - 1, i - 1] else i - 1
                    else:
                        if (sequence[i] == sequence[i - 2]):
                            bp[t, i] = i if bscr[t - 1, i] > bscr[t - 1, i - 1] else i - 1
                        else:
                            bp[t, i] = i if (bscr[t - 1, i] > bscr[t - 1, i - 1] and bscr[t - 1, i] > bscr[
                                t - 1, i - 2]) else \
                                (i - 1 if (bscr[t - 1, i - 1] > bscr[t - 1, i] and bscr[t - 1, i - 1] > bscr[
                                    t - 1, i - 2]) else \
                                     i - 2)
                    bscr[t, i] = bscr[t - 1, bp[t, i]] + s[t, i]

            # print(bp.T)
            # print(np.exp(bscr).T)

            aligned_seq1[T - 1], path_score_1 = N - 1, 0
            for t in range(T - 1, 0, -1):
                aligned_seq1[t - 1] = bp[t, aligned_seq1[t]]
                path_score_1 += bscr[t, aligned_seq1[t]]

            aligned_seq2[T - 1], path_score_2 = N - 2, 0
            for t in range(T - 1, 0, -1):
                aligned_seq2[t - 1] = bp[t, aligned_seq2[t]]
                path_score_2 += bscr[t, aligned_seq2[t]]

            aligned_seq = aligned_seq1 if (path_score_1 > path_score_2) else aligned_seq2

            aligned_symbols_idx = []
            for i in range(len(aligned_seq)):
                if i > 0 and aligned_seq[i] == aligned_seq[i - 1]:
                    aligned_symbols_idx.append(0)
                else:
                    aligned_symbols_idx.append(sequence[aligned_seq[i]])
            # aligned_idx = np.where(np.array(aligned_symbols_idx) != 0)

            return aligned_symbols_idx

        lpz = self.log_softmax(h)
        lpz = lpz.squeeze(0)
        res = viterbi_align(lpz.cpu().numpy().T, [x for x in y.tolist() if x != -1])

        return res