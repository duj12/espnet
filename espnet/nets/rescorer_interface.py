"""Scorer interface module."""

import torch

from typing import Any
from typing import Tuple
from typing import List


class RescorerInterface:
    """Rescorer interface for beam search.

    The rescorer performs rescoring of the all hypothesis.

    Examples:
        * Transformer-xl LM rescore
            * :class:`espnet.nets.rescorers.transformer_xl`
    """

    def init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        return None

    def rescore(self, y: torch.Tensor, state: Any) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                scores for next token that has a shape of `(n_vocab)`
                and next state for ys

        """
        raise NotImplementedError


class BatchRescorerInterface(RescorerInterface):
    """Batch scorer interface."""

    def batch_init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        return self.init_state(x)

    def batch_rescore(
        self, ys: torch.Tensor, states: List[Any]
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # warnings.warn(
        #     "{} batch score is implemented through for loop not parallelized".format(
        #         self.__class__.__name__
        #     )
        # )
        scores = list()
        outstates = list()
        for i, (y, state) in enumerate(zip(ys, states)):
            score, outstate = self.rescore(y, state)
            outstates.append(outstate)
            scores.append(score)
        #scores = torch.cat(scores, 0).view(ys.shape[0], -1)
        return scores, outstates
