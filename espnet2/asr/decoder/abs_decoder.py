from abc import ABC, abstractmethod
from typing import Tuple, List, Any

import torch

from espnet.nets.scorer_interface import ScorerInterface


class AbsDecoder(torch.nn.Module, ScorerInterface, ABC):
    @abstractmethod
    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

first_pass_params = dict(
    beams=10,
    token_list=None,
    model_path=None,
    alpha=0,
    beta=0,
    cutoff_top_n=16,
    cutoff_prob=1.0,
    num_processes=32,
    apply=False,
)

second_pass_encoder_params = dict(
    type="blstm",
    hidden_size=300,
    output_size=300,
    num_blocks=1,
)

class AbsFirstPassDecoder(torch.nn.Module, ABC):
    def beam_search(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        **kwargs,
    ) -> List[Any]:
        raise NotImplementedError

    def greedy_search(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
    ) -> List[Any]:
        raise NotImplementedError
