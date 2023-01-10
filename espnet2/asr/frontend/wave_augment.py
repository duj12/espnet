#!/usr/bin/env python3
#  2022, Jean Du, xmov.ai
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Augmentation on raw wave, including:
        add_noise, add_reverb, time_stretch,
        pitch_shift, volume_perturb, speed_perturb."""

from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
import random
import librosa

class WaveAugment(AbsFrontend):
    """
    perform augmentation on wave samples' level.
    """

    def __init__(
        self,
        noise_list: str,
        reverb_list: str,
        noise_add_prob: float = 0.8,
        min_snr: int = 5,
        max_snr: int = 15,
        reverb_add_prob: float = 0.2,
        speed_perturb_prob: float = 0.5,
        min_speed_ratio: float = 0.9,
        max_speed_ratio: float = 1.1,
        volume_perturb_prob: float = 0.5,
        min_volume_ratio: float = 0.5,
        max_volume_ratio: float = 2.0,
        pitch_shift_prob: float = 0.5,
        low_pitch_shift: int = -5,
        high_pitch_shift: int = 5,
        random_seed: int = 0,

    ):
        """Initialize.
        Args:
        """
        assert check_argument_types()
        super().__init__()
        self.noise_list = []
        with open(noise_list, 'r') as f_noise:
            for line in f_noise:
                path = line.strip()
                self.noise_list.append(path)
        self.reverb_list = []
        with open(reverb_list, 'r') as f_reverb:
            for line in f_reverb:
                path = line.strip()
                self.reverb_list.append(path)
        self.noise_add_prob = noise_add_prob
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.reverb_add_prob = reverb_add_prob
        self.speed_perturb_prob = speed_perturb_prob
        self.min_speed_ratio = min_speed_ratio
        self.max_speed_ratio = max_speed_ratio
        self.volume_perturb_prob = volume_perturb_prob
        self.min_volume_ratio = min_volume_ratio
        self.max_volume_ratio = max_volume_ratio
        self.pitch_shift_prob = pitch_shift_prob
        self.low_pitch_shift = low_pitch_shift
        self.high_pitch_shift = high_pitch_shift
        self.rand_generator = random.Random(random_seed)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation on wave.

        Args:
            input: Input (B, T).
            input_lengths: Input lengths within batch(sample counts).

        Returns:
            Tensor: Output with dimensions (B, T).
            Tensor: Output lengths within batch.
        """
        input_size = input.size()
        B = input_size[0]
        T = input_size[1]
        output = input.clone()
        output_lengths = input_lengths.clone()

        #1.time_stretch with sox_tfm.tempo
        for i, wave in enumerate(input.cpu().numpy()):
            pass

        # add_noise


        # add_reverb

        return output, output_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D."""
        return 1
