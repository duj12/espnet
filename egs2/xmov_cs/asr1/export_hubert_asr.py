#!/usr/bin/env python3
import argparse
import sys
import torch
from espnet2.tasks.asr import ASRTask
from espnet2.utils import config_argparse
from espnet.utils.cli_utils import get_commandline_args

def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASR Hubert Encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.

    parser.add_argument("--output_path", type=str, required=True)
    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--asr_train_config",
        type=str,
        help="ASR training configuration",
    )
    group.add_argument(
        "--asr_model_file",
        type=str,
        help="ASR model parameter file",
    )
    group.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="model device",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)

    asr_model, asr_train_args = ASRTask.build_model_from_file(
            args.asr_train_config, args.asr_model_file, args.device
    )
    hubert_encoder = asr_model.encoder.encoders
    torch.save(hubert_encoder, args.output_path)

if __name__ == "__main__":
    main()
