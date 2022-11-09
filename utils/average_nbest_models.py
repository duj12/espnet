#!/usr/bin/env python

from espnet2.train.reporter import Reporter
from espnet2.main_funcs.average_nbest_models import average_nbest_models
from espnet2.utils.types import str2triple_str

from pathlib import Path
import argparse

import torch


def main():
    parser = argparse.ArgumentParser(
        description='Average nbest models from exp dirctory ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nbest-models', type=int, default=10,
                        help='number of models to average')
    parser.add_argument('--criterion', type=str2triple_str,
                        default=[
                            ("valid", "acc", "max"),
                        ],
                        help='critrion to define "best"')
    parser.add_argument('exp_dir', type=str,
                        help='espnet2 asr train exp dir.  e.g. exp/asr_train_transformer/')
    args = parser.parse_args()

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"

    exp_dir = Path(args.exp_dir)
    state = torch.load(exp_dir / 'checkpoint.pth', map_location='cpu')

    reporter = Reporter()
    reporter.load_state_dict(state['reporter'])

    average_nbest_models(reporter=reporter, output_dir=exp_dir,
                         best_model_criterion=args.criterion, nbest=args.nbest_models)


if __name__ == "__main__":
    main()