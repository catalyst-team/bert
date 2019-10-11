import argparse
import logging

import pandas as pd
from torch.utils.data import DataLoader

from bert_ner.train import get_model, get_runner
from bert_ner.dataset import KeyphrasesDataset


def build_args():
    parser = argparse.ArgumentParser(description="Run inference using trained model")
    parser.add_argument(
        'checkpoint_path',
        type=str,
        help="Path to the checkpoint of the model",
    )
    parser.add_argument(
        'data_path',
        type=str,
        help="Path to the JSONL file",
    )
    return parser


def main():
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.FATAL)

    parser = build_args()
    args = parser.parse_args()

    runner, state_keys = get_runner()
    model, _ = get_model()

    df = pd.read_json(args.data_path, lines=True)
    predict_dataset = KeyphrasesDataset(
        df['content'],
        df['tagged_attributes'],
        state_keys,
    )
    loader = DataLoader(predict_dataset, batch_size=8)

    output = runner.predict_loader(
        model,
        resume=args.checkpoint_path,
        loader=loader,
        verbose=True,
    )
    print(output)


if __name__ == '__main__':
    main()
