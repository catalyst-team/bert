from typing import Dict
from collections import OrderedDict

import pandas as pd
from catalyst.dl import ConfigExperiment

from bert_ner.catalyst_ext import StateKeys
from bert_ner.dataset import KeyphrasesDataset


class Experiment(ConfigExperiment):
    def __init__(self, config: Dict, keys: StateKeys = StateKeys.default()):
        super().__init__(config)
        self.keys = keys

    def get_transforms(self, stage: str = None, mode: str = None):
        return []

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        train = pd.read_json(
            './input/train.jsonl',
            lines=True,
            orient='records',
        )
        val = pd.read_json(
            './input/test.jsonl',
            lines=True,
            orient='records',
        )

        trainset = KeyphrasesDataset(
            train['content'],
            train['tagged_attributes'],
            self.keys,
        )
        testset = KeyphrasesDataset(
            val['content'],
            val['tagged_attributes'],
            self.keys,
        )

        datasets['train'] = trainset
        datasets['valid'] = testset

        return datasets
