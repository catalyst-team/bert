from collections import OrderedDict

import pandas as pd
from catalyst.dl import ConfigExperiment

from bert_ner.dataset import KeyphrasesDataset


class Experiment(ConfigExperiment):
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

        trainset = KeyphrasesDataset(train['content'], train['tagged_attributes'])
        testset = KeyphrasesDataset(val['content'], val['tagged_attributes'])

        datasets['train'] = trainset
        datasets['valid'] = testset

        return datasets
