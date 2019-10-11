#!/usr/bin/env python
from typing import Tuple
import logging

import torch.nn as nn
from catalyst.dl.callbacks import OptimizerCallback
from transformers import AdamW, WarmupLinearSchedule

from bert_ner.experiment import Experiment
from bert_ner.model import DistilBertForTokenClassification
from bert_ner.catalyst_ext import (BertSupervisedRunner, BertCrossEntropyLoss,
                                   BertCriterionCallback, StateKeys)

# experiment setup
CONFIG = dict(stages=dict(
    data_params=dict(num_workers=1, batch_size=8),
    stage_1=dict(),
))

NUM_EPOCHS = 100
LOGDIR = "./logs/"


def get_runner() -> Tuple[BertSupervisedRunner, StateKeys]:
    state_keys = StateKeys('input_ids', 'attention_mask', 'targets', 'logits')
    return BertSupervisedRunner(state_keys), state_keys


def get_model(num_classes=2) -> Tuple[nn.Module, nn.Module]:
    model = DistilBertForTokenClassification.from_pretrained(
        'distilbert-base-uncased',
        num_classes=num_classes,
    )
    criterion = BertCrossEntropyLoss(num_classes)
    return model, criterion


def main():
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.FATAL)

    # model, criterion
    model, criterion = get_model()
    # model runner
    runner, state_keys = get_runner()
    # data
    loaders = Experiment(CONFIG, state_keys).get_loaders('stage_1')

    # optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01,
        correct_bias=False,
    )
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=500,
        t_total=3000,
    )

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=LOGDIR,
        num_epochs=NUM_EPOCHS,
        callbacks=[
            BertCriterionCallback(state_keys),
            OptimizerCallback(accumulation_steps=4),
            # SchedulerCallback(reduce_metric='accuracy01'),
        ],
        verbose=True)


if __name__ == '__main__':
    main()
