#!/usr/bin/env python
import logging

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

NUM_CLASSES = 2
NUM_EPOCHS = 100
LOGDIR = "./logs/"

STATE_KEYS = StateKeys('input_ids', 'attention_mask', 'targets', 'logits')


def main():
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.FATAL)

    # data
    loaders = Experiment(CONFIG, STATE_KEYS).get_loaders('stage_1')

    # model, criterion, optimizer
    model = DistilBertForTokenClassification.from_pretrained(
        'distilbert-base-uncased',
        num_classes=NUM_CLASSES,
    )
    criterion = BertCrossEntropyLoss(NUM_CLASSES)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, correct_bias=False)
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=500,
        t_total=3000,
    )

    # model runner
    runner = BertSupervisedRunner(STATE_KEYS)

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
            BertCriterionCallback(STATE_KEYS),
            OptimizerCallback(accumulation_steps=4),
            # SchedulerCallback(reduce_metric='accuracy01'),
        ],
        verbose=True)


if __name__ == '__main__':
    main()
