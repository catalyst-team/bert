# flake8: noqa
# pylint: disable=unused-import
from catalyst.dl import registry
from transformers import AdamW, WarmupLinearSchedule

from .experiment import Experiment
from .catalyst_ext.runner import BertSupervisedRunner as Runner
from .model_wrapper import BertModel

from .catalyst_ext.bert_criterion import BertCrossEntropyLoss, BertCriterionCallback

registry.Model(BertModel)
registry.Criterion(BertCrossEntropyLoss)
registry.Callback(BertCriterionCallback)
registry.Optimizer(AdamW, name='TransformersAdamW')
registry.Scheduler(WarmupLinearSchedule)
