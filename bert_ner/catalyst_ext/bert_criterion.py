import torch.nn as nn

from catalyst.dl.callbacks import CriterionCallback as _CriterionCallback
from catalyst.dl.core import RunnerState


class BertCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    # pylint: disable=arguments-differ
    def forward(self, logits, attention_mask, target):
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_classes)[active_loss]
            active_labels = target.view(-1)[active_loss]
            loss = super().forward(active_logits, active_labels)
        else:
            loss = super().forward(logits.view(-1, self.num_classes), target.view(-1))

        return loss


class BertCriterionCallback(_CriterionCallback):
    def _compute_loss(self, state: RunnerState, criterion):
        logits = self._get(state.output, self.output_key)
        attention_mask = self._get(state.input, 'attention_mask')
        targets = self._get(state.input, self.input_key)

        loss = criterion(logits, attention_mask, targets)
        return loss
