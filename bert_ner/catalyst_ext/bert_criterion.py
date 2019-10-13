import torch.nn as nn

from catalyst.dl.callbacks import CriterionCallback as _CriterionCallback
from catalyst.dl.core import RunnerState

from .runner import StateKeys


class BertCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        self.num_classes = kwargs.pop('num_classes')
        super().__init__(**kwargs)

    # pylint: disable=arguments-differ
    def forward(self, logits, attention_mask, target):
        if attention_mask is None:
            loss = super().forward(logits.view(-1, self.num_classes), target.view(-1))
            return loss

        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_classes)[active_loss]
        active_labels = target.view(-1)[active_loss]
        return super().forward(active_logits, active_labels)


class BertCriterionCallback(_CriterionCallback):
    def __init__(self, keys: StateKeys = StateKeys.default(), **kwargs):
        super().__init__(**kwargs)
        self.keys = keys

    def _compute_loss(self, state: RunnerState, criterion: BertCrossEntropyLoss):
        logits = self._get(state.output, self.keys.model_output)
        attention_mask = self._get(state.input, self.keys.attention_mask)
        targets = self._get(state.input, self.keys.targets)

        loss = criterion(logits, attention_mask, targets)
        return loss
