import torch.nn as nn

from .model import DistilBertForTokenClassification

MODEL_MAP = {
    'distilbert/token_class':
    (DistilBertForTokenClassification, 'distilbert-base-uncased'),
}


class BertModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        model_type = kwargs.pop('model_type', None)
        model_cls, model_base = MODEL_MAP[model_type]
        self.model = model_cls.from_pretrained(model_base, **kwargs)

    # pylint: disable=arguments-differ
    def forward(self, input_ids, attention_mask=None, head_mask=None):
        return self.model(input_ids, attention_mask=attention_mask, head_mask=head_mask)
