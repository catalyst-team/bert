import torch.nn as nn

from transformers.modeling_distilbert import DistilBertPreTrainedModel, DistilBertModel

from catalyst.contrib import registry


@registry.Model
class DistilBertForTokenClassification(DistilBertPreTrainedModel):
    def __init__(self, config, num_classes=None):
        super().__init__(config)

        self.bert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

        self.init_weights()

    # pylint: disable=arguments-differ
    def forward(self, input_ids, attention_mask=None, head_mask=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits
