from typing import Mapping, Any

from catalyst.dl import SupervisedRunner

from bert_ner.catalyst_ext import StateKeys


class BertSupervisedRunner(SupervisedRunner):
    def __init__(self, keys: StateKeys = StateKeys.default(), **kwargs):
        kwargs.update(
            dict(
                input_key=keys.input_ids,
                input_target_key=keys.targets,
                output_key=keys.model_output,
            ))
        super().__init__(**kwargs)

        self.keys = keys

    def forward(self, batch: Mapping[str, Any]):
        output = self.model(
            batch[self.keys.input_ids],
            batch[self.keys.attention_mask],
        )
        output = self._process_output(output)
        return output
