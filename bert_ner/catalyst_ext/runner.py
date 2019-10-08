from typing import Mapping, Any

from catalyst.dl import SupervisedRunner


class BertSupervisedRunner(SupervisedRunner):
    def _batch2device(self, batch: Mapping[str, Any], device):
        assert len(batch) == 3
        batch = {
            self.input_key: batch[0],
            self.target_key: batch[1],
            'attention_mask': batch[2],
        }
        batch = super()._batch2device(batch, device)
        return batch

    def forward(self, batch):
        """
        Should not be called directly outside of runner.
        If your model has specific interface, override this method to use it
        """
        output = self.model(batch[self.input_key], batch['attention_mask'])
        output = self._process_output(output)
        return output
