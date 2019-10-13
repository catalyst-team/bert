class StateKeys:
    def __init__(self, input_ids, attention_mask, targets, model_output):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets
        self.model_output = model_output

    @classmethod
    def default(cls):
        return StateKeys('input_ids', 'attention_mask', 'targets', 'logits')
