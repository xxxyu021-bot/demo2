import torch
import torch.nn as nn
#方便兼容其他模型
from transformers import AutoModel

class MineModel(nn.Module):

    def __init__(
        self,
        model_name,
        dropout_rate,
        num_labels
    ):

        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name
        )
        self.dropout = nn.Dropout(
            dropout_rate
        )
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_labels
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = (
            outputs.last_hidden_state
        )

        pooled_output = self.dropout(
            sequence_output
        )
        logits = self.classifier(
            pooled_output
        )

        return logits
