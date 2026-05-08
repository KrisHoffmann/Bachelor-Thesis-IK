import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification


def build_model(model_name: str):
    """Return an untrained AutoModelForSequenceClassification with 2 output labels."""
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


class CLTStage2Model(nn.Module):
    """Four-head CLT dimension classifier sharing one BERT encoder.

    Design choice: four parallel 3-class linear heads (one per CLT dimension)
    rather than a single 12-logit head.

    Rationale:
    - Per-dimension loss weighting: Temporal and Spatial are severely
      imbalanced (~90% N/A), so they need class weights computed and applied
      independently. A shared 12-logit head would conflate these gradients.
    - Per-dimension metrics: macro-F1 and Krippendorff α are computed per
      dimension; separate heads make the correspondence explicit.
    - No additional parameters vs. a 12-logit head (4 × Linear(H,3) ==
      Linear(H,12) in total capacity), so there is no complexity cost.

    Args:
        model_name: HuggingFace model identifier (e.g. 'bert-base-uncased').
        num_dims:   Number of CLT dimensions (default 4, always 4 for Stage 2).
    """

    DIMS = ("temporal", "spatial", "social", "hypothetical")

    def __init__(self, model_name: str, num_dims: int = 4):
        super().__init__()
        if num_dims != 4:
            raise ValueError("CLTStage2Model always uses num_dims=4")
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.head_temporal = nn.Linear(hidden_size, 3)
        self.head_spatial = nn.Linear(hidden_size, 3)
        self.head_social = nn.Linear(hidden_size, 3)
        self.head_hypothetical = nn.Linear(hidden_size, 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run encoder + four classification heads.

        Returns:
            dict with keys 'temporal', 'spatial', 'social', 'hypothetical',
            each a tensor of shape (batch_size, 3).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (first token of last hidden state)
        cls = outputs.last_hidden_state[:, 0, :]
        return {
            "temporal": self.head_temporal(cls),
            "spatial": self.head_spatial(cls),
            "social": self.head_social(cls),
            "hypothetical": self.head_hypothetical(cls),
        }
