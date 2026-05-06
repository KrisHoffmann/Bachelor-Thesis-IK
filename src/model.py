from transformers import AutoModelForSequenceClassification


def build_model(model_name: str):
    """Return an untrained AutoModelForSequenceClassification with 2 output labels."""
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
