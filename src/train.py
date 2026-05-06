"""
Stage 1 salience training script.

Usage:
    python src/train.py --config configs/bert_stage1.yaml

Device detection:
    GPU present  → fp16=True,  per_device batch = gpu_batch_size
    CPU only     → fp16=False, per_device batch = cpu_batch_size

HF cache:
    Respects HF_HOME env var if set; otherwise uses the HuggingFace default.
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# allow `python src/train.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_split, verify_splits
from src.evaluate import compute_metrics_dict
from src.model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = [
        "model_name", "lr", "num_epochs", "gpu_batch_size", "cpu_batch_size",
        "max_length", "seed", "output_dir", "run_name",
    ]
    for key in required:
        if key not in cfg:
            raise KeyError(f"Config missing required key: '{key}'")
    return cfg


class WeightedTrainer(Trainer):
    """Trainer subclass that applies class-weighted cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    use_gpu = torch.cuda.is_available()
    device_name = "cuda" if use_gpu else "cpu"
    batch_size = cfg["gpu_batch_size"] if use_gpu else cfg["cpu_batch_size"]
    print(f"Device: {device_name}  |  batch_size: {batch_size}  |  fp16: {use_gpu}")

    data_dir = Path("data/processed")
    verify_splits(data_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg["max_length"],
            padding=False,
        )

    train_ds = load_split(data_dir / "train.json").map(tokenize, batched=True)
    dev_ds = load_split(data_dir / "dev.json").map(tokenize, batched=True)
    train_ds = train_ds.remove_columns(["text"])
    dev_ds = dev_ds.remove_columns(["text"])
    train_ds.set_format("torch")
    dev_ds.set_format("torch")

    # class weights from train split only: w_c = n_total / (2 * n_c)
    labels = train_ds["label"]
    n_total = len(labels)
    n_pos = int(labels.sum().item())
    n_neg = n_total - n_pos
    w_neg = n_total / (2 * n_neg)
    w_pos = n_total / (2 * n_pos)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32)
    print(f"Class weights — 0 (not salient): {w_neg:.4f}, 1 (salient): {w_pos:.4f}")

    model = build_model(cfg["model_name"])

    run_dir = Path(cfg["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=cfg["lr"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=use_gpu,
        seed=cfg["seed"],
        report_to="none",
        run_name=cfg["run_name"],
    )

    def hf_compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics = compute_metrics_dict(preds.tolist(), labels.tolist())
        return {
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
        }

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_compute_metrics,
    )

    trainer.train()

    # save config copy, dev predictions, metrics
    shutil.copy(args.config, run_dir / "config.yaml")

    preds_output = trainer.predict(dev_ds)
    pred_labels = np.argmax(preds_output.predictions, axis=-1).tolist()
    gold_labels = dev_ds["label"]
    if hasattr(gold_labels, "tolist"):
        gold_labels = gold_labels.tolist()

    metrics = compute_metrics_dict(pred_labels, gold_labels)

    with open(run_dir / "dev_predictions.json", "w") as f:
        json.dump({"predictions": pred_labels, "gold": gold_labels}, f)

    metrics_serialisable = {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in metrics.items()
        if k != "confusion_matrix"
    }
    metrics_serialisable["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_serialisable, f, indent=2)

    # confusion matrix PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        disp = ConfusionMatrixDisplay(
            confusion_matrix=metrics["confusion_matrix"],
            display_labels=["not salient", "salient"],
        )
        disp.plot(colorbar=False)
        plt.title("Dev confusion matrix")
        plt.tight_layout()
        plt.savefig(run_dir / "confusion_matrix.png", dpi=150)
        plt.close()
    except ImportError:
        print("matplotlib not installed — skipping confusion matrix PNG")

    print(f"\nDone. Outputs saved to {run_dir}/")
    print(f"Dev macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
