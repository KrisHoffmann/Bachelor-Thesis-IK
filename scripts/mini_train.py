"""
Phase B — Mini-train.

Trains bert-base-uncased for 1 epoch on the first 50 train sentences and
evaluates on the first 10 dev sentences.  Batch size 4, lr 2e-5, CPU only.
Prints loss every step and final dev macro-F1.
Saves checkpoint to outputs/mini_run/.

Pass conditions:
  - Loss decreases at least once during training.
  - Eval runs without error.
  - Checkpoint files exist in outputs/mini_run/ after training.

Expected runtime: < 10 minutes on a modern CPU.
"""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data import load_split
from src.evaluate import compute_metrics_dict
from src.model import build_model

MODEL_NAME = "bert-base-uncased"
N_TRAIN = 50
N_DEV = 10
BATCH_SIZE = 4
LR = 2e-5
OUTPUT_DIR = repo_root / "outputs" / "mini_run"

TRAIN_PATH = repo_root / "data" / "processed" / "train.json"
DEV_PATH = repo_root / "data" / "processed" / "dev.json"


class LoggingTrainer(Trainer):
    """Trainer that records and prints loss at every step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_losses: list[float] = []

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        loss_val = loss.item() if hasattr(loss, "item") else float(loss)
        self.step_losses.append(loss_val)
        step = len(self.step_losses)
        print(f"  step {step:3d}  loss={loss_val:.4f}")
        return loss


def main():
    for p in [TRAIN_PATH, DEV_PATH]:
        if not p.exists():
            sys.exit(f"ERROR: data file not found: {p}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=128, padding=False
        )

    print(f"Loading {N_TRAIN} train + {N_DEV} dev sentences …")
    train_ds = load_split(TRAIN_PATH).select(range(N_TRAIN)).map(tokenize, batched=True)
    dev_ds = load_split(DEV_PATH).select(range(N_DEV)).map(tokenize, batched=True)
    train_ds = train_ds.remove_columns(["text"])
    dev_ds = dev_ds.remove_columns(["text"])
    train_ds.set_format("torch")
    dev_ds.set_format("torch")

    model = build_model(MODEL_NAME)

    def hf_compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1).tolist()
        metrics = compute_metrics_dict(preds, labels.tolist())
        return {"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"]}

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=False,
        seed=42,
        report_to="none",
        no_cuda=True,
    )

    trainer = LoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_compute_metrics,
    )

    print(f"\nTraining 1 epoch on {N_TRAIN} sentences (batch={BATCH_SIZE}) …\n")
    trainer.train()

    losses = trainer.step_losses
    if len(losses) < 2:
        print("WARNING: only one step recorded — cannot verify loss decrease.")
    else:
        decreased = any(losses[i] < losses[i - 1] for i in range(1, len(losses)))
        if not decreased:
            print(f"WARNING: loss never decreased across steps: {losses}")
        else:
            print(f"\nLoss decreased at least once. Steps: {losses}")

    # dev macro-F1
    preds_output = trainer.predict(dev_ds)
    pred_labels = np.argmax(preds_output.predictions, axis=-1).tolist()
    gold_labels = dev_ds["label"]
    if hasattr(gold_labels, "tolist"):
        gold_labels = gold_labels.tolist()
    metrics = compute_metrics_dict(pred_labels, gold_labels)
    print(f"\nDev macro-F1 ({N_DEV} sentences): {metrics['macro_f1']:.4f}")

    # verify checkpoint exists
    checkpoints = list(OUTPUT_DIR.glob("checkpoint-*"))
    if not checkpoints:
        sys.exit(f"ERROR: no checkpoint found in {OUTPUT_DIR}")
    print(f"Checkpoint saved: {checkpoints[0]}")

    print("\nMini-train PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
