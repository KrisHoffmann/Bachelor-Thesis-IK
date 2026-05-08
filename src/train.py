"""
CLT classifier training script — Stage 1 (salience) and Stage 2 (CLT dimensions).

Usage:
    python src/train.py --config configs/bert_stage1.yaml            # Stage 1
    python src/train.py --config configs/bert_stage2.yaml --stage 2  # Stage 2

The --stage flag overrides the 'stage' key in the config file if both are present;
the config key is used as a default when --stage is not passed on the CLI.

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
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

# allow `python src/train.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DIMS, INV_LABEL_MAP, load_split, load_split_stage2, verify_splits
from src.evaluate import compute_metrics_dict, evaluate_stage2
from src.model import CLTStage2Model, build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str, stage: int) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = [
        "model_name", "lr", "num_epochs", "gpu_batch_size", "cpu_batch_size",
        "max_length", "seed", "output_dir", "run_name",
    ]
    if stage == 2:
        required.append("warmup_ratio")
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


def _stage2_class_weights(dataset, dim: str, device: torch.device) -> torch.Tensor:
    """Compute per-class weights for one dimension from a Stage 2 dataset.

    Formula: weight_c = n_total / (3 * n_class_c).
    Classes 0/1/2 correspond to N/A, Near, Far.
    If a class is absent, its weight is set to 0.0 (zero gradient contribution).
    """
    labels = dataset[dim]
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    n_total = len(labels)
    weights = []
    for c in range(3):
        n_c = labels.count(c)
        weights.append(n_total / (3 * n_c) if n_c > 0 else 0.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_stage2(cfg: dict, use_gpu: bool, config_path: str) -> None:
    """Custom training loop for Stage 2 (CLT dimension classifier).

    Uses torch AdamW + linear warmup scheduler. HF Trainer is NOT used here
    because Trainer does not natively support multi-output losses with
    per-output class weights.
    """
    device = torch.device("cuda" if use_gpu else "cpu")
    batch_size = cfg["gpu_batch_size"] if use_gpu else cfg["cpu_batch_size"]
    run_dir = Path(cfg["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data/processed")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    def tokenize_s2(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg["max_length"],
            padding=False,
        )

    train_ds = load_split_stage2(data_dir / "train.json").map(tokenize_s2, batched=True)
    dev_ds = load_split_stage2(data_dir / "dev.json").map(tokenize_s2, batched=True)

    print(f"Stage 2 data — train: {len(train_ds)} salient, dev: {len(dev_ds)} salient")

    # compute class weights from train split only
    train_ds_torch = train_ds.remove_columns(["text"])
    dev_ds_torch = dev_ds.remove_columns(["text"])
    train_ds_torch.set_format("torch")
    dev_ds_torch.set_format("torch")

    class_weights = {}
    for dim in DIMS:
        w = _stage2_class_weights(train_ds_torch, dim, device)
        class_weights[dim] = w
        print(f"  {dim:<14} class weights — N/A={w[0]:.3f}, Near={w[1]:.3f}, Far={w[2]:.3f}")

    loss_fns = {
        dim: torch.nn.CrossEntropyLoss(weight=class_weights[dim])
        for dim in DIMS
    }

    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    train_loader = DataLoader(
        train_ds_torch, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    dev_loader = DataLoader(
        dev_ds_torch, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    model = CLTStage2Model(cfg["model_name"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_mean_f1 = -1.0
    best_epoch = -1
    all_epoch_metrics: list[dict] = []

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        epoch_losses = []
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits_dict = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = sum(
                loss_fns[dim](logits_dict[dim], batch[dim].to(device))
                for dim in DIMS
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            print(f"  epoch {epoch} step {step:4d}  loss={loss.item():.4f}")

        # evaluation
        model.eval()
        all_preds: dict[str, list[int]] = {d: [] for d in DIMS}
        all_gold: dict[str, list[int]] = {d: [] for d in DIMS}
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits_dict = model(input_ids=input_ids, attention_mask=attention_mask)
                for dim in DIMS:
                    pred_idx = torch.argmax(logits_dict[dim], dim=-1).cpu().tolist()
                    gold_idx = batch[dim].tolist()
                    # decode class indices back to {-1, 0, 1}
                    all_preds[dim].extend(INV_LABEL_MAP[p] for p in pred_idx)
                    all_gold[dim].extend(INV_LABEL_MAP[g] for g in gold_idx)

        metrics = evaluate_stage2(all_preds, all_gold)
        mean_f1 = metrics["mean_macro_f1"]

        print(f"\nEpoch {epoch} — mean macro-F1: {mean_f1:.4f}")
        for dim in DIMS:
            dm = metrics[dim]
            collapse = " [COLLAPSE]" if dm.get("collapsed") else ""
            print(f"  {dim:<14} macro-F1={dm['macro_f1']:.4f}{collapse}")
        print()

        epoch_record = {
            "epoch": epoch,
            "mean_train_loss": float(np.mean(epoch_losses)),
            "mean_macro_f1": mean_f1,
            **{dim: {k: (v.tolist() if hasattr(v, "tolist") else v)
                     for k, v in metrics[dim].items()
                     if k != "confusion_matrix"}
               for dim in DIMS},
        }
        all_epoch_metrics.append(epoch_record)

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            # save best-epoch dev predictions (decoded)
            with open(run_dir / "dev_predictions.json", "w") as f:
                json.dump({"predictions": all_preds, "gold": all_gold}, f, indent=2)

    print(f"Best epoch: {best_epoch}  mean macro-F1: {best_mean_f1:.4f}")

    # save metrics log and config copy
    with open(run_dir / "epoch_metrics.json", "w") as f:
        json.dump(all_epoch_metrics, f, indent=2)

    # re-load best model for final confusion matrix PNGs
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()
    final_preds: dict[str, list[int]] = {d: [] for d in DIMS}
    final_gold: dict[str, list[int]] = {d: [] for d in DIMS}
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits_dict = model(input_ids=input_ids, attention_mask=attention_mask)
            for dim in DIMS:
                pred_idx = torch.argmax(logits_dict[dim], dim=-1).cpu().tolist()
                gold_idx = batch[dim].tolist()
                final_preds[dim].extend(INV_LABEL_MAP[p] for p in pred_idx)
                final_gold[dim].extend(INV_LABEL_MAP[g] for g in gold_idx)

    final_metrics = evaluate_stage2(final_preds, final_gold)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        for dim in DIMS:
            cm = final_metrics[dim]["confusion_matrix"]
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=["N/A", "Near", "Far"]
            )
            disp.plot(colorbar=False)
            plt.title(f"Dev confusion matrix — {dim}")
            plt.tight_layout()
            plt.savefig(run_dir / f"confusion_matrix_{dim}.png", dpi=150)
            plt.close()
    except ImportError:
        print("matplotlib not installed — skipping confusion matrix PNGs")

    shutil.copy(config_path, run_dir / "config.yaml")
    print(f"\nDone. Outputs saved to {run_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        help="Training stage (1 or 2). Overrides 'stage' key in config if set.",
    )
    args = parser.parse_args()

    # resolve stage: CLI flag takes precedence over config key
    stage_from_config: int | None = None
    with open(args.config, encoding="utf-8") as _f:
        _raw = yaml.safe_load(_f)
        if "stage" in _raw:
            stage_from_config = int(_raw["stage"])

    stage = args.stage if args.stage is not None else (stage_from_config or 1)
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")

    cfg = load_config(args.config, stage)
    seed_everything(cfg["seed"])

    use_gpu = torch.cuda.is_available()
    device_name = "cuda" if use_gpu else "cpu"
    batch_size = cfg["gpu_batch_size"] if use_gpu else cfg["cpu_batch_size"]
    print(f"Stage: {stage}  |  Device: {device_name}  |  batch_size: {batch_size}  |  fp16: {use_gpu}")

    if stage == 2:
        train_stage2(cfg, use_gpu, args.config)
        return

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
