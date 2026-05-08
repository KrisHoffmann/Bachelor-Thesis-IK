"""
Phase B2 — Stage 2 mini-train.

Trains CLTStage2Model (bert-base-uncased) for 1 epoch on the first 50 salient
train sentences and evaluates on the first 15 salient dev sentences.
Batch size 4, lr 2e-5, CPU only.

Prints total (summed 4-dim) loss at every step and per-dimension macro-F1
after the single epoch.

Pass conditions:
  - Total loss decreases at least once across steps.
  - All four per-dimension macro-F1 values compute without error (poor values
    for Temporal/Spatial are expected and fine — collapse to N/A is normal).
  - Checkpoint file exists in outputs/mini_run_stage2/ after training.

Expected runtime: < 15 minutes on a modern CPU.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from src.data import DIMS, INV_LABEL_MAP, load_split_stage2
from src.evaluate import evaluate_stage2
from src.model import CLTStage2Model

MODEL_NAME = "bert-base-uncased"
N_TRAIN = 50
N_DEV = 15
BATCH_SIZE = 4
LR = 2e-5
WARMUP_RATIO = 0.1
OUTPUT_DIR = repo_root / "outputs" / "mini_run_stage2"

TRAIN_PATH = repo_root / "data" / "processed" / "train.json"
DEV_PATH = repo_root / "data" / "processed" / "dev.json"


def class_weights_for_dim(dataset, dim: str) -> torch.Tensor:
    """Compute per-class weights for one dimension.

    Formula: w_c = n_total / (3 * n_c). Zero weight for absent classes.
    """
    labels = dataset[dim]
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    n_total = len(labels)
    weights = []
    for c in range(3):
        n_c = labels.count(c)
        weights.append(n_total / (3 * n_c) if n_c > 0 else 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def main():
    for p in [TRAIN_PATH, DEV_PATH]:
        if not p.exists():
            sys.exit(f"ERROR: data file not found: {p}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128, padding=False)

    print(f"Loading first {N_TRAIN} salient train + first {N_DEV} salient dev sentences …")
    train_ds = load_split_stage2(TRAIN_PATH).select(range(N_TRAIN)).map(tokenize, batched=True)
    dev_ds = load_split_stage2(DEV_PATH).select(range(N_DEV)).map(tokenize, batched=True)

    train_ds_torch = train_ds.remove_columns(["text"])
    dev_ds_torch = dev_ds.remove_columns(["text"])
    train_ds_torch.set_format("torch")
    dev_ds_torch.set_format("torch")

    # class weights per dimension from train split
    print("\nClass weights:")
    loss_fns = {}
    for dim in DIMS:
        w = class_weights_for_dim(train_ds_torch, dim)
        loss_fns[dim] = torch.nn.CrossEntropyLoss(weight=w)
        print(f"  {dim:<14}: N/A={w[0]:.3f}, Near={w[1]:.3f}, Far={w[2]:.3f}")

    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    train_loader = DataLoader(train_ds_torch, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
    dev_loader = DataLoader(dev_ds_torch, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    model = CLTStage2Model(MODEL_NAME)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    total_steps = len(train_loader)
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"\nTraining 1 epoch on {N_TRAIN} salient sentences (batch={BATCH_SIZE}) …\n")
    model.train()
    step_losses: list[float] = []

    for step, batch in enumerate(train_loader, 1):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        logits_dict = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = sum(
            loss_fns[dim](logits_dict[dim], batch[dim])
            for dim in DIMS
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        step_losses.append(loss_val)
        print(f"  step {step:3d}  total_loss={loss_val:.4f}")

    # check pass condition: loss decreases at least once
    if len(step_losses) < 2:
        print("WARNING: only one step — cannot verify loss decrease.")
    else:
        decreased = any(step_losses[i] < step_losses[i - 1] for i in range(1, len(step_losses)))
        if not decreased:
            print(f"WARNING: loss never decreased. Steps: {step_losses}")
        else:
            print(f"\nLoss decreased at least once.")

    # evaluation
    model.eval()
    all_preds: dict[str, list[int]] = {d: [] for d in DIMS}
    all_gold: dict[str, list[int]] = {d: [] for d in DIMS}

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            logits_dict = model(input_ids=input_ids, attention_mask=attention_mask)
            for dim in DIMS:
                pred_idx = torch.argmax(logits_dict[dim], dim=-1).tolist()
                gold_idx = batch[dim].tolist()
                all_preds[dim].extend(INV_LABEL_MAP[p] for p in pred_idx)
                all_gold[dim].extend(INV_LABEL_MAP[g] for g in gold_idx)

    metrics = evaluate_stage2(all_preds, all_gold)
    print(f"\nDev results ({N_DEV} salient sentences):")
    print(f"  mean macro-F1: {metrics['mean_macro_f1']:.4f}")
    for dim in DIMS:
        dm = metrics[dim]
        collapse_note = " [COLLAPSE — predicts only one class, expected for imbalanced dims]" if dm["collapsed"] else ""
        print(f"  {dim:<14} macro-F1={dm['macro_f1']:.4f}{collapse_note}")

    # save checkpoint
    ckpt_path = OUTPUT_DIR / "mini_stage2_model.pt"
    torch.save(model.state_dict(), ckpt_path)

    if not ckpt_path.exists():
        sys.exit(f"ERROR: checkpoint not saved to {ckpt_path}")

    print(f"\nCheckpoint saved: {ckpt_path}")
    print("\nMini-train Stage 2 PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
