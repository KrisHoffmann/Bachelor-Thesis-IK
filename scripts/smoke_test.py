"""
Smoke test — Stage 1 (salience) and Stage 2 (CLT dimensions).

Phase A: Loads 5 mixed sentences (salient + non-salient), runs untrained
         bert-base-uncased Stage 1 forward pass, prints shapes + predictions.

Phase B: Loads 5 salient sentences, runs untrained CLTStage2Model forward pass,
         prints encoder output shape, all four head logit shapes, and argmax
         predictions per dimension.

CPU only. Expected runtime: < 1 minute total.
Exit 0 on success.
"""

import json
import sys
from pathlib import Path

# run from repo root: python scripts/smoke_test.py
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model import CLTStage2Model

DATA_PATH = repo_root / "data" / "processed" / "train.json"
MODEL_NAME = "bert-base-uncased"


def pick_sentences(records: list, n: int = 5) -> list:
    """Return n records: first (n-2) salient + first 2 non-salient, deterministic."""
    salient = [r for r in records if r["salient"]]
    non_salient = [r for r in records if not r["salient"]]
    if len(non_salient) < 2:
        raise ValueError("Not enough non-salient sentences in train.json")
    selected = salient[: n - 2] + non_salient[:2]
    return selected[:n]


def pick_salient(records: list, n: int = 5) -> list:
    """Return first n salient records, deterministic."""
    salient = [r for r in records if r["salient"]]
    if len(salient) < n:
        raise ValueError(f"Not enough salient sentences: need {n}, got {len(salient)}")
    return salient[:n]


def smoke_stage1(records: list, tokenizer, model_name: str) -> None:
    print("=" * 60)
    print("Phase A — Stage 1 (salience) smoke test")
    print("=" * 60)

    sentences = pick_sentences(records)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.eval()

    texts = [s["sentence_text"] for s in sentences]
    gold = [int(s["salient"]) for s in sentences]

    encoding = tokenizer(
        texts, truncation=True, max_length=128, padding=True, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).tolist()

    print(f"input_ids shape:      {encoding['input_ids'].shape}")
    print(f"attention_mask shape: {encoding['attention_mask'].shape}")
    print(f"logits shape:         {logits.shape}\n")

    for i, (sent, gold_label, pred) in enumerate(zip(sentences, gold, preds)):
        tag = "salient" if gold_label == 1 else "not_salient"
        print(f"[{i}] gold={tag:<12} pred={pred}  | {sent['sentence_text'][:80]}")

    print("\nStage 1 smoke test PASSED.\n")


def smoke_stage2(records: list, tokenizer, model_name: str) -> None:
    print("=" * 60)
    print("Phase B — Stage 2 (CLT dimensions) smoke test")
    print("=" * 60)

    sentences = pick_salient(records, n=5)
    model = CLTStage2Model(model_name)
    model.eval()

    texts = [s["sentence_text"] for s in sentences]

    encoding = tokenizer(
        texts, truncation=True, max_length=128, padding=True, return_tensors="pt"
    )

    with torch.no_grad():
        logits_dict = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )

    # encoder hidden state shape via a direct encoder call
    with torch.no_grad():
        enc_out = model.encoder(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )
    cls_shape = enc_out.last_hidden_state[:, 0, :].shape
    print(f"encoder [CLS] shape:  {cls_shape}   (batch=5, hidden=768)")

    dims = ("temporal", "spatial", "social", "hypothetical")
    for dim in dims:
        shape = logits_dict[dim].shape
        assert shape == (5, 3), f"Expected (5,3) for {dim}, got {shape}"
        print(f"{dim:<14} logits shape: {shape}")

    print()
    # raw label map: index → decoded value
    inv = {0: -1, 1: 0, 2: 1}
    dim_names = ("temporal", "spatial", "social", "hypothetical")
    for i, sent in enumerate(sentences):
        preds_str = "  ".join(
            f"{d[:4]}={inv[torch.argmax(logits_dict[d][i]).item()]}"
            for d in dim_names
        )
        print(f"[{i}] {preds_str}  | {sent['sentence_text'][:70]}")

    print("\nStage 2 smoke test PASSED.\n")


def main():
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: data file not found: {DATA_PATH}")

    with open(DATA_PATH, encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loading tokenizer: {MODEL_NAME}\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    smoke_stage1(records, tokenizer, MODEL_NAME)
    smoke_stage2(records, tokenizer, MODEL_NAME)

    print("All smoke tests PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
