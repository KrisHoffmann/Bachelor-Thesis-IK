"""
Phase A — Smoke test.

Loads 5 sentences from data/processed/train.json (deterministic by index,
drawn to include both salient and non-salient examples), runs a single
untrained forward pass of bert-base-uncased, and prints tensor shapes +
per-sentence argmax predictions.

CPU only. Expected runtime: < 1 minute.
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

DATA_PATH = repo_root / "data" / "processed" / "train.json"
MODEL_NAME = "bert-base-uncased"

# Indices chosen to include both salient (majority) and non-salient records.
# train.json records 0-1 are salient; scan for two non-salient ones to mix in.
SENTENCE_INDICES = [0, 1, 2, 3, 4]  # will be replaced by mixed selection below


def pick_sentences(records: list, n: int = 5) -> list:
    """Return n records: first (n-2) salient + first 2 non-salient, deterministic."""
    salient = [r for r in records if r["salient"]]
    non_salient = [r for r in records if not r["salient"]]
    if len(non_salient) < 2:
        raise ValueError("Not enough non-salient sentences in train.json")
    selected = salient[: n - 2] + non_salient[:2]
    return selected[:n]


def main():
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: data file not found: {DATA_PATH}")

    with open(DATA_PATH, encoding="utf-8") as f:
        records = json.load(f)

    sentences = pick_sentences(records)

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.eval()

    texts = [s["sentence_text"] for s in sentences]
    gold = [int(s["salient"]) for s in sentences]

    encoding = tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).tolist()

    print(f"\ninput_ids shape:      {encoding['input_ids'].shape}")
    print(f"attention_mask shape: {encoding['attention_mask'].shape}")
    print(f"logits shape:         {logits.shape}\n")

    for i, (sent, gold_label, pred) in enumerate(zip(sentences, gold, preds)):
        tag = "salient" if gold_label == 1 else "not_salient"
        print(f"[{i}] gold={tag:<12} pred={pred}  | {sent['sentence_text'][:80]}")

    print("\nSmoke test PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
