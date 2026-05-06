"""
Data loading utilities for the CLT salience classifier (Stage 1).

JSON schema (each record):
    task_id        int   — Label Studio task ID
    post_id        str   — Reddit post ID
    sentence_idx   int   — Sentence position within the post
    sentence_text  str   — Raw sentence text  →  mapped to  'text'
    salient        bool  — Gold label (true=salient) →  mapped to  'label' (1=salient, 0=not)
    temporal/spatial/social/hypothetical  — CLT dimensions, Stage 2 only, ignored here

Field mapping:
    sentence_text  →  text   (str)
    salient        →  label  (int: 1 if True else 0)
"""

import json
from pathlib import Path
from datasets import Dataset


def load_split(path: str) -> Dataset:
    """Load a single JSON split file and return a HuggingFace Dataset.

    Columns returned: 'text' (str), 'label' (int).
    Crashes loudly if the file is missing or a required field is absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"Expected a non-empty JSON array in {path}")

    texts = []
    labels = []
    for i, rec in enumerate(records):
        if "sentence_text" not in rec:
            raise KeyError(f"Record {i} in {path} missing 'sentence_text'")
        if "salient" not in rec:
            raise KeyError(f"Record {i} in {path} missing 'salient'")
        texts.append(rec["sentence_text"])
        labels.append(1 if rec["salient"] else 0)

    return Dataset.from_dict({"text": texts, "label": labels})


def verify_splits(data_dir: str) -> None:
    """Assert split counts and absence of sentence overlap between splits.

    Expected total counts:  train=723, dev=176, test=155.
    Expected salient counts: train≈621, dev≈147, test≈127.
    Hard-fails on any mismatch.
    """
    data_dir = Path(data_dir)
    expected_totals = {"train": 723, "dev": 176, "test": 155}
    expected_salient = {"train": 621, "dev": 147, "test": 127}

    all_keys: dict[str, set] = {}

    for split, expected_n in expected_totals.items():
        path = data_dir / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")

        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        n = len(records)
        assert n == expected_n, (
            f"{split}: expected {expected_n} records, got {n}"
        )

        salient_n = sum(1 for r in records if r["salient"])
        assert salient_n == expected_salient[split], (
            f"{split}: expected {expected_salient[split]} salient records, got {salient_n}"
        )

        keys = set()
        for r in records:
            key = (r["task_id"], r["sentence_idx"])
            if key in keys:
                raise ValueError(
                    f"{split}: duplicate (task_id, sentence_idx) = {key}"
                )
            keys.add(key)
        all_keys[split] = keys

    # cross-split overlap check
    splits = list(all_keys.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = all_keys[splits[i]] & all_keys[splits[j]]
            assert len(overlap) == 0, (
                f"Overlap between {splits[i]} and {splits[j]}: {len(overlap)} shared sentences"
            )

    print(
        "verify_splits OK — train=723 (621 sal), dev=176 (147 sal), test=155 (127 sal), no overlap"
    )
