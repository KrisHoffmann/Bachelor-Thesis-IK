"""
Data loading utilities for the CLT classifier — Stage 1 (salience) and Stage 2 (CLT dims).

JSON schema (each record):
    task_id        int   — Label Studio task ID
    post_id        str   — Reddit post ID
    sentence_idx   int   — Sentence position within the post
    sentence_text  str   — Raw sentence text  →  mapped to  'text'
    salient        bool  — Gold label (true=salient) →  mapped to  'label' (1=salient, 0=not)
    temporal       int|None  — {-1=N/A, 0=Near, 1=Far}; None treated as -1 (N/A)
    spatial        int|None  — {-1=N/A, 0=Near, 1=Far}; None treated as -1 (N/A)
    social         int|None  — {-1=N/A, 0=Near, 1=Far}; None treated as -1 (N/A)
    hypothetical   int|None  — {-1=N/A, 0=Near, 1=Far}; None treated as -1 (N/A)

Stage 1 field mapping:
    sentence_text  →  text   (str)
    salient        →  label  (int: 1 if True else 0)

Stage 2 field mapping (salient sentences only):
    sentence_text  →  text         (str)
    temporal       →  temporal     (int class index via LABEL_MAP)
    spatial        →  spatial      (int class index via LABEL_MAP)
    social         →  social       (int class index via LABEL_MAP)
    hypothetical   →  hypothetical (int class index via LABEL_MAP)

None dimension values (4 records in train, annotation gaps):
    Treated as -1 (N/A). Documented here so the choice is explicit and auditable.

Stage 2 label encoding — raw {-1, 0, 1} → class indices {0, 1, 2}:
    LABEL_MAP    = {-1: 0, 0: 1, 1: 2}
    INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}
"""

import json
from collections import Counter
from pathlib import Path

from datasets import Dataset

# Stage 2 label encoding: raw annotation value → PyTorch class index
LABEL_MAP: dict[int, int] = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP: dict[int, int] = {0: -1, 1: 0, 2: 1}

DIMS = ("temporal", "spatial", "social", "hypothetical")


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


# ---------------------------------------------------------------------------
# Stage 2 — CLT dimension loader
# ---------------------------------------------------------------------------

def load_split_stage2(path: str) -> Dataset:
    """Load a split file and return only salient sentences for Stage 2 training.

    Filters to records where salient=True.
    Maps raw dimension values {-1, 0, 1} to class indices {0, 1, 2} via LABEL_MAP.
    None dimension values (annotation gaps) are treated as -1 (N/A → class index 0).

    Columns returned:
        text         str
        temporal     int  (class index 0/1/2)
        spatial      int  (class index 0/1/2)
        social       int  (class index 0/1/2)
        hypothetical int  (class index 0/1/2)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"Expected a non-empty JSON array in {path}")

    texts: list[str] = []
    dim_data: dict[str, list[int]] = {d: [] for d in DIMS}

    for i, rec in enumerate(records):
        if not rec.get("salient", False):
            continue
        if "sentence_text" not in rec:
            raise KeyError(f"Record {i} in {path} missing 'sentence_text'")
        texts.append(rec["sentence_text"])
        for dim in DIMS:
            if dim not in rec:
                raise KeyError(f"Record {i} in {path} missing '{dim}'")
            raw = rec[dim]
            if raw is None:
                raw = -1  # treat annotation gap as N/A
            if raw not in LABEL_MAP:
                raise ValueError(
                    f"Record {i} in {path}: unexpected value {raw!r} for '{dim}'"
                )
            dim_data[dim].append(LABEL_MAP[raw])

    return Dataset.from_dict({"text": texts, **dim_data})


def verify_stage2_splits(data_dir: str) -> None:
    """Print per-dimension class counts for train and dev Stage 2 splits.

    Flags any dimension-class combination below 10% of observations with a
    WARNING — expected for Temporal Near/Far and Spatial Near/Far.
    Does NOT load the test split (never touch test during training/eval).
    """
    data_dir = Path(data_dir)
    class_names = {0: "N/A (-1)", 1: "Near (0)", 2: "Far (1)"}

    for split in ("train", "dev"):
        path = data_dir / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")

        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        salient = [r for r in records if r.get("salient", False)]
        n = len(salient)
        print(f"\nStage 2 split: {split}  ({n} salient sentences)")

        for dim in DIMS:
            raw_vals = [r[dim] if r[dim] is not None else -1 for r in salient]
            counts = Counter(raw_vals)
            parts = []
            for raw_val in (-1, 0, 1):
                c = counts.get(raw_val, 0)
                pct = 100 * c / n if n > 0 else 0.0
                flag = "  <-- WARNING: below 10%" if pct < 10.0 and c > 0 else (
                    "  <-- WARNING: class absent" if c == 0 else ""
                )
                parts.append(f"{class_names[LABEL_MAP[raw_val]]}={c} ({pct:.1f}%){flag}")
            print(f"  {dim:<14}: " + " | ".join(parts))
