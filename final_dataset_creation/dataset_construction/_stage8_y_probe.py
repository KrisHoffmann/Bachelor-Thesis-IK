"""Probe y-bearing JSONL files: row counts, y distribution, useful columns."""
import json
from pathlib import Path

BASE = Path("/home/krishoffmann/projects/thesis_repository/final_dataset_creation/dataset_construction/outputs")

files = [
    ("comments_filtered.jsonl",  BASE / "comments_filtered.jsonl"),
    ("comments_segmented.jsonl", BASE / "comments_segmented.jsonl"),
]

for label, path in files:
    print(f"\n{'='*60}")
    print(f"[{label}]")
    rows = 0
    y0 = y1 = 0
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            rows += 1
            if obj.get("y") == 1:
                y1 += 1
            else:
                y0 += 1
    print(f"  rows: {rows:,}  y=0: {y0:,}  y=1: {y1:,}")

# Also check comments_filtered for the exact columns we'd want to carry forward
print(f"\n{'='*60}")
print("[comments_filtered.jsonl — first row full contents (minus body)]")
with open(BASE / "comments_filtered.jsonl") as f:
    obj = json.loads(f.readline())
for k, v in obj.items():
    if k != "body":
        print(f"  {k}: {repr(v)}")
