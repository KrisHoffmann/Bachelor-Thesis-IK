"""Count y=1 comments in each pipeline stage output."""
import json
from pathlib import Path

BASE = Path("/home/krishoffmann/projects/thesis_repository/final_dataset_creation/dataset_construction/outputs")

files = [
    ("comments_filtered.jsonl",       BASE / "comments_filtered.jsonl",       "comment"),
    ("comments_segmented.jsonl",       BASE / "comments_segmented.jsonl",       "comment"),
    ("sentences_for_classifier.jsonl", BASE / "sentences_for_classifier.jsonl", "sentence"),
]

for label, path, mode in files:
    if not path.exists():
        print(f"{label}: NOT FOUND")
        continue
    y1_cids = set()
    y0 = y1 = 0
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("y") == 1:
                y1 += 1
                y1_cids.add(obj["comment_id"])
            else:
                y0 += 1
    if mode == "sentence":
        print(f"{label}: total_rows={y0+y1:,}  y=1_rows={y1:,}  unique_comment_ids_with_y1={len(y1_cids):,}")
    else:
        print(f"{label}: total_rows={y0+y1:,}  y=0={y0:,}  y=1={y1:,}")
