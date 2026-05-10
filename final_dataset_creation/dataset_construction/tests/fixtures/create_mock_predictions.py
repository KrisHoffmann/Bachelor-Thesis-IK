"""
Generate mock_sentence_predictions.parquet for post-classifier smoke tests.

Covers comments that survive Stage 1+2 filtering from mock_pairs.jsonl.bz2.
Includes edge cases:
  - all-NaN comment (must be dropped by Stage 7)
  - single-dimension salient
  - all-dimensions salient
  - mixed labels
"""

import sys
from pathlib import Path

import pandas as pd

# Comments that survive filtering (not deleted/bot/short/quote-only):
# c001, c002, c003, c004, c005 (econ01)
# c006, c006b, c012 (econ02 — c007=[deleted], c008=AutoMod, c009=FooBot, c010=short, c011=quote)
# c013, c014 (econ03)
# c015, c016, c017 (econ04)
SURVIVING_COMMENTS = [
    "c001", "c002", "c003", "c004", "c005",
    "c006", "c006b", "c012",
    "c013", "c014",
    "c015", "c016", "c017",
]

# Sentence counts per comment (mock — 2-4 sentences each)
COMMENT_SENTENCES = {cid: (2 + i % 3) for i, cid in enumerate(SURVIVING_COMMENTS)}

rows = []

# Patterns to exercise different Stage 7 paths:
label_patterns = [
    # (label_T, label_S, label_Soc, label_H)
    (1, 0, -1, -1),    # T-salient, S-salient only
    (-1, -1, -1, -1),  # all -1 (zero-salience for this sentence, but others may save comment)
    (1, 1, 1, 1),      # all salient
    (0, 1, -1, 0),     # mixed
    (-1, 0, 1, -1),    # S+Soc salient
    (1, -1, 0, 1),     # T+Soc+H salient
]

all_nan_comment = "c003"  # This comment gets ALL -1 → must be dropped by Stage 7

for i, cid in enumerate(SURVIVING_COMMENTS):
    n_sents = COMMENT_SENTENCES[cid]
    for sent_idx in range(n_sents):
        if cid == all_nan_comment:
            lt, ls, lsoc, lh = -1, -1, -1, -1
        else:
            lt, ls, lsoc, lh = label_patterns[(i + sent_idx) % len(label_patterns)]
        rows.append({
            "comment_id": cid,
            "sentence_index": sent_idx,
            "label_T": lt,
            "label_S": ls,
            "label_Soc": lsoc,
            "label_H": lh,
        })

df = pd.DataFrame(rows)
out_path = Path(__file__).parent / "mock_sentence_predictions.parquet"
df.to_parquet(out_path, index=False)

n_all_nan = df.groupby("comment_id").apply(
    lambda g: (g[["label_T", "label_S", "label_Soc", "label_H"]] == -1).all().all()
).sum()
print(f"Written {len(df)} sentence rows across {df['comment_id'].nunique()} comments to {out_path}")
print(f"Comments with all-NaN labels (expect 1): {n_all_nan}")
print(f"Comments that should survive Stage 7: {df['comment_id'].nunique() - n_all_nan}")
