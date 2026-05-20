"""
Find the 132 delta-awarded comments excluded from the primary sample
(y=1, A_social=NaN, A_hypothetical=NaN), then fetch their bodies from
comments_filtered.jsonl and print 5 examples in full.
"""
import json, math, random
from pathlib import Path
import polars as pl

BASE = Path("/home/krishoffmann/projects/thesis_repository/final_dataset_creation/dataset_construction")

# ── 1. Find the excluded comment_ids from the merged parquet ──────────────────
df = pl.read_parquet(BASE / "outputs/stage8_comment_full.parquet")

excluded = df.filter(
    (pl.col("y") == 1)
    & pl.col("A_social").is_null()
    & pl.col("A_hypothetical").is_null()
)

print(f"Total excluded delta comments (y=1, A_soc=NaN, A_hyp=NaN): {len(excluded):,}\n")

# Keep a few columns for display alongside the body
meta_cols = ["comment_id", "n_sentences_total", "n_salient_total",
             "n_salient_on_social", "n_salient_on_hypothetical",
             "A_temporal", "A_spatial"]
meta = {row["comment_id"]: row for row in excluded.select(meta_cols).to_dicts()}

# Sample 5 deterministically
random.seed(42)
sample_cids = set(random.sample(list(meta.keys()), 5))

# ── 2. Fetch bodies from comments_filtered.jsonl ──────────────────────────────
bodies = {}
with open(BASE / "outputs/comments_filtered.jsonl") as fh:
    for line in fh:
        obj = json.loads(line)
        if obj["comment_id"] in sample_cids:
            bodies[obj["comment_id"]] = obj["body"]
            if len(bodies) == len(sample_cids):
                break  # early exit once all found

# ── 3. Print ──────────────────────────────────────────────────────────────────
for i, cid in enumerate(sorted(sample_cids), 1):
    m = meta[cid]
    body = bodies.get(cid, "[body not found]")
    print(f"{'='*70}")
    print(f"Example {i}: comment_id={cid}")
    print(f"  n_sentences_total : {m['n_sentences_total']}")
    print(f"  n_salient_total   : {m['n_salient_total']}")
    print(f"  n_salient_on_soc  : {m['n_salient_on_social']}")
    print(f"  n_salient_on_hyp  : {m['n_salient_on_hypothetical']}")
    a_t = m["A_temporal"]
    a_s = m["A_spatial"]
    print(f"  A_temporal        : {a_t}")
    print(f"  A_spatial         : {a_s}")
    print(f"  A_social          : NaN (excluded)")
    print(f"  A_hypothetical    : NaN (excluded)")
    print(f"\n  Body:\n{body}\n")
