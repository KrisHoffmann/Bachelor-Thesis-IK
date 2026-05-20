"""Phase 1 investigation for Stage 8 merge."""
import polars as pl
from pathlib import Path

BASE = Path("/home/krishoffmann/projects/thesis_repository/final_dataset_creation/dataset_construction")

def report_parquet(label, path):
    p = Path(path)
    if not p.exists():
        print(f"\n[{label}] NOT FOUND: {path}")
        return None
    size_mb = p.stat().st_size / 1_048_576
    df = pl.read_parquet(path)
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"  Path:  {path}")
    print(f"  Size:  {size_mb:.1f} MB")
    print(f"  Rows:  {len(df):,}")
    print(f"  Cols:  {len(df.columns)}")
    print(f"\n  Schema (name | dtype | null_count):")
    for col in df.columns:
        nc = df[col].null_count()
        print(f"    {col:<35} {str(df[col].dtype):<20} nulls={nc:,}")
    return df

# ── 1. Load both parquets ─────────────────────────────────────────────────────
df_agg   = report_parquet("stage7_comment_abstraction", BASE / "outputs/stage7_comment_abstraction.parquet")
df_ctrl  = report_parquet("comments_controls",          BASE / "outputs/comments_controls.parquet")

# ── 2. Check if y is already in controls ─────────────────────────────────────
print(f"\n\n{'='*60}")
print("[y presence check]")
if "y" in df_ctrl.columns:
    print(f"  'y' IS in comments_controls.parquet")
    print(f"  y distribution: {df_ctrl['y'].value_counts().sort('y')}")
else:
    print(f"  'y' is NOT in comments_controls.parquet")

# ── 3. Check stage7 abstraction for y ────────────────────────────────────────
if "y" in df_agg.columns:
    print(f"  'y' IS in stage7_comment_abstraction.parquet")
else:
    print(f"  'y' is NOT in stage7_comment_abstraction.parquet")

# ── 4. Overlap / anti-join analysis ──────────────────────────────────────────
print(f"\n\n{'='*60}")
print("[Overlap analysis: stage7_abstraction vs comments_controls]")

cids_agg  = set(df_agg["comment_id"].to_list())
cids_ctrl = set(df_ctrl["comment_id"].to_list())

only_agg  = cids_agg  - cids_ctrl
only_ctrl = cids_ctrl - cids_agg
both      = cids_agg  & cids_ctrl

print(f"  stage7_abstraction unique comment_ids: {len(cids_agg):,}")
print(f"  comments_controls  unique comment_ids: {len(cids_ctrl):,}")
print(f"  In both:                               {len(both):,}")
print(f"  In stage7 only (not in controls):      {len(only_agg):,}")
print(f"  In controls only (not in stage7):      {len(only_ctrl):,}")

if only_agg:
    print(f"  Examples only in stage7: {list(only_agg)[:5]}")
if only_ctrl:
    print(f"  Examples only in controls: {list(only_ctrl)[:5]}")

# ── 5. Duplicate comment_id check ────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("[Duplicate comment_id check]")

for label, df in [("stage7_abstraction", df_agg), ("comments_controls", df_ctrl)]:
    n_rows  = len(df)
    n_uniq  = df["comment_id"].n_unique()
    n_dups  = n_rows - n_uniq
    print(f"  {label}: rows={n_rows:,}  unique_cids={n_uniq:,}  duplicates={n_dups:,}")

# ── 6. Author consistency check ───────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("[Author column check]")

agg_has_author  = "author"    in df_agg.columns
agg_has_authid  = "author_id" in df_agg.columns
ctrl_has_author = "author"    in df_ctrl.columns
ctrl_has_authid = "author_id" in df_ctrl.columns

print(f"  stage7_abstraction: has 'author'={agg_has_author}  has 'author_id'={agg_has_authid}")
print(f"  comments_controls:  has 'author'={ctrl_has_author}  has 'author_id'={ctrl_has_authid}")

if agg_has_author:
    samples = df_agg["author"].drop_nulls().head(5).to_list()
    print(f"  stage7 'author' sample values: {samples}")
if ctrl_has_author:
    samples = df_ctrl["author"].drop_nulls().head(5).to_list()
    print(f"  controls 'author' sample values: {samples}")
if ctrl_has_authid:
    samples = df_ctrl["author_id"].drop_nulls().head(5).to_list()
    print(f"  controls 'author_id' sample values: {samples}")

# ── 7. Probe other candidate files for y ─────────────────────────────────────
print(f"\n\n{'='*60}")
print("[Probing other candidate files for y]")

candidates = [
    BASE / "outputs/comments_filtered.jsonl",
    BASE / "outputs/comments_segmented.jsonl",
    BASE / "outputs/comments_raw.jsonl",
]
for cand in candidates:
    if not cand.exists():
        print(f"  {cand.name}: NOT FOUND")
        continue
    size_mb = cand.stat().st_size / 1_048_576
    # read just the first line to check keys
    with open(cand) as f:
        import json
        first = json.loads(f.readline())
    keys = list(first.keys())
    has_y = "y" in keys
    print(f"  {cand.name} ({size_mb:.0f} MB): has_y={has_y}  keys={keys[:12]}")
