"""Recheck primary sample exclusion counts carefully."""
import math
import polars as pl
from pathlib import Path

BASE = Path("/home/krishoffmann/projects/thesis_repository/final_dataset_creation/dataset_construction")
df = pl.read_parquet(BASE / "outputs/stage8_comment_full.parquet")

def is_nan_or_null(col):
    s = df[col]
    return s.is_null() | s.is_nan()

soc_missing = is_nan_or_null("A_social")
hyp_missing = is_nan_or_null("A_hypothetical")

# The two ways to define "excluded from primary sample"
# A: either A_social OR A_hypothetical missing  (union — excluded if ANY missing)
# B: BOTH A_social AND A_hypothetical missing   (intersection — excluded only if BOTH missing)
excluded_either = soc_missing | hyp_missing
excluded_both   = soc_missing & hyp_missing

n_primary_either = (~excluded_either).sum()   # 543,740 per earlier check
n_primary_both   = (~excluded_both).sum()

print(f"Total comments: {len(df):,}")
print()
print(f"Excluded if EITHER A_soc or A_hyp missing:  {excluded_either.sum():,}")
print(f"  -> primary sample (both defined):          {n_primary_either:,}  [this is the 543,740 figure]")
print()
print(f"Excluded only if BOTH A_soc AND A_hyp missing: {excluded_both.sum():,}")
print(f"  -> primary sample (at least one defined):    {n_primary_both:,}")
print()

# delta breakdowns
y1 = df["y"] == 1
print(f"y=1 total: {y1.sum():,}")
print()
print(f"y=1 AND A_soc missing AND A_hyp missing (BOTH null): {(y1 & excluded_both).sum():,}")
print(f"y=1 AND (A_soc missing OR A_hyp missing) (EITHER):   {(y1 & excluded_either).sum():,}")
print(f"y=1 AND both defined (in primary sample):             {(y1 & ~excluded_either).sum():,}")
