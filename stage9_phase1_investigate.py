"""
Stage 9 Phase 1 — Diagnostic Investigation (READ-ONLY)
=======================================================
Investigates stage8_comment_full.parquet to report per-thread case/control
availability and projected fallback rates for k=3 and k=5 matching.

DOES NOT write or modify the parquet; the only output file is
outputs/stage9_phase1_diagnostics.csv (per-thread audit table).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
PARQUET_PATH = REPO_ROOT / "final_dataset_creation/dataset_construction/outputs/stage8_comment_full.parquet"
OUT_DIR = REPO_ROOT / "outputs"
DIAG_CSV = OUT_DIR / "stage9_phase1_diagnostics.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_ROWS = 598_272
EXPECTED_ELIGIBLE_CASES = 5_881
EXPECTED_ELIGIBLE_CONTROLS = 537_859

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Stage 9 Phase 1 — Diagnostic Investigation")
print("=" * 70)
print(f"\nLoading: {PARQUET_PATH}")
df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded:  {len(df):,} rows × {len(df.columns)} columns")

# ── 2. Schema confirmation ─────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 1 — Schema confirmation")
print("─" * 70)
print(f"\nAll columns ({len(df.columns)}):\n")
for col in df.columns:
    print(f"  {col:<40}  {df[col].dtype}")

# Verify expected column names
EXPECTED = {
    "outcome":         "y",
    "thread_id":       "thread_id",
    "author":          "author",          # could be author_id
    "A_social":        "A_social",
    "A_hypothetical":  "A_hypothetical",
}

# Detect actual names (case-insensitive fallback)
col_lower = {c.lower(): c for c in df.columns}

def find_col(candidates, label):
    for c in candidates:
        if c in df.columns:
            return c, False           # found as-is
    for c in candidates:
        if c.lower() in col_lower:
            return col_lower[c.lower()], True  # found case-insensitively
    return None, True

y_col,      y_flag      = find_col(["y", "outcome", "delta"], "outcome")
tid_col,    tid_flag    = find_col(["thread_id", "subreddit_id", "thread"], "thread_id")
auth_col,   auth_flag   = find_col(["author", "author_id"], "author")
asoc_col,   asoc_flag   = find_col(["A_social", "a_social", "A_Soc", "a_soc"], "A_social")
ahyp_col,   ahyp_flag   = find_col(["A_hypothetical", "a_hypothetical", "A_H", "a_h"], "A_hypothetical")

print("\nColumn name verification:")
for label, col, expected, flag in [
    ("outcome",        y_col,    "y",               y_flag),
    ("thread_id",      tid_col,  "thread_id",       tid_flag),
    ("author",         auth_col, "author",          auth_flag),
    ("A_social",       asoc_col, "A_social",        asoc_flag),
    ("A_hypothetical", ahyp_col, "A_hypothetical",  ahyp_flag),
]:
    status = "OK" if col == expected else "MISMATCH"
    flag_str = " *** FLAGGED" if (col != expected) else ""
    print(f"  {label:<18} expected={expected:<20} actual={str(col):<20} [{status}]{flag_str}")

if any(c is None for c in [y_col, tid_col, auth_col, asoc_col, ahyp_col]):
    print("\nFATAL: one or more required columns not found. Aborting.")
    sys.exit(1)

print(f"\nUsing: y='{y_col}', thread_id='{tid_col}', author='{auth_col}', "
      f"A_social='{asoc_col}', A_hypothetical='{ahyp_col}'")

# ── 3. Overall counts ─────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 2 — Overall counts")
print("─" * 70)

total_rows = len(df)
total_cases = int((df[y_col] == 1).sum())
total_controls = int((df[y_col] == 0).sum())

eligible_mask = df[asoc_col].notna() & df[ahyp_col].notna()
elig = df[eligible_mask]
elig_cases = int((elig[y_col] == 1).sum())
elig_controls = int((elig[y_col] == 0).sum())

print(f"\n  Total rows:                {total_rows:>10,}")
print(f"  Total cases (y=1):         {total_cases:>10,}")
print(f"  Total controls (y=0):      {total_controls:>10,}")
print(f"  Eligible cases:            {elig_cases:>10,}  "
      f"(both {asoc_col} & {ahyp_col} defined)")
print(f"  Eligible controls:         {elig_controls:>10,}  "
      f"(both {asoc_col} & {ahyp_col} defined)")

# Cross-check
row_ok   = total_rows    == EXPECTED_ROWS
ecase_ok = elig_cases    == EXPECTED_ELIGIBLE_CASES
ectrl_ok = elig_controls == EXPECTED_ELIGIBLE_CONTROLS

print(f"\n  Row count check:     expected {EXPECTED_ROWS:,}  "
      f"→ {'PASS' if row_ok else f'DISCREPANCY (got {total_rows:,})'}")
print(f"  Eligible cases check: expected {EXPECTED_ELIGIBLE_CASES:,}  "
      f"→ {'PASS' if ecase_ok else f'DISCREPANCY (got {elig_cases:,})'}")
print(f"  Eligible ctrl check:  expected {EXPECTED_ELIGIBLE_CONTROLS:,}  "
      f"→ {'PASS' if ectrl_ok else f'DISCREPANCY (got {elig_controls:,})'}")

# ── 4. Per-thread availability ─────────────────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 3 — Per-thread availability (eligible comments only)")
print("─" * 70)

thread_stats = (
    elig.groupby(tid_col)[y_col]
    .agg(
        n_case=lambda x: (x == 1).sum(),
        n_ctrl=lambda x: (x == 0).sum(),
    )
    .reset_index()
)

n_threads_eligible = len(thread_stats)
print(f"\n  Threads with ≥1 eligible comment: {n_threads_eligible:,}")

for col_name, label in [("n_case", "Eligible cases per thread"),
                         ("n_ctrl", "Eligible controls per thread")]:
    s = thread_stats[col_name]
    print(f"\n  {label}:")
    print(f"    min={s.min():.0f}  median={s.median():.0f}  mean={s.mean():.2f}  "
          f"p95={s.quantile(0.95):.0f}  max={s.max():.0f}")

# ── 5. Stranded cases ──────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 4 — Stranded cases (threads with zero eligible controls)")
print("─" * 70)

stranded_threads = thread_stats[thread_stats["n_ctrl"] == 0]
stranded_cases = int(stranded_threads["n_case"].sum())
pct_stranded = 100.0 * stranded_cases / EXPECTED_ELIGIBLE_CASES

print(f"\n  Threads with 0 eligible controls: {len(stranded_threads):,}")
print(f"  Eligible cases in those threads:  {stranded_cases:,}")
print(f"  As % of {EXPECTED_ELIGIBLE_CASES:,} eligible cases:   {pct_stranded:.2f}%")

if len(stranded_threads) > 0:
    examples = stranded_threads.nlargest(5, "n_case")[[tid_col, "n_case", "n_ctrl"]]
    print(f"\n  Top stranded threads (by n_case):")
    print(examples.to_string(index=False))

# ── 6. Projected yield at k=3 and k=5 ─────────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 5 — Projected yield and fallback (per-thread, variable-ratio rule)")
print("─" * 70)

def realized_k(n_case, n_ctrl, target_k):
    """Largest integer k' ≤ target_k such that n_ctrl ≥ k' * n_case; 0 if impossible."""
    if n_case == 0:
        return 0
    k = min(target_k, n_ctrl // n_case)
    return max(0, k)

for target_k in [3, 5]:
    col_rk = f"realized_k_{target_k}"
    thread_stats[col_rk] = thread_stats.apply(
        lambda r: realized_k(r["n_case"], r["n_ctrl"], target_k), axis=1
    )

# Save diagnostics CSV (the only file the script writes)
thread_stats.to_csv(DIAG_CSV, index=False)
print(f"\n  Diagnostics CSV saved: {DIAG_CSV}")

for target_k in [3, 5]:
    col_rk = f"realized_k_{target_k}"
    rk = thread_stats[col_rk]

    # Threads and cases where at least one match is possible (rk ≥ 1)
    matched_threads = thread_stats[rk >= 1]
    dropped_threads = thread_stats[rk == 0]

    cases_retained = int(matched_threads["n_case"].sum())
    cases_dropped  = int(dropped_threads["n_case"].sum())

    proj_controls = int((matched_threads["n_case"] * matched_threads[col_rk]).sum())

    fallback_threads = matched_threads[matched_threads[col_rk] < target_k]
    pct_fallback = 100.0 * len(fallback_threads) / len(matched_threads) if len(matched_threads) else 0.0

    print(f"\n  ── k={target_k} (target) ──────────────────────────────────────────────")

    # Realized-k distribution
    rk_dist = rk.value_counts().sort_index()
    print(f"\n  Realized-k distribution across all {len(thread_stats):,} strata:")
    for k_val, cnt in rk_dist.items():
        bar = "#" * min(50, int(50 * cnt / len(thread_stats)))
        print(f"    k={k_val}: {cnt:>6,} threads  {bar}")

    print(f"\n  Strata with realized k ≥ 1 (can match):  {len(matched_threads):,}")
    print(f"  Strata with realized k = 0 (dropped):     {len(dropped_threads):,}")
    print(f"  Cases retained (in matched strata):       {cases_retained:,}")
    print(f"  Cases dropped  (in k=0 strata):           {cases_dropped:,}")
    print(f"  Projected controls drawn:                 {proj_controls:,}")
    print(f"  Strata triggering fallback (rk < {target_k}):   {len(fallback_threads):,}  "
          f"({pct_fallback:.1f}% of matched strata)")

print("\n" + "=" * 70)
print("Investigation complete. No matching performed.")
print("=" * 70)
