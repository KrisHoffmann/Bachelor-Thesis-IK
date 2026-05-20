"""
Stage 9 — Matched Case-Control Sampling
========================================
Builds two independent matched datasets from stage8_comment_full.parquet:
  - outputs/stage9_matched_1to3.parquet   (k=3 primary,    seed=42)
  - outputs/stage9_matched_1to5.parquet   (k=5 robustness, seed=43)

Algorithm: variable-ratio, within-thread, without replacement (dataset-wide).
  Stratum       = 1 Δ case (y=1) + its k matched non-Δ controls (y=0)
                  from the same thread_id.
  Variable-ratio = per thread: realized_k = min(target_k, floor(n_ctrl / n_case))
                  Threads with realized_k = 0 are excluded (their cases dropped).
  Within-thread draw order:
    Cases are sorted by comment_id (ascending, deterministic).
    Case i draws its realized_k controls from the thread's control pool
    AFTER cases 0..i-1 have already consumed their shares.
    This ensures the consumption order is auditable and reproducible under seed.

stratum_id: assigned as the case's own comment_id (unique per stratum).
  The handoff spec says "stratum_id = thread_id", but that is only correct for
  single-case threads. In multi-case threads each case is a distinct clogit
  stratum and must have a unique stratum_id. Using the case comment_id gives a
  stable, auditable, unique key per stratum. thread_id is retained as an original
  column and remains the basis for all within-thread sampling logic.

Seeds: k=3 → 42, k=5 → 43. Two distinct seeds so the samples are genuinely
independent (not nested). Both are documented in stage9_manifest.json.

Expected Phase-1 projections to assert against:
  Eligible cases: 5,881  |  Eligible controls: 537,859
  Cases retained: 5,849  |  Cases dropped: 32
  Controls drawn: k=3 → 17,247; k=5 → 28,041
  Realized-k distribution:
    k=3: {0:7571, 1:25, 2:55, 3:3476}
    k=5: {0:7571, 1:25, 2:55, 3:55, 4:80, 5:3341}
  Dropped-cases split: 5 (zero_controls) + 27 (insufficient_controls_floor_zero) = 32
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent
PARQUET_IN  = REPO_ROOT / "final_dataset_creation/dataset_construction/outputs/stage8_comment_full.parquet"
OUT_DIR     = REPO_ROOT / "outputs"

OUT_K3      = OUT_DIR / "stage9_matched_1to3.parquet"
OUT_K5      = OUT_DIR / "stage9_matched_1to5.parquet"
DROPPED_CSV = OUT_DIR / "stage9_dropped_cases.csv"
MANIFEST    = OUT_DIR / "stage9_manifest.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Phase-1 projections (assert against these) ─────────────────────────────────
EXPECTED_ROWS           = 598_272
EXPECTED_ELIG_CASES     = 5_881
EXPECTED_ELIG_CONTROLS  = 537_859
EXPECTED_RETAINED       = 5_849
EXPECTED_DROPPED        = 32
EXPECTED_DROPPED_ZERO   = 5    # threads with n_ctrl == 0
EXPECTED_DROPPED_FLOOR  = 27   # threads with n_ctrl > 0 but floor(n_ctrl/n_case) == 0
EXPECTED_CONTROLS_K3    = 17_247
EXPECTED_CONTROLS_K5    = 28_041
# Thread-level realized-k distributions (used in Step 3 to verify per-thread inventory)
EXPECTED_RK_K3_THREADS = {0: 7571, 1: 25, 2: 55, 3: 3476}
EXPECTED_RK_K5_THREADS = {0: 7571, 1: 25, 2: 55, 3: 55, 4: 80, 5: 3341}

# Strata-level realized-k distributions (used in V6; differs from thread-level
# because multi-case threads contribute n_case strata each at the same rk).
# k=3: threads {1:25,2:55,3:3476} × their n_case → strata {1:77,2:146,3:5626}
# k=5: threads {1:25,2:55,3:55,4:80,5:3341} × their n_case → strata {1:77,2:146,3:136,4:186,5:5304}
EXPECTED_RK_K3_STRATA = {1: 77, 2: 146, 3: 5626}
EXPECTED_RK_K5_STRATA = {1: 77, 2: 146, 3: 136, 4: 186, 5: 5304}

SEED_K3 = 42
SEED_K5 = 43

# ── CONSORT thread counts (first two are historical, last two recomputed) ──────
THREADS_TOPIC_FILTER    = 11_142   # Phase A topic filter output (historical)
THREADS_IN_PREDICTIONS  = 11_134   # unique thread_ids in stage6 predictions (historical)
# THREADS_GE1_ELIGIBLE and THREADS_GE1_CASE_CONTRIBUTING are recomputed below


def loud_assert(condition, message):
    if not condition:
        print(f"\nFATAL ASSERTION FAILED: {message}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load and schema re-verification
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Stage 9 — Matched Case-Control Sampling")
print("=" * 70)

print(f"\nLoading: {PARQUET_IN}")
df = pd.read_parquet(PARQUET_IN)
print(f"Loaded:  {len(df):,} rows × {len(df.columns)} columns\n")

print("─" * 70)
print("STEP 1 — Schema re-verification")
print("─" * 70)
print(f"\nAll columns ({len(df.columns)}):")
for col in df.columns:
    print(f"  {col:<40}  {df[col].dtype}")

REQUIRED = {"y", "thread_id", "author", "A_social", "A_hypothetical"}
missing  = REQUIRED - set(df.columns)
loud_assert(
    len(missing) == 0,
    f"Missing required columns: {missing}"
)
print(f"\nAll 5 required columns present: {sorted(REQUIRED)}")

loud_assert(len(df) == EXPECTED_ROWS,
            f"Row count mismatch: expected {EXPECTED_ROWS:,}, got {len(df):,}")
print(f"Row count OK: {len(df):,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Eligibility filter (rule iii)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 2 — Eligibility filter (rule iii)")
print("─" * 70)

elig_mask = df["A_social"].notna() & df["A_hypothetical"].notna()
elig      = df[elig_mask].copy()

elig_cases    = int((elig["y"] == 1).sum())
elig_controls = int((elig["y"] == 0).sum())

print(f"\n  Eligible rows:     {len(elig):,}")
print(f"  Eligible cases:    {elig_cases:,}")
print(f"  Eligible controls: {elig_controls:,}")

loud_assert(elig_cases    == EXPECTED_ELIG_CASES,
            f"Eligible cases: expected {EXPECTED_ELIG_CASES}, got {elig_cases}")
loud_assert(elig_controls == EXPECTED_ELIG_CONTROLS,
            f"Eligible controls: expected {EXPECTED_ELIG_CONTROLS}, got {elig_controls}")
print("  Assertions PASSED.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Per-thread inventory and realized-k calculation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 3 — Per-thread inventory and realized-k")
print("─" * 70)

thread_inv = (
    elig.groupby("thread_id")["y"]
    .agg(n_case=lambda x: (x == 1).sum(),
         n_ctrl=lambda x: (x == 0).sum())
    .reset_index()
)

# Recompute CONSORT thread counts from data
threads_ge1_eligible = int((thread_inv["n_case"] + thread_inv["n_ctrl"] >= 1).sum())
threads_contributing = int((thread_inv["n_case"] >= 1).sum())  # has ≥1 eligible case

print(f"\n  Threads with ≥1 eligible comment (CONSORT): {threads_ge1_eligible:,}")
print(f"  Threads with ≥1 eligible case (contributing): {threads_contributing:,}")


def compute_realized_k(n_case: int, n_ctrl: int, target_k: int) -> int:
    """Largest integer k' ≤ target_k s.t. n_ctrl ≥ k' * n_case; 0 if impossible."""
    if n_case == 0:
        return 0
    return max(0, min(target_k, n_ctrl // n_case))


for tk in [3, 5]:
    thread_inv[f"rk_{tk}"] = thread_inv.apply(
        lambda r: compute_realized_k(r["n_case"], r["n_ctrl"], tk), axis=1
    )

# Verify thread-level realized-k distributions against Phase 1 projections
for tk, expected_dist in [(3, EXPECTED_RK_K3_THREADS), (5, EXPECTED_RK_K5_THREADS)]:
    actual_dist = dict(thread_inv[f"rk_{tk}"].value_counts().sort_index())
    # Compare only thread-level counts (include k=0 threads with no cases)
    label = f"k={tk} rk-distribution"
    for k_val, exp_cnt in expected_dist.items():
        act_cnt = actual_dist.get(k_val, 0)
        loud_assert(act_cnt == exp_cnt,
                    f"{label}: rk={k_val} expected {exp_cnt}, got {act_cnt}")
    print(f"  Realized-k distribution for k={tk}: MATCHES Phase 1 projection")

# ── Dropped-cases analysis ─────────────────────────────────────────────────────
# Dropped = eligible cases in threads with rk=0 (same for both k values,
# since rk_3 and rk_5 both = 0 for the same threads when n_case>0 but
# floor(n_ctrl/n_case)=0).
# A thread with rk_3=0 AND n_case>0 is a drop thread.
drop_threads = thread_inv[(thread_inv["rk_3"] == 0) & (thread_inv["n_case"] > 0)].copy()

# Classify drop reason
drop_threads["drop_reason"] = np.where(
    drop_threads["n_ctrl"] == 0,
    "zero_controls",
    "insufficient_controls_floor_zero",
)

# Get the actual dropped case rows for the CSV
dropped_cases_list = []
for _, row in drop_threads.iterrows():
    tid = row["thread_id"]
    thread_cases = elig[(elig["thread_id"] == tid) & (elig["y"] == 1)][
        ["comment_id", "thread_id"]
    ].copy()
    thread_cases["n_case"]     = int(row["n_case"])
    thread_cases["n_ctrl"]     = int(row["n_ctrl"])
    thread_cases["drop_reason"] = row["drop_reason"]
    dropped_cases_list.append(thread_cases)

dropped_cases_df = pd.concat(dropped_cases_list, ignore_index=True)

n_dropped_total = len(dropped_cases_df)
n_dropped_zero  = int((dropped_cases_df["drop_reason"] == "zero_controls").sum())
n_dropped_floor = int((dropped_cases_df["drop_reason"] == "insufficient_controls_floor_zero").sum())

print(f"\n  Dropped cases: {n_dropped_total}  "
      f"(zero_controls: {n_dropped_zero}, floor_zero: {n_dropped_floor})")

loud_assert(n_dropped_total == EXPECTED_DROPPED,
            f"Total dropped: expected {EXPECTED_DROPPED}, got {n_dropped_total}")
loud_assert(n_dropped_zero  == EXPECTED_DROPPED_ZERO,
            f"Dropped (zero_controls): expected {EXPECTED_DROPPED_ZERO}, got {n_dropped_zero}")
loud_assert(n_dropped_floor == EXPECTED_DROPPED_FLOOR,
            f"Dropped (floor_zero): expected {EXPECTED_DROPPED_FLOOR}, got {n_dropped_floor}")
print("  Dropped-cases breakdown assertions PASSED.")

dropped_cases_df.to_csv(DROPPED_CSV, index=False)
print(f"  Saved: {DROPPED_CSV}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Matching function
# ══════════════════════════════════════════════════════════════════════════════

def build_matched_sample(elig: pd.DataFrame,
                         thread_inv: pd.DataFrame,
                         target_k: int,
                         seed: int) -> pd.DataFrame:
    """
    Draw a matched case-control sample.

    Within each thread with realized_k ≥ 1:
      1. Sort eligible cases by comment_id ascending (deterministic order).
      2. Shuffle the thread's control pool once under `seed`.
      3. Case i draws controls from index [i*rk : (i+1)*rk] in the shuffled pool,
         consuming controls sequentially so each control is used ≤ 1 time.

    Returns a DataFrame with all original stage8 columns plus:
      stratum_id, is_case, realized_k, is_fallback_stratum
    """
    rng = np.random.default_rng(seed)

    rk_col = f"rk_{target_k}"
    # Thread-level realized-k lookup
    rk_lookup = thread_inv.set_index("thread_id")[rk_col].to_dict()

    records = []   # list of DataFrames to concat

    # Only iterate over threads that contribute strata
    active_threads = thread_inv[
        (thread_inv["n_case"] >= 1) & (thread_inv[rk_col] >= 1)
    ]["thread_id"].tolist()

    for tid in active_threads:
        rk = rk_lookup[tid]
        thread_mask = elig["thread_id"] == tid

        # Cases: sorted deterministically by comment_id
        cases = (
            elig[thread_mask & (elig["y"] == 1)]
            .sort_values("comment_id")
            .copy()
        )
        n_case = len(cases)

        # Controls: shuffle once under seed, then slice sequentially
        ctrl_pool = (
            elig[thread_mask & (elig["y"] == 0)]
            .copy()
        )
        # Shuffle the control pool for this thread
        shuffled_idx = rng.permutation(len(ctrl_pool))
        ctrl_pool = ctrl_pool.iloc[shuffled_idx].reset_index(drop=True)

        is_fallback = int(rk < target_k)

        for i, (_, case_row) in enumerate(cases.iterrows()):
            # Case i gets controls at positions [i*rk : (i+1)*rk]
            start = i * rk
            end   = start + rk
            matched_ctrls = ctrl_pool.iloc[start:end].copy()

            # stratum_id = case's own comment_id (unique per stratum).
            # Multi-case threads yield multiple strata; thread_id is still
            # the sampling boundary and is retained as an original column.
            stratum_id = case_row["comment_id"]

            # Annotate case
            case_df = pd.DataFrame([case_row])
            case_df["stratum_id"]          = stratum_id
            case_df["is_case"]             = 1
            case_df["realized_k"]          = rk
            case_df["is_fallback_stratum"] = is_fallback

            # Annotate controls
            matched_ctrls["stratum_id"]          = stratum_id
            matched_ctrls["is_case"]             = 0
            matched_ctrls["realized_k"]          = rk
            matched_ctrls["is_fallback_stratum"] = is_fallback

            records.append(case_df)
            records.append(matched_ctrls)

    return pd.concat(records, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Build both samples
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 5 — Building matched samples")
print("─" * 70)

print(f"\n  Building k=3 sample (seed={SEED_K3}) ...")
matched_k3 = build_matched_sample(elig, thread_inv, target_k=3, seed=SEED_K3)
print(f"  k=3 sample rows: {len(matched_k3):,}")

print(f"\n  Building k=5 sample (seed={SEED_K5}) ...")
matched_k5 = build_matched_sample(elig, thread_inv, target_k=5, seed=SEED_K5)
print(f"  k=5 sample rows: {len(matched_k5):,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Validation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 6 — Validation")
print("─" * 70)

def validate_sample(df_m: pd.DataFrame, target_k: int,
                    expected_controls: int, expected_rk_dist: dict):
    label = f"k={target_k}"
    print(f"\n  ── {label} ────────────────────────────────────────────────────────")

    cases_m    = df_m[df_m["is_case"] == 1]
    controls_m = df_m[df_m["is_case"] == 0]
    n_cases    = len(cases_m)
    n_controls = len(controls_m)

    print(f"    Cases:    {n_cases:,}")
    print(f"    Controls: {n_controls:,}")
    print(f"    Total:    {len(df_m):,}")

    # V1 — Cases retained / dropped
    loud_assert(n_cases == EXPECTED_RETAINED,
                f"{label}: cases retained expected {EXPECTED_RETAINED}, got {n_cases}")
    cases_dropped_actual = EXPECTED_ELIG_CASES - n_cases
    loud_assert(cases_dropped_actual == EXPECTED_DROPPED,
                f"{label}: cases dropped expected {EXPECTED_DROPPED}, got {cases_dropped_actual}")
    print(f"    V1 PASS  Cases retained={n_cases:,}, dropped={cases_dropped_actual}")

    # V2 — Total controls vs Phase-1 projection
    loud_assert(n_controls == expected_controls,
                f"{label}: controls expected {expected_controls:,}, got {n_controls:,} "
                f"— without-replacement draw diverged from per-thread projection; STOP.")
    print(f"    V2 PASS  Controls drawn={n_controls:,} matches projection")

    # V3 — Every stratum has exactly 1 case and ≥1 control.
    # stratum_id = case's comment_id, so each stratum is unique per case.
    strata_case_counts = df_m.groupby("stratum_id")["is_case"].sum()
    bad_case_strata = (strata_case_counts != 1).sum()
    loud_assert(bad_case_strata == 0,
                f"{label}: {bad_case_strata} strata do not have exactly 1 case")
    strata_ctrl_count = controls_m.groupby("stratum_id").size()
    bad_ctrl_strata = (strata_ctrl_count < 1).sum()
    loud_assert(bad_ctrl_strata == 0,
                f"{label}: {bad_ctrl_strata} strata have 0 controls")
    print(f"    V3 PASS  Every stratum has exactly 1 case and ≥1 control")

    # V4 — No control comment_id appears in more than one stratum
    ctrl_stratum_counts = controls_m.groupby("comment_id")["stratum_id"].nunique()
    loud_assert((ctrl_stratum_counts == 1).all(),
                f"{label}: some control comment_ids appear in multiple strata")
    print(f"    V4 PASS  No control appears in more than one stratum")

    # V5 — All A_social and A_hypothetical non-null
    loud_assert(df_m["A_social"].notna().all(),
                f"{label}: A_social has nulls in output")
    loud_assert(df_m["A_hypothetical"].notna().all(),
                f"{label}: A_hypothetical has nulls in output")
    print(f"    V5 PASS  A_social and A_hypothetical fully non-null")

    # V6 — Strata-level realized-k distribution.
    # Each case row IS a stratum (stratum_id = case comment_id), so we count
    # realized_k values across case rows only.
    actual_rk_dist = dict(
        cases_m["realized_k"].value_counts().sort_index()
    )
    for k_val, exp_cnt in expected_rk_dist.items():
        act_cnt = actual_rk_dist.get(k_val, 0)
        loud_assert(act_cnt == exp_cnt,
                    f"{label}: realized_k={k_val} strata expected {exp_cnt}, got {act_cnt}")
    print(f"    V6 PASS  Strata-level realized-k distribution verified:")
    for k_val in sorted(actual_rk_dist):
        print(f"           rk={k_val}: {actual_rk_dist[k_val]:,} strata")

    # V7 — Strata count == cases retained (stratum_id is unique per case)
    n_strata = df_m["stratum_id"].nunique()
    loud_assert(n_strata == n_cases,
                f"{label}: n_strata ({n_strata}) != n_cases ({n_cases})")
    print(f"    V7 PASS  Strata count={n_strata:,} == cases retained")

    # Summary
    fb_strata = int((df_m.drop_duplicates("stratum_id")["is_fallback_stratum"] == 1).sum())
    print(f"    Summary: {n_cases:,} cases, {n_controls:,} controls, "
          f"{n_strata:,} threads, {fb_strata:,} fallback strata")

    return {
        "cases_retained":         n_cases,
        "cases_dropped":          cases_dropped_actual,
        "controls_drawn":         n_controls,
        "strata_count":           n_strata,
        "fallback_strata":        fb_strata,
        "realized_k_dist_strata": {str(k): v for k, v in actual_rk_dist.items()},
    }

stats_k3 = validate_sample(matched_k3, 3, EXPECTED_CONTROLS_K3, EXPECTED_RK_K3_STRATA)
stats_k5 = validate_sample(matched_k5, 5, EXPECTED_CONTROLS_K5, EXPECTED_RK_K5_STRATA)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Write outputs
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 7 — Writing outputs")
print("─" * 70)

matched_k3.to_parquet(OUT_K3, index=False)
print(f"\n  Saved: {OUT_K3}  ({len(matched_k3):,} rows)")

matched_k5.to_parquet(OUT_K5, index=False)
print(f"  Saved: {OUT_K5}  ({len(matched_k5):,} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Manifest
# ══════════════════════════════════════════════════════════════════════════════
manifest = {
    "description": "Stage 9 matched case-control sampling manifest",
    "stratum_id_note": (
        "stratum_id = case comment_id (unique per stratum). "
        "Multi-case threads produce multiple strata; thread_id is retained as "
        "an original column and is the clogit strata() argument in R."
    ),
    "input": str(PARQUET_IN),
    "eligible_cases":    EXPECTED_ELIG_CASES,
    "eligible_controls": EXPECTED_ELIG_CONTROLS,
    "seeds": {"k3": SEED_K3, "k5": SEED_K5},
    "consort_thread_counts": {
        "threads_topic_filter":      THREADS_TOPIC_FILTER,
        "threads_in_predictions":    THREADS_IN_PREDICTIONS,
        "threads_ge1_eligible_comment": threads_ge1_eligible,
        "threads_ge1_eligible_case_contributing": threads_contributing,
    },
    "k3": {
        "target_k": 3,
        "seed": SEED_K3,
        "output": str(OUT_K3),
        **stats_k3,
    },
    "k5": {
        "target_k": 5,
        "seed": SEED_K5,
        "output": str(OUT_K5),
        **stats_k5,
    },
    "dropped_cases": {
        "total":                         n_dropped_total,
        "zero_controls":                 n_dropped_zero,
        "insufficient_controls_floor_zero": n_dropped_floor,
        "csv": str(DROPPED_CSV),
    },
}

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

with open(MANIFEST, "w") as f:
    json.dump(manifest, f, indent=2, cls=_NpEncoder)
print(f"  Saved: {MANIFEST}")


# ══════════════════════════════════════════════════════════════════════════════
# Final summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Stage 9 complete — all assertions passed")
print("=" * 70)
print(f"""
  Input:           {PARQUET_IN.name}  ({EXPECTED_ROWS:,} rows)
  Eligible cases:  {EXPECTED_ELIG_CASES:,}  |  Eligible controls: {EXPECTED_ELIG_CONTROLS:,}
  Dropped cases:   {n_dropped_total} (zero_ctrl={n_dropped_zero}, floor_zero={n_dropped_floor})

  k=3 (seed={SEED_K3}):  {stats_k3['cases_retained']:,} cases + {stats_k3['controls_drawn']:,} controls
           = {stats_k3['cases_retained'] + stats_k3['controls_drawn']:,} rows  |  fallback strata: {stats_k3['fallback_strata']:,}
           → {OUT_K3.name}

  k=5 (seed={SEED_K5}):  {stats_k5['cases_retained']:,} cases + {stats_k5['controls_drawn']:,} controls
           = {stats_k5['cases_retained'] + stats_k5['controls_drawn']:,} rows  |  fallback strata: {stats_k5['fallback_strata']:,}
           → {OUT_K5.name}

  Audit files:
    {DROPPED_CSV.name}
    {MANIFEST.name}
""")
