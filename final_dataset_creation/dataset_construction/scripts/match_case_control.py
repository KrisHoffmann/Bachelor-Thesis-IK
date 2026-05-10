"""
Stage 9 — Within-thread matched case-control sampling.

Input:  outputs/analysis_dataset_pre_matching.parquet
Output: outputs/matched_dataset_ratio<N>.parquet  (N = 3 or 5)
        consort/matching_diagnostics_ratio<N>.json

Run twice:
  python match_case_control.py --ratio 3   # primary analysis
  python match_case_control.py --ratio 5   # robustness check

Algorithm:
  1. Group by thread_id.
  2. For each thread: n_delta = sum(y==1), n_nondelta = sum(y==0).
  3. Skip thread if n_delta == 0.
  4. k = min(ratio, n_nondelta // n_delta). Skip if k == 0.
  5. Sample k * n_delta non-delta comments without replacement (thread-local pool).
  6. Global used_ids guard: non-delta sampled in an earlier thread cannot be sampled again.
     (Defensive: comment_id is unique across threads, so cross-thread collision should not
     occur in practice, but guard ensures correctness regardless.)
  7. stratum_id = f"{thread_id}_{delta_comment_id}" for each delta comment with its k matches.
  8. k_realised = actual k used for this stratum (may be < requested ratio if pool was depleted).

Output columns: all input columns + [stratum_id, k_realised]

Diagnostics (consort/matching_diagnostics_ratio<N>.json):
  total_threads_with_delta, total_threads_excluded_k0,
  total_strata, total_comments_retained,
  k_distribution: {1: N, 2: N, 3: N},
  variable_k_fallback_proportion

Column names: y (0/1 delta label), thread_id, comment_id, author — all set in Stage 1.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import ConsortCounter, get_logger

COL_Y = "y"
COL_THREAD_ID = "thread_id"
COL_COMMENT_ID = "comment_id"
COL_AUTHOR_ID = "author"


def main():
    parser = argparse.ArgumentParser(description="Stage 9: within-thread matched case-control sampling")
    parser.add_argument("--input", default="outputs/analysis_dataset_pre_matching.parquet")
    parser.add_argument("--output", default=None, help="Output parquet path (default: outputs/matched_dataset_ratio<N>.parquet)")
    parser.add_argument("--ratio", type=int, default=3, choices=[3, 5], help="Control-to-case ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--counts-json", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    import json
    import numpy as np
    import pandas as pd

    if args.output is None:
        args.output = f"outputs/matched_dataset_ratio{args.ratio}.parquet"
    if args.counts_json is None:
        args.counts_json = f"consort/matching_diagnostics_ratio{args.ratio}.json"

    logger = get_logger("match_case_control", args.log_level)
    rng = np.random.default_rng(args.seed)

    logger.info("Loading pre-matching dataset from %s", args.input)
    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows", len(df))

    threads_with_delta = 0
    threads_excluded_k0 = 0
    total_strata = 0
    k_distribution: dict[int, int] = {}

    global_used_ids: set = set()
    output_rows: list[int] = []  # collect iloc indices

    # Stable thread ordering
    thread_groups = {tid: grp for tid, grp in df.groupby(COL_THREAD_ID, sort=True)}

    for thread_id, grp in thread_groups.items():
        delta_mask = grp[COL_Y] == 1
        nondelta_mask = grp[COL_Y] == 0
        n_delta = int(delta_mask.sum())
        n_nondelta = int(nondelta_mask.sum())

        if n_delta == 0:
            continue
        threads_with_delta += 1

        # Build thread-local non-delta pool (exclude globally used)
        nondelta_idxs = list(grp.index[nondelta_mask])
        available_idxs = [i for i in nondelta_idxs if df.loc[i, COL_COMMENT_ID] not in global_used_ids]

        k = min(args.ratio, len(available_idxs) // n_delta)
        if k == 0:
            threads_excluded_k0 += 1
            continue

        delta_idxs = list(grp.index[delta_mask])

        for delta_idx in delta_idxs:
            if len(available_idxs) < k:
                break
            # Sample k controls for this delta comment
            chosen = list(rng.choice(available_idxs, size=k, replace=False))
            chosen_ids = set(df.loc[chosen, COL_COMMENT_ID])

            stratum_id = f"{thread_id}_{df.loc[delta_idx, COL_COMMENT_ID]}"
            df.loc[delta_idx, "stratum_id"] = stratum_id
            df.loc[delta_idx, "k_realised"] = k
            output_rows.append(delta_idx)

            for ctrl_idx in chosen:
                df.loc[ctrl_idx, "stratum_id"] = stratum_id
                df.loc[ctrl_idx, "k_realised"] = k
                output_rows.append(ctrl_idx)

            global_used_ids.update(chosen_ids)
            available_idxs = [i for i in available_idxs if i not in chosen]
            total_strata += 1
            k_distribution[k] = k_distribution.get(k, 0) + 1

    df_matched = df.loc[output_rows].copy()
    total_retained = len(df_matched)

    variable_k_strata = sum(n for kv, n in k_distribution.items() if kv < args.ratio)
    variable_k_proportion = variable_k_strata / total_strata if total_strata > 0 else 0.0

    logger.info(
        "Matching done. Threads with delta: %d | excluded (k=0): %d | strata: %d | comments retained: %d",
        threads_with_delta, threads_excluded_k0, total_strata, total_retained,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_matched.to_parquet(args.output, index=False)
    logger.info("Wrote matched dataset to %s", args.output)

    diagnostics = {
        "ratio_requested": args.ratio,
        "total_threads_with_delta": threads_with_delta,
        "total_threads_excluded_k0": threads_excluded_k0,
        "total_strata": total_strata,
        "total_comments_retained": total_retained,
        "k_distribution": {str(k): v for k, v in sorted(k_distribution.items())},
        "variable_k_fallback_proportion": round(variable_k_proportion, 4),
    }
    Path(args.counts_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.counts_json, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info("Matching diagnostics written to %s", args.counts_json)


if __name__ == "__main__":
    main()
