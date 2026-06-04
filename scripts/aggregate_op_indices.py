"""
aggregate_op_indices.py
=======================
Aggregate sentence-level CLT predictions for OP selftexts into per-thread
abstraction indices A_Soc_OP and A_H_OP.

Input
-----
  final_dataset_creation/outputs/op_selftexts/op_predictions.jsonl
      Output of stage6_inference.py on OP sentences.
      Relevant fields per row:
        thread_id       str
        social_pred     int or null   (-1 = N/A, 0 = Near, 1 = Far, null = failed)
        hypothetical_pred int or null

Aggregation formula (identical to aggregate_indices.py / Stage 7):
  A_d = mean(pred for sentences where pred in {0, 1})
  A_d = NaN if no sentence has pred in {0, 1}

Output
------
  final_dataset_creation/outputs/op_selftexts/op_abstraction_indices.parquet
      One row per thread_id, columns:
        thread_id, A_Soc_OP, A_H_OP,
        n_salient_social_OP, n_salient_hypothetical_OP, n_sentences_OP

Run from the repo root after the Habrok inference job completes:
    python scripts/aggregate_op_indices.py
"""

import json
import pathlib
import sys
from collections import defaultdict

import pandas as pd
import numpy as np

INPUT_FILE  = pathlib.Path(
    "final_dataset_creation/outputs/op_selftexts/op_predictions.jsonl"
)
OUTPUT_FILE = pathlib.Path(
    "final_dataset_creation/outputs/op_selftexts/op_abstraction_indices.parquet"
)


def main() -> None:
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: predictions file not found: {INPUT_FILE}\n"
                 "Run sbatch slurm/stage6_op_selftext.sbatch first.")

    # Accumulate per-thread salient values
    soc_vals: dict[str, list[int]]  = defaultdict(list)
    hyp_vals: dict[str, list[int]]  = defaultdict(list)
    n_sents:  dict[str, int]        = defaultdict(int)

    print(f"Reading {INPUT_FILE} …")
    n_rows = 0
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec["thread_id"]
            n_sents[tid] += 1
            n_rows += 1

            soc = rec.get("social_pred")
            hyp = rec.get("hypothetical_pred")

            if soc in (0, 1):
                soc_vals[tid].append(soc)
            if hyp in (0, 1):
                hyp_vals[tid].append(hyp)

    print(f"Read {n_rows:,} sentence rows across {len(n_sents):,} threads.")

    all_tids = sorted(n_sents.keys())
    rows = []
    for tid in all_tids:
        sv = soc_vals[tid]
        hv = hyp_vals[tid]
        rows.append({
            "thread_id":              tid,
            "A_Soc_OP":               float(np.mean(sv)) if sv else float("nan"),
            "A_H_OP":                 float(np.mean(hv)) if hv else float("nan"),
            "n_salient_social_OP":    len(sv),
            "n_salient_hypothetical_OP": len(hv),
            "n_sentences_OP":         n_sents[tid],
        })

    df = pd.DataFrame(rows)

    n_soc_def = df["A_Soc_OP"].notna().sum()
    n_hyp_def = df["A_H_OP"].notna().sum()
    print(f"A_Soc_OP defined for {n_soc_def:,} / {len(df):,} threads "
          f"({100*n_soc_def/len(df):.1f}%)")
    print(f"A_H_OP  defined for {n_hyp_def:,} / {len(df):,} threads "
          f"({100*n_hyp_def/len(df):.1f}%)")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print("\nNext step: the file is ready for clt_fit_interaction.R")
    print(f"  Join key: thread_id")
    print(f"  New columns: A_Soc_OP, A_H_OP")


if __name__ == "__main__":
    main()
