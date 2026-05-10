"""
Stage 7 — Aggregate sentence-level CLT predictions to comment-level indices.

Input:  outputs/sentence_predictions.parquet
        Columns: [comment_id, sentence_index, label_T, label_S, label_Soc, label_H]
        Each label ∈ {-1, 0, 1}  (-1 = not salient on dimension)

Output: outputs/comment_indices.parquet
        Columns: [comment_id, A_T, A_S, A_Soc, A_H,
                  num_salient_T, num_salient_S, num_salient_Soc, num_salient_H]

Aggregation formula:
  A_d = mean(label_d for sentences where label_d ∈ {0, 1})
  A_d = NaN if no sentence has label_d ∈ {0, 1}

Zero-salience exclusion:
  Drop comments where ALL four A_d values are NaN.

CONSORT keys written:
  comments_with_predictions, after_remove_zero_salience
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import ConsortCounter, get_logger

DIMENSIONS = ["T", "S", "Soc", "H"]
LABEL_COLS = {d: f"label_{d}" for d in DIMENSIONS}


def main():
    parser = argparse.ArgumentParser(description="Stage 7: aggregate sentence predictions to comment indices")
    parser.add_argument("--input", default="outputs/sentence_predictions.parquet")
    parser.add_argument("--output", default="outputs/comment_indices.parquet")
    parser.add_argument("--counts-json", default="consort/stage7_counts.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    logger = get_logger("aggregate_indices", args.log_level)
    consort = ConsortCounter()

    logger.info("Loading sentence predictions from %s", args.input)
    df = pd.read_parquet(args.input)
    n_before = df["comment_id"].nunique()
    consort.set("comments_with_predictions", n_before)
    logger.info("Loaded %d sentence rows across %d comments", len(df), n_before)

    rows = []
    for comment_id, group in df.groupby("comment_id", sort=False):
        row = {"comment_id": comment_id}
        all_nan = True
        for d in DIMENSIONS:
            col = LABEL_COLS[d]
            if col not in group.columns:
                row[f"A_{d}"] = float("nan")
                row[f"num_salient_{d}"] = 0
                continue
            salient_mask = group[col].isin([0, 1])
            salient_vals = group.loc[salient_mask, col]
            n_salient = int(salient_mask.sum())
            row[f"num_salient_{d}"] = n_salient
            if n_salient > 0:
                row[f"A_{d}"] = float(salient_vals.mean())
                all_nan = False
            else:
                row[f"A_{d}"] = float("nan")

        if not all_nan:
            rows.append(row)

    df_out = pd.DataFrame(rows)
    n_after = len(df_out)
    removed = n_before - n_after
    consort.set("after_remove_zero_salience", n_after)
    logger.info("Zero-salience exclusion: removed %d comments → %d remaining", removed, n_after)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.output, index=False)
    consort.write(args.counts_json)
    logger.info("Wrote %d comment index rows to %s", n_after, args.output)


if __name__ == "__main__":
    main()
