"""
Stage 8 — Merge all sources into the pre-matching analysis dataset.

Join order (left from filtered comments):
  comments_filtered (JSONL) ← comment_indices (parquet, inner join on comment_id)
                            ← controls_final  (parquet, left join on comment_id)

Note: Stage 4 (author features) has been removed from the pipeline. None of the
three regression models (conditional logistic Model 1, mediation decomposition
Model 2, mixed-effects logistic Model 3) use author features as fixed-effect
predictors. Author identity enters Model 3 only through a random intercept, which
requires only the 'author' column already present from Stage 1.

Inner join with comment_indices drops zero-salience comments (already excluded by
Stage 7). Remaining NaN controls are kept as-is (handled by Stage 9 covariates).

Input:
  outputs/comments_filtered.jsonl
  outputs/comment_indices.parquet
  outputs/controls_final.parquet

Output:
  outputs/analysis_dataset_pre_matching.parquet

CONSORT keys written:
  pre_merge_comments, post_merge_rows
  (full CONSORT chain also written to consort/consort_pre_matching.json)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import ConsortCounter, get_logger, stream_jsonl


def main():
    parser = argparse.ArgumentParser(description="Stage 8: merge final pre-matching dataset")
    parser.add_argument("--comments-input", default="outputs/comments_filtered.jsonl")
    parser.add_argument("--indices-input", default="outputs/comment_indices.parquet")
    parser.add_argument("--controls-input", default="outputs/controls_final.parquet")
    parser.add_argument("--output", default="outputs/analysis_dataset_pre_matching.parquet")
    parser.add_argument("--counts-json", default="consort/stage8_counts.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    import pandas as pd

    logger = get_logger("merge_final_dataset", args.log_level)
    consort = ConsortCounter()

    logger.info("Loading filtered comments from %s", args.comments_input)
    comment_rows = list(stream_jsonl(args.comments_input))
    df_comments = pd.DataFrame(comment_rows)
    consort.set("pre_merge_comments", len(df_comments))
    logger.info("Comments: %d rows", len(df_comments))

    logger.info("Loading comment indices from %s", args.indices_input)
    df_indices = pd.read_parquet(args.indices_input)
    logger.info("Comment indices: %d rows", len(df_indices))

    logger.info("Loading controls from %s", args.controls_input)
    df_controls = pd.read_parquet(args.controls_input)
    logger.info("Controls: %d rows", len(df_controls))

    # Inner join: comments ← indices (drops zero-salience and unmatched)
    df = df_comments.merge(df_indices, on="comment_id", how="inner")
    logger.info("After inner join with indices: %d rows", len(df))

    # Left join: ← controls (controls may be missing for some comments; keep all)
    df = df.merge(df_controls, on="comment_id", how="left")
    logger.info("After left join with controls: %d rows", len(df))

    consort.set("post_merge_rows", len(df))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    consort.write(args.counts_json)

    # Write full pre-matching CONSORT chain
    consort_path = Path("consort/consort_pre_matching.json")
    consort_path.parent.mkdir(parents=True, exist_ok=True)
    with open(consort_path, "w", encoding="utf-8") as f:
        json.dump(consort.as_dict(), f, indent=2)

    logger.info("Final pre-matching dataset: %d rows, %d columns", len(df), len(df.columns))
    logger.info("Output: %s", args.output)


if __name__ == "__main__":
    main()
