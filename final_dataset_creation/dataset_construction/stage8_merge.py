"""
Stage 8 — Merge comment-level abstraction indices with linguistic controls
          and the outcome variable.

INPUTS:
  outputs/stage7_comment_abstraction.parquet  (598,272 rows)
  outputs/comments_controls.parquet           (598,272 rows)
  outputs/comments_filtered.jsonl             (598,272 rows; 6 fields extracted)

OUTPUT:
  outputs/stage8_comment_full.parquet         (598,272 rows)

Usage:
  python stage8_merge.py [--abstraction PATH] [--controls PATH]
                         [--filtered PATH]  [--output PATH]
"""

import argparse
import json
import random
import time
from pathlib import Path

import polars as pl

EXPECTED_ROWS   = 598_272
EXPECTED_Y1     = 6_013
EXPECTED_Y0     = EXPECTED_ROWS - EXPECTED_Y1          # 592,259
EXPECTED_PRIMARY = 543_740   # both A_social & A_hypothetical defined

JSONL_FIELDS = {"comment_id", "y", "created_utc", "score", "level", "parent_id"}

COLUMN_ORDER = [
    # identifiers
    "comment_id", "thread_id", "author", "parent_id",
    # outcome
    "y",
    # timing / engagement
    "created_utc", "score", "level",
    # abstraction indices
    "A_temporal", "A_spatial", "A_social", "A_hypothetical",
    # denomination counts
    "n_salient_on_temporal", "n_salient_on_spatial",
    "n_salient_on_social", "n_salient_on_hypothetical",
    # sentence counts
    "n_sentences_total", "n_salient_total",
    # linguistic controls (order preserved from comments_controls.parquet)
    "noun_to_verb_ratio", "type_token_ratio", "flesch_kincaid",
    "hedge_density", "mean_sentence_length", "mean_word_length",
    "punctuation_density", "paragraph_count",
    "num_sentences", "num_tokens", "num_word_tokens", "num_urls",
]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    base = Path(__file__).parent / "outputs"
    p = argparse.ArgumentParser(description="Stage 8: merge abstraction, controls, and outcome")
    p.add_argument("--abstraction", default=str(base / "stage7_comment_abstraction.parquet"))
    p.add_argument("--controls",    default=str(base / "comments_controls.parquet"))
    p.add_argument("--filtered",    default=str(base / "comments_filtered.jsonl"))
    p.add_argument("--output",      default=str(base / "stage8_comment_full.parquet"))
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def assert_eq(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"ASSERTION FAILED — {label}: expected {expected:,}, got {actual:,}")
    print(f"  {label}: {actual:,}  OK", flush=True)


def anti_join_report(df_a, df_b, label_a, label_b):
    """Print how many comment_ids are in a but not b, and vice versa."""
    cids_a = set(df_a["comment_id"].to_list())
    cids_b = set(df_b["comment_id"].to_list())
    only_a = cids_a - cids_b
    only_b = cids_b - cids_a
    print(f"  {label_a} only (not in {label_b}): {len(only_a):,}"
          + (f"  examples: {list(only_a)[:5]}" if only_a else ""), flush=True)
    print(f"  {label_b} only (not in {label_a}): {len(only_b):,}"
          + (f"  examples: {list(only_b)[:5]}" if only_b else ""), flush=True)
    return len(only_a), len(only_b)


def defined(col: pl.Series) -> pl.Series:
    """Boolean mask: not null and not NaN."""
    if col.dtype in (pl.Float32, pl.Float64):
        return col.is_not_null() & ~col.is_nan()
    return col.is_not_null()


# ── Step 3: stream JSONL ───────────────────────────────────────────────────────

def stream_jsonl_fields(path: str) -> pl.DataFrame:
    """
    Stream comments_filtered.jsonl and extract only JSONL_FIELDS.
    Avoids loading the full 525 MB (body text etc.) into memory.

    Fields from the Reddit API dump:
      comment_id  str   parent_id   str
      y           int   level       int
      created_utc int   score       int
    Let each list hold native Python types and rely on polars inference.
    """
    fields_sorted = sorted(JSONL_FIELDS)
    records: dict[str, list] = {f: [] for f in fields_sorted}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            obj = json.loads(line)
            for field in fields_sorted:
                records[field].append(obj.get(field))

    # Build with strict=False so mixed-type columns (if any) survive inference
    df = pl.DataFrame(records, strict=False)
    # Normalise dtypes: y → Int8, integer fields → Int32, utc → Int64 (unix ts)
    df = df.with_columns([
        pl.col("y").cast(pl.Int8),
        pl.col("score").cast(pl.Int32),
        pl.col("level").cast(pl.Int32),
        pl.col("created_utc").cast(pl.Int64),
    ])
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = time.time()

    print(f"polars {pl.__version__}", flush=True)
    print(f"abstraction : {args.abstraction}", flush=True)
    print(f"controls    : {args.controls}", flush=True)
    print(f"filtered    : {args.filtered}", flush=True)
    print(f"output      : {args.output}", flush=True)
    print(flush=True)

    # ── 1. Load parquets ──────────────────────────────────────────────────────
    print("Loading stage7_comment_abstraction.parquet ...", flush=True)
    df_agg = pl.read_parquet(args.abstraction)
    print(f"  {len(df_agg):,} rows, {len(df_agg.columns)} cols", flush=True)

    print("Loading comments_controls.parquet ...", flush=True)
    df_ctrl = pl.read_parquet(args.controls)
    print(f"  {len(df_ctrl):,} rows, {len(df_ctrl.columns)} cols", flush=True)

    # ── 3. Stream JSONL ───────────────────────────────────────────────────────
    print("Streaming comments_filtered.jsonl (extracting 6 fields) ...", flush=True)
    t_jsonl = time.time()
    df_y = stream_jsonl_fields(args.filtered)
    print(f"  {len(df_y):,} rows in {time.time()-t_jsonl:.1f}s", flush=True)

    # ── 4. Inner joins ────────────────────────────────────────────────────────
    print("Joining ...", flush=True)
    t_join = time.time()

    merged = (
        df_agg
        .join(df_ctrl, on="comment_id", how="inner")
        .join(df_y,    on="comment_id", how="inner")
    )
    n_merged = len(merged)
    print(f"  {n_merged:,} rows after join in {time.time()-t_join:.1f}s", flush=True)

    # ── 5. Assert row count ───────────────────────────────────────────────────
    if n_merged != EXPECTED_ROWS:
        print("\nJOIN ROW COUNT MISMATCH — diagnosing ...", flush=True)
        anti_join_report(df_agg,  df_ctrl, "stage7_abstraction", "comments_controls")
        anti_join_report(df_agg,  df_y,    "stage7_abstraction", "filtered_jsonl")
        anti_join_report(df_ctrl, df_y,    "comments_controls",  "filtered_jsonl")
        raise AssertionError(
            f"Expected {EXPECTED_ROWS:,} rows after inner join, got {n_merged:,}."
        )

    # ── 6. y dtype ────────────────────────────────────────────────────────────
    if merged["y"].dtype not in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8):
        merged = merged.with_columns(pl.col("y").cast(pl.Int8))

    # ── 7. Column reorder ─────────────────────────────────────────────────────
    # Verify all expected columns are present before reordering
    missing = [c for c in COLUMN_ORDER if c not in merged.columns]
    if missing:
        raise ValueError(f"Expected columns missing from merged DataFrame: {missing}")
    extra = [c for c in merged.columns if c not in COLUMN_ORDER]
    if extra:
        print(f"  Note: extra columns not in COLUMN_ORDER (appended at end): {extra}", flush=True)
        final_order = COLUMN_ORDER + extra
    else:
        final_order = COLUMN_ORDER
    merged = merged.select(final_order)

    # ── 8. Write parquet ──────────────────────────────────────────────────────
    print(f"Writing {args.output} ...", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(args.output, compression="snappy")
    out_bytes = Path(args.output).stat().st_size
    print(f"  {out_bytes / 1_048_576:.1f} MB written", flush=True)

    elapsed = time.time() - t0

    # ── Validation ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72, flush=True)
    print("POST-MERGE VALIDATION", flush=True)
    print("=" * 72, flush=True)

    assert_eq("output row count", n_merged, EXPECTED_ROWS)

    y1 = int((merged["y"] == 1).sum())
    y0 = int((merged["y"] == 0).sum())
    assert_eq("y=1 count", y1, EXPECTED_Y1)
    assert_eq("y=0 count", y0, EXPECTED_Y0)

    print(flush=True)
    print("  Null counts per column:", flush=True)
    any_unexpected_nulls = False
    for col in merged.columns:
        s = merged[col]
        n_null = s.null_count()
        n_nan  = s.is_nan().sum() if s.dtype in (pl.Float32, pl.Float64) else 0
        n_miss = n_null + n_nan
        flag = ""
        # unexpected = non-float column with any nulls, or known-clean column with nulls
        if n_null > 0 and s.dtype not in (pl.Float32, pl.Float64):
            flag = "  <-- UNEXPECTED NULL"
            any_unexpected_nulls = True
        if n_miss > 0:
            print(f"    {col:<35} nulls={n_null:>7,}  nans={n_nan:>7,}{flag}", flush=True)
    if not any_unexpected_nulls:
        print("    (no unexpected nulls in non-float columns)", flush=True)

    # Primary sample and delta retention
    print(flush=True)
    soc_def = defined(merged["A_social"])
    hyp_def = defined(merged["A_hypothetical"])
    primary = (soc_def & hyp_def).sum()
    assert_eq("primary sample (A_soc & A_hyp defined)", primary, EXPECTED_PRIMARY)

    delta_primary = int(((merged["y"] == 1) & soc_def & hyp_def).sum())
    delta_pct = delta_primary / y1 * 100
    print(f"  Δ into primary sample: {delta_primary:,} / {y1:,}  ({delta_pct:.1f}% of cases)",
          flush=True)

    # Random sample rows
    print(flush=True)
    print("  Random row sample (seed=99):", flush=True)
    rng = random.Random(99)
    idxs = rng.sample(range(n_merged), 3)
    for idx in idxs:
        row = merged.row(idx, named=True)
        print(f"\n    idx={idx}  comment_id={row['comment_id']}  y={row['y']}", flush=True)
        for col in COLUMN_ORDER:
            v = row[col]
            print(f"      {col:<35} {v}", flush=True)

    # Spot-check: one comment_id's A_social in stage7 vs stage8
    print(flush=True)
    print("  Spot-check A_social consistency (stage7 → stage8):", flush=True)
    spot_cid = merged["comment_id"][0]
    a_social_stage7 = df_agg.filter(pl.col("comment_id") == spot_cid)["A_social"][0]
    a_social_stage8 = merged.filter(pl.col("comment_id") == spot_cid)["A_social"][0]
    match = (a_social_stage7 is None and a_social_stage8 is None) or (
        a_social_stage7 is not None
        and a_social_stage8 is not None
        and abs(a_social_stage7 - a_social_stage8) < 1e-9
    )
    print(f"    comment_id: {spot_cid}", flush=True)
    print(f"    A_social stage7: {a_social_stage7}", flush=True)
    print(f"    A_social stage8: {a_social_stage8}", flush=True)
    print(f"    match: {'OK' if match else 'MISMATCH'}", flush=True)

    print(flush=True)
    print(f"  Wall time: {elapsed:.1f}s", flush=True)
    print(f"  Output:    {args.output}  ({out_bytes/1_048_576:.1f} MB)", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
