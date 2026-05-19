"""
Stage 7 — Sentence-level CLT predictions → comment-level abstraction indices.

INPUT  : outputs/stage6_full_predictions.jsonl   (4,040,317 rows)
OUTPUT : outputs/stage7_comment_abstraction.parquet  (one row per comment_id)

Aggregation formula for each dimension d in {temporal, spatial, social, hypothetical}:
    S_d = { s : {d}_pred in {0, 1} }      # salient AND non-N/A sentences
    A_d = mean({d}_pred over S_d)         # Far-share (0 = all Near, 1 = all Far)
    If |S_d| == 0 → A_d = NaN (no imputation)

dim_pred encoding:
    null → not salient (excluded from S_d)
    -1   → N/A within salient (excluded from S_d)
     0   → Near  (enters mean as 0)
     1   → Far   (enters mean as 1)

Usage:
    python stage7_aggregate.py [--input PATH] [--output PATH] [--dry-run]

    --dry-run  processes only the first 100,000 rows (fast test, no spot-check)
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import polars as pl

DIMS = ("temporal", "spatial", "social", "hypothetical")
DRY_RUN_ROWS = 100_000


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    base = Path(__file__).parent
    p = argparse.ArgumentParser(
        description="Stage 7: aggregate CLT sentence predictions to comment level"
    )
    p.add_argument(
        "--input",
        default=str(base / "outputs" / "stage6_full_predictions.jsonl"),
    )
    p.add_argument(
        "--output",
        default=str(base / "outputs" / "stage7_comment_abstraction.parquet"),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Process only the first {DRY_RUN_ROWS:,} rows",
    )
    return p.parse_args()


# ── Reading ────────────────────────────────────────────────────────────────────

def read_jsonl(path: str, max_rows: int | None = None) -> pl.DataFrame:
    """
    Read a JSONL file into a polars DataFrame.

    For dry-run (max_rows is set) we buffer only the needed lines and parse
    that small slice — avoids reading 2.4 GB to get 100k rows.
    For the full run, polars.read_ndjson is used directly (fast, parallelised).

    Schema is always inferred. The nullable integer {dim}_pred columns would
    cause a panic if we forced Int8; polars infers Int64 (nullable) correctly.
    """
    if max_rows is not None:
        lines = []
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if line:
                    lines.append(line)
                if len(lines) >= max_rows:
                    break
        return pl.read_ndjson("\n".join(lines).encode())
    else:
        return pl.read_ndjson(path)


# ── Aggregation ────────────────────────────────────────────────────────────────

def add_masked_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    For each dim d, add Float32 column _mask_{d}:
      - float value of {d}_pred  when {d}_pred in {0, 1}
      - null                     otherwise (original null, or value == -1)

    Polars .mean() over a group skips nulls and returns null for an all-null
    group — which is the NaN-for-missing semantics we want.
    """
    exprs = []
    for d in DIMS:
        pred = f"{d}_pred"
        exprs.append(
            pl.when(pl.col(pred).is_in([0, 1]))
            .then(pl.col(pred).cast(pl.Float64))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(f"_mask_{d}")
        )
    return df.with_columns(exprs)


def aggregation_exprs() -> list:
    """
    Expressions for .group_by("comment_id").agg(...).

    The group key (comment_id) must NOT appear here — polars raises
    DuplicateError if it does.
    """
    exprs = [
        pl.col("thread_id").first(),
        pl.col("author").first(),
        pl.len().cast(pl.Int32).alias("n_sentences_total"),
        pl.col("salience_pred").sum().cast(pl.Int32).alias("n_salient_total"),
    ]
    for d in DIMS:
        exprs.append(pl.col(f"_mask_{d}").mean().alias(f"A_{d}"))
        exprs.append(
            pl.col(f"{d}_pred")
            .is_in([0, 1])
            .sum()
            .cast(pl.Int32)
            .alias(f"n_salient_on_{d}")
        )
    return exprs


# ── Spot-check ─────────────────────────────────────────────────────────────────

def manual_agg_comment(rows: list[dict]) -> dict[str, tuple[float, int]]:
    """Manual A_d computation from raw sentence-level dicts."""
    result = {}
    for d in DIMS:
        vals = [r[f"{d}_pred"] for r in rows if r.get(f"{d}_pred") in (0, 1)]
        n = len(vals)
        a = sum(vals) / n if n > 0 else float("nan")
        result[d] = (a, n)
    return result


def spotcheck(input_path: str, agg_df: pl.DataFrame, n_samples: int = 5, seed: int = 42) -> bool:
    """
    Sample n_samples comment_ids, re-read their rows from the JSONL,
    manually compute A_d, and compare against the aggregated parquet values.
    Returns True if all checks pass.
    """
    print("\n" + "=" * 72, flush=True)
    print(f"SPOT-CHECK — {n_samples} random comments  (seed={seed})", flush=True)
    print("=" * 72, flush=True)

    all_cids = agg_df["comment_id"].to_list()
    rng = random.Random(seed)
    sampled = rng.sample(all_cids, min(n_samples, len(all_cids)))
    cid_set = set(sampled)

    print(f"Sampled comment_ids: {sampled}", flush=True)
    print("Re-reading JSONL to collect sentence rows ...", flush=True)

    comment_rows: dict[str, list[dict]] = {cid: [] for cid in sampled}
    with open(input_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            obj = json.loads(line)
            if obj["comment_id"] in cid_set:
                comment_rows[obj["comment_id"]].append(obj)

    agg_rows = {r["comment_id"]: r for r in agg_df.filter(
        pl.col("comment_id").is_in(sampled)
    ).to_dicts()}

    all_ok = True
    for cid in sampled:
        rows = comment_rows[cid]
        manual = manual_agg_comment(rows)
        agg = agg_rows[cid]

        print(f"\n  comment_id : {cid}  (n_sentences={len(rows)}, "
              f"n_salient={agg['n_salient_total']})", flush=True)
        print(f"  {'dim':<14} {'manual_A':>12} {'agg_A':>12} "
              f"{'manual_n':>9} {'agg_n':>9} {'status':>9}", flush=True)
        print(f"  {'-'*14} {'-'*12} {'-'*12} {'-'*9} {'-'*9} {'-'*9}", flush=True)

        for d in DIMS:
            m_a, m_n = manual[d]
            a_a = agg[f"A_{d}"]
            a_n = agg[f"n_salient_on_{d}"]

            m_nan = math.isnan(m_a)
            a_nan = a_a is None or (isinstance(a_a, float) and math.isnan(a_a))

            if m_nan and a_nan:
                status = "NaN=NaN"
            elif m_nan or a_nan:
                status = "MISMATCH"
                all_ok = False
            elif abs(m_a - a_a) < 1e-5 and m_n == a_n:
                status = "OK"
            else:
                status = "MISMATCH"
                all_ok = False

            a_a_disp = float("nan") if a_nan else a_a
            print(f"  {d:<14} {m_a:>12.6f} {a_a_disp:>12.6f} "
                  f"{m_n:>9} {a_n:>9} {status:>9}", flush=True)

    print(flush=True)
    if all_ok:
        print("  Result: ALL spot-checks PASSED.", flush=True)
    else:
        print("  Result: WARNING — one or more spot-checks FAILED.", flush=True)
    print("=" * 72, flush=True)
    return all_ok


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    input_path  = args.input
    output_path = args.output
    dry_run     = args.dry_run

    t0 = time.time()

    print(f"polars {pl.__version__}", flush=True)
    print(f"Input:   {input_path}", flush=True)
    print(f"Output:  {output_path}", flush=True)
    print(f"Mode:    {'DRY-RUN (first 100,000 rows)' if dry_run else 'FULL CORPUS'}", flush=True)
    print(flush=True)

    # ── 1. Read ───────────────────────────────────────────────────────────────
    print("Reading ...", flush=True)
    t1 = time.time()
    df = read_jsonl(input_path, max_rows=DRY_RUN_ROWS if dry_run else None)
    n_rows = len(df)
    print(f"  {n_rows:,} rows read in {time.time()-t1:.1f}s", flush=True)

    # ── 2. Add masked Float64 columns for aggregation ─────────────────────────
    print("Masking dim columns ...", flush=True)
    df = add_masked_columns(df)

    # ── 3. GroupBy aggregation ────────────────────────────────────────────────
    print("Aggregating ...", flush=True)
    t3 = time.time()
    result = df.group_by("comment_id").agg(aggregation_exprs())
    n_comments = len(result)
    print(f"  {n_rows:,} rows → {n_comments:,} comments in {time.time()-t3:.1f}s", flush=True)

    # ── 4. Write parquet ──────────────────────────────────────────────────────
    print(f"Writing {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path, compression="snappy")
    out_bytes = Path(output_path).stat().st_size
    print(f"  {out_bytes / 1_048_576:.1f} MB written", flush=True)

    elapsed = time.time() - t0

    # ── 5. Validation summary ─────────────────────────────────────────────────
    print("\n" + "=" * 72, flush=True)
    print("VALIDATION SUMMARY", flush=True)
    print("=" * 72, flush=True)

    n_sent_sum    = result["n_sentences_total"].sum()
    n_salient_sum = result["n_salient_total"].sum()
    sent_ok = n_sent_sum == n_rows

    print(f"  Output comments:              {n_comments:>12,}")
    print(f"  n_sentences_total sum:        {n_sent_sum:>12,}  "
          f"(expected {n_rows:,}) {'PASS' if sent_ok else 'MISMATCH'}")
    print(f"  n_salient_total sum:          {n_salient_sum:>12,}")

    print()
    print(f"  {'dim':<14} {'n_missing':>10} {'miss%':>8}  {'mean_A':>9}  {'median_A':>9}")
    print(f"  {'-'*14} {'-'*10} {'-'*8}  {'-'*9}  {'-'*9}")

    defined_masks = []
    for d in DIMS:
        col = result[f"A_{d}"]
        n_null    = col.is_null().sum()
        n_nan     = col.is_nan().sum()
        n_missing = n_null + n_nan
        miss_pct  = n_missing / n_comments * 100
        valid     = col.drop_nulls().drop_nans()
        mean_a    = valid.mean()   if len(valid) > 0 else float("nan")
        median_a  = valid.median() if len(valid) > 0 else float("nan")
        print(f"  {d:<14} {n_missing:>10,} {miss_pct:>8.2f}%  {mean_a:>9.4f}  {median_a:>9.4f}")
        defined_masks.append(col.is_not_null() & ~col.is_nan())

    all4 = defined_masks[0]
    for m in defined_masks[1:]:
        all4 = all4 & m
    n_all4 = all4.sum()
    print(f"\n  Comments with all 4 A_d defined: {n_all4:>10,}  ({n_all4/n_comments*100:.2f}%)")
    print(f"  Wall time: {elapsed:.1f}s")
    print("=" * 72, flush=True)

    # ── 6. Spot-check (full run only) ─────────────────────────────────────────
    if not dry_run:
        spotcheck(input_path, result, n_samples=5, seed=42)
    else:
        print("\n(Spot-check omitted in --dry-run mode)", flush=True)

    print(f"\nDone.  {output_path}  ({out_bytes/1_048_576:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
