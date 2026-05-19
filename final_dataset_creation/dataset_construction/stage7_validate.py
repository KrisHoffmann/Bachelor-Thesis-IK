"""
Stage 7 validation: streaming diagnostic pass over stage6_full_predictions.jsonl.
Run from inside WSL on the native Linux filesystem for best performance.
Usage: python stage7_validate.py [--input PATH] [--output PATH]
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import orjson as _json_lib
    def _loads(s):
        return _json_lib.loads(s)
    JSON_BACKEND = "orjson"
except ImportError:
    def _loads(s):
        return json.loads(s)
    JSON_BACKEND = "stdlib json"

DIMS = ("temporal", "spatial", "social", "hypothetical")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(
        Path(__file__).parent / "outputs" / "stage6_full_predictions.jsonl"))
    parser.add_argument("--output", default=str(
        Path(__file__).parent / "outputs" / "stage7_validation_report.json"))
    args = parser.parse_args()

    input_path = args.input
    print(f"JSON backend: {JSON_BACKEND}", flush=True)
    print(f"Input:  {input_path}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(flush=True)

    total_rows = 0
    parse_errors = 0
    comment_ids = set()
    thread_ids = set()
    sal0 = 0
    sal1 = 0

    # dim_pred distributions over salient rows: {dim: {-1:n, 0:n, 1:n}}
    dim_dist = {d: {-1: 0, 0: 0, 1: 0} for d in DIMS}

    # schema violation counters
    v_salient_null = defaultdict(int)      # salient row but dim_pred is None
    v_nonsalient_nonnull = defaultdict(int) # non-salient row but dim_pred not None

    # per-comment sentence tracking
    # comment_id -> n_sentences_in_comment (as declared)
    comment_declared_n = {}
    # comment_id -> set of sentence_indexes seen
    comment_idx_seen = defaultdict(set)

    t0 = time.time()
    report_every = 500_000

    with open(input_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            total_rows += 1
            if total_rows % report_every == 0:
                elapsed = time.time() - t0
                rate = total_rows / elapsed
                print(f"  ... {total_rows:,} rows  ({rate:,.0f} rows/s  {elapsed:.1f}s elapsed)", flush=True)

            line = raw_line.rstrip("\n")
            if not line:
                continue
            try:
                obj = _loads(line)
            except Exception:
                parse_errors += 1
                continue

            cid = obj["comment_id"]
            tid = obj["thread_id"]
            sp  = obj["salience_pred"]
            si  = obj["sentence_index"]
            ns  = obj["n_sentences_in_comment"]

            comment_ids.add(cid)
            thread_ids.add(tid)

            if sp == 0:
                sal0 += 1
            else:
                sal1 += 1

            # record declared sentence count (take first seen; should be constant)
            if cid not in comment_declared_n:
                comment_declared_n[cid] = ns
            # track which indexes we've seen
            comment_idx_seen[cid].add(si)

            if sp == 1:
                for d in DIMS:
                    pred = obj.get(f"{d}_pred")
                    if pred is None:
                        v_salient_null[d] += 1
                    elif pred in (-1, 0, 1):
                        dim_dist[d][pred] += 1
                    # unexpected value: don't count in distribution
            else:
                for d in DIMS:
                    pred = obj.get(f"{d}_pred")
                    if pred is not None:
                        v_nonsalient_nonnull[d] += 1

    elapsed = time.time() - t0
    print(f"\nFull pass complete in {elapsed:.1f}s ({total_rows / elapsed:,.0f} rows/s)\n", flush=True)

    # ── Sentences-per-comment distribution ───────────────────────────────────
    spc_values = sorted(comment_declared_n.values())
    n_comments = len(spc_values)
    spc_min    = spc_values[0]
    spc_max    = spc_values[-1]
    spc_mean   = sum(spc_values) / n_comments
    spc_median = spc_values[n_comments // 2]
    spc_p95    = spc_values[int(n_comments * 0.95)]
    spc_p99    = spc_values[int(n_comments * 0.99)]

    # ── Sentence-index integrity ──────────────────────────────────────────────
    gap_comments = []   # actual < declared (gaps)
    dup_comments = []   # duplicate indexes within a comment
    for cid, idx_set in comment_idx_seen.items():
        declared = comment_declared_n[cid]
        actual   = len(idx_set)
        # check for duplicates: if actual sentences > unique indexes
        # (we can only detect gaps/dups relative to declared n)
        if actual < declared:
            gap_comments.append(cid)
        # duplicate indexes: if total sentences seen > unique indexes seen
        # We'd need per-comment counts — re-derive from total rows minus unique
    # For dup detection we need raw counts; approximate with declared vs unique
    n_gap = len(gap_comments)
    # True dup check: total sentences per comment vs unique sentence_indexes
    # We tracked sets so we can't count actual dups without a separate pass
    # Flag: if any comment has more actual rows than declared, that signals dups
    # (We can only tell if comment_id rows > n_sentences_in_comment)
    # Better: compare total_rows sum to sum of declared n per comment
    total_declared = sum(comment_declared_n.values())

    # ── Print report ─────────────────────────────────────────────────────────
    def pct(n, total): return f"{n / total * 100:.2f}%"

    print("=" * 60)
    print("PHASE 1 VALIDATION REPORT")
    print("=" * 60)
    print(f"Input file:             {input_path}")
    print(f"File size (bytes):      {os.path.getsize(input_path):,}")
    print(f"Total rows:             {total_rows:,}")
    print(f"Parse errors:           {parse_errors}")
    print(f"Unique comment_ids:     {len(comment_ids):,}")
    print(f"Unique thread_ids:      {len(thread_ids):,}")
    print()
    print("── Salience distribution ──")
    print(f"  salience_pred=0 (not salient): {sal0:>10,}  ({pct(sal0, total_rows)})")
    print(f"  salience_pred=1 (salient):     {sal1:>10,}  ({pct(sal1, total_rows)})")
    print()
    print("── dim_pred distributions (over salient rows, n={:,}) ──".format(sal1))
    for d in DIMS:
        neg1 = dim_dist[d][-1]
        z    = dim_dist[d][0]
        one  = dim_dist[d][1]
        non_na = z + one
        far_pct = f"{one / non_na * 100:.2f}%" if non_na > 0 else "N/A"
        print(f"  {d:>12s}: N/A=-1: {neg1:>8,}  Near=0: {z:>8,}  Far=1: {one:>8,}  "
              f"(non-NA: {non_na:,}  Far-share: {far_pct})")
    print()
    print("── Schema violations ──")
    any_viol = False
    for d in DIMS:
        sn = v_salient_null[d]
        nn = v_nonsalient_nonnull[d]
        if sn > 0 or nn > 0:
            print(f"  {d}: salient-but-null={sn}  nonsalient-nonnull={nn}")
            any_viol = True
    if not any_viol:
        print("  None (schema is clean)")
    print()
    print("── Sentences-per-comment distribution ──")
    print(f"  n_comments:  {n_comments:,}")
    print(f"  min:         {spc_min}")
    print(f"  max:         {spc_max}")
    print(f"  mean:        {spc_mean:.2f}")
    print(f"  median:      {spc_median}")
    print(f"  p95:         {spc_p95}")
    print(f"  p99:         {spc_p99}")
    print()
    print("── Sentence-index integrity ──")
    print(f"  Sum of declared n_sentences_in_comment: {total_declared:,}")
    print(f"  Total JSONL rows (should match):        {total_rows:,}")
    if total_declared == total_rows:
        print("  MATCH: declared totals == actual row count")
    else:
        print(f"  MISMATCH: delta = {total_rows - total_declared:+,}")
    print(f"  Comments with gaps (actual unique idx < declared n): {n_gap}")
    print(f"    (first 10: {gap_comments[:10]})" if gap_comments else "")
    print()
    print(f"Wall time: {elapsed:.1f}s")
    print("=" * 60)

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "input_path": input_path,
        "file_size_bytes": os.path.getsize(input_path),
        "total_rows": total_rows,
        "parse_errors": parse_errors,
        "unique_comment_ids": len(comment_ids),
        "unique_thread_ids": len(thread_ids),
        "salience": {"0": sal0, "1": sal1},
        "dim_dist_salient": {
            d: {"NA": dim_dist[d][-1], "Near": dim_dist[d][0], "Far": dim_dist[d][1]}
            for d in DIMS
        },
        "schema_violations": {
            d: {"salient_null": v_salient_null[d], "nonsalient_nonnull": v_nonsalient_nonnull[d]}
            for d in DIMS
        },
        "sentences_per_comment": {
            "n_comments": n_comments,
            "min": spc_min,
            "max": spc_max,
            "mean": round(spc_mean, 4),
            "median": spc_median,
            "p95": spc_p95,
            "p99": spc_p99,
        },
        "index_integrity": {
            "total_declared": total_declared,
            "total_rows": total_rows,
            "match": total_declared == total_rows,
            "gap_comment_count": n_gap,
            "gap_comment_examples": gap_comments[:20],
        },
        "wall_time_seconds": round(elapsed, 1),
    }

    out_path = args.output
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
