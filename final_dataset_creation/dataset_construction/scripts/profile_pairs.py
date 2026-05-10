"""
Stage 0a — Profile the pairs.jsonl.bz2 file.

Input:  pairs.jsonl.bz2 (top-level structure with nested nodelta_comment / delta_comment)
Output: schema_profiles/pairs_schema.json
        Human-readable summary printed to stderr.

Two-pass approach:
  Pass 1 — count total records.
  Pass 2 — sample first/middle/last 1000 records and derive per-key types/null_pct/samples.

Run this BEFORE filling in TODO field names in pull_comments.py.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import get_logger, stream_jsonl_bz2


def _infer_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _flatten_keys(record: dict, prefix: str = "") -> dict:
    """Flatten one level of nesting for schema inspection."""
    out = {}
    for k, v in record.items():
        full_key = f"{prefix}.{k}" if prefix else k
        out[full_key] = v
        if isinstance(v, dict):
            for sk, sv in v.items():
                out[f"{full_key}.{sk}"] = sv
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            # Show keys of first list element
            for sk, sv in v[0].items():
                out[f"{full_key}[0].{sk}"] = sv
    return out


def _build_profile(samples: list[dict]) -> dict:
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    null_counts: dict[str, int] = defaultdict(int)
    sample_values: dict[str, list] = defaultdict(list)
    all_keys: set[str] = set()

    for rec in samples:
        flat = _flatten_keys(rec)
        all_keys.update(flat.keys())
        for k, v in flat.items():
            t = _infer_type(v)
            type_counts[k][t] += 1
            if v is None:
                null_counts[k] += 1
            if len(sample_values[k]) < 3 and v is not None:
                sv = v if not isinstance(v, (dict, list)) else str(v)[:120]
                sample_values[k].append(sv)

    n = len(samples)
    profile = {}
    for k in sorted(all_keys):
        null_pct = round(null_counts[k] / n * 100, 1) if n else 0.0
        profile[k] = {
            "types": dict(type_counts[k]),
            "null_pct": null_pct,
            "sample_values": sample_values[k],
        }
    return profile


def main():
    parser = argparse.ArgumentParser(description="Profile pairs.jsonl.bz2 schema")
    parser.add_argument("--input", required=True, help="Path to pairs.jsonl.bz2")
    parser.add_argument("--output-dir", default="schema_profiles", help="Directory for schema JSON output")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = get_logger("profile_pairs", args.log_level)

    if not args.input.endswith(".bz2"):
        logger.error("--input must be a .bz2 file, got: %s", args.input)
        sys.exit(1)

    logger.info("Pass 1: counting records in %s", args.input)
    total = sum(1 for _ in stream_jsonl_bz2(args.input))
    logger.info("Total records: %d", total)

    logger.info("Pass 2: sampling first/middle/last 1000 records")
    mid_start = max(0, total // 2 - 500)
    last_start = max(0, total - 1000)
    target_indices = (
        set(range(0, min(1000, total)))
        | set(range(mid_start, min(mid_start + 1000, total)))
        | set(range(last_start, total))
    )

    samples = []
    for i, rec in enumerate(stream_jsonl_bz2(args.input)):
        if i in target_indices:
            samples.append(rec)
        if i > max(target_indices):
            break

    profile = _build_profile(samples)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pairs_schema.json"
    result = {"total_records": total, "sampled_records": len(samples), "fields": profile}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Schema written to %s", out_path)

    # Human-readable summary to stderr
    print(f"\n=== PAIRS SCHEMA SUMMARY ({total} total records, {len(samples)} sampled) ===", file=sys.stderr)
    print(f"{'Field':<50} {'Types':<30} {'Null%':>6}  {'Samples'}", file=sys.stderr)
    print("-" * 120, file=sys.stderr)
    for k, v in profile.items():
        types_str = ", ".join(f"{t}({n})" for t, n in v["types"].items())
        samples_str = str(v["sample_values"])[:60]
        print(f"{k:<50} {types_str:<30} {v['null_pct']:>5.1f}%  {samples_str}", file=sys.stderr)


if __name__ == "__main__":
    main()
