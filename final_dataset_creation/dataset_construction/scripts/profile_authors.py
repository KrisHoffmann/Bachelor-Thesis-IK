"""
DEPRECATED — Stage 4 (extract_author_features) has been removed from the pipeline.

Stage 4 was removed because none of the three regression models (conditional logistic
Model 1, mediation decomposition Model 2, mixed-effects logistic Model 3) use author
features as fixed-effect predictors. Author identity enters Model 3 only through a
random intercept, which requires only the 'author' column already present from Stage 1.

This script is retained for reference in case one-off author analyses are needed
outside the registered methodology. Do NOT include it in pipeline runs.

Original purpose: Profile an author feature .jsonl.bz2 file.
Input:  author_liwc.jsonl.bz2  OR  author_subreddit.jsonl.bz2 (or any author bz2)
Output: schema_profiles/<basename>_schema.json
        Human-readable summary printed to stderr.
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


def _flatten_keys(record: dict, prefix: str = "", depth: int = 0) -> dict:
    """Recursively flatten dict keys up to depth=2 for schema inspection."""
    out = {}
    for k, v in record.items():
        full_key = f"{prefix}.{k}" if prefix else k
        out[full_key] = v
        if depth < 2 and isinstance(v, dict):
            out.update(_flatten_keys(v, full_key, depth + 1))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
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
    parser = argparse.ArgumentParser(description="Profile an author .jsonl.bz2 schema")
    parser.add_argument("--input", required=True, help="Path to author .jsonl.bz2 file")
    parser.add_argument("--output-dir", default="schema_profiles", help="Directory for schema JSON output")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = get_logger("profile_authors", args.log_level)

    if not args.input.endswith(".bz2"):
        logger.error("--input must be a .bz2 file, got: %s", args.input)
        sys.exit(1)

    basename = Path(args.input).stem.replace(".jsonl", "")
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
        if len(target_indices) > 0 and i > max(target_indices):
            break

    profile = _build_profile(samples)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{basename}_schema.json"
    result = {"total_records": total, "sampled_records": len(samples), "fields": profile}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Schema written to %s", out_path)

    print(f"\n=== AUTHOR SCHEMA SUMMARY: {basename} ({total} records, {len(samples)} sampled) ===", file=sys.stderr)
    print(f"{'Field':<60} {'Types':<30} {'Null%':>6}  {'Samples'}", file=sys.stderr)
    print("-" * 130, file=sys.stderr)
    for k, v in profile.items():
        types_str = ", ".join(f"{t}({n})" for t, n in v["types"].items())
        samples_str = str(v["sample_values"])[:50]
        print(f"{k:<60} {types_str:<30} {v['null_pct']:>5.1f}%  {samples_str}", file=sys.stderr)


if __name__ == "__main__":
    main()
