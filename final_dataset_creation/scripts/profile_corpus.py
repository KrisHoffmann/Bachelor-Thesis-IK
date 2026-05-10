"""
Profile the first 100 records of a bz2-compressed JSONL corpus.

Writes final_dataset_creation/outputs/topic_filter/schema_probe.json so that
filter_economics.py can discover field names without re-reading the corpus.
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import pathlib
import sys
from collections import Counter, defaultdict
from datetime import datetime

OUTPUT_DIR = pathlib.Path("final_dataset_creation/outputs/topic_filter")
SCHEMA_PROBE_PATH = OUTPUT_DIR / "schema_probe.json"

BODY_CANDIDATES = ["selftext", "body", "op_text", "text", "content"]
TITLE_CANDIDATES = ["title", "submission_title"]
ID_CANDIDATES = ["id", "thread_id", "name", "link_id"]
COUNT_CANDIDATES = ["num_comments", "n_comments", "comment_count"]

DELETED_MARKERS = {"[deleted]", "[removed]"}


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def resolve_input(raw: str) -> pathlib.Path:
    p = pathlib.Path(raw)
    if p.is_dir():
        candidates = sorted(p.glob("*.bz2"))
        thread_candidates = [c for c in candidates if "thread" in c.name.lower()]
        chosen = thread_candidates[0] if thread_candidates else (candidates[0] if candidates else None)
        if chosen is None:
            sys.exit(f"[{ts()}] ERROR: No *.bz2 files found in directory {p}")
        log(f"Directory given — using: {chosen}")
        return chosen
    if not p.exists():
        sys.exit(f"[{ts()}] ERROR: Input path does not exist: {p}")
    return p


def probe_field(
    records: list[dict],
    candidates: list[str],
) -> tuple[str | None, Counter]:
    """Return (best_field_name, occurrence_counter_across_candidates)."""
    counts: Counter = Counter()
    for rec in records:
        for name in candidates:
            if name in rec:
                counts[name] += 1
    if not counts:
        return None, counts
    return counts.most_common(1)[0][0], counts


def body_quality(records: list[dict], field: str | None) -> dict:
    if field is None:
        return {"field": None, "deleted": 0, "removed": 0, "empty": 0, "null": 0, "ok": 0}
    stats: dict[str, int] = {"field": field, "deleted": 0, "removed": 0, "empty": 0, "null": 0, "ok": 0}
    for rec in records:
        val = rec.get(field)
        if val is None:
            stats["null"] += 1
        elif val == "[deleted]":
            stats["deleted"] += 1
        elif val == "[removed]":
            stats["removed"] += 1
        elif str(val).strip() == "":
            stats["empty"] += 1
        else:
            stats["ok"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile first 100 records of a bz2 JSONL corpus.")
    parser.add_argument("--input", required=True, help="Path to .bz2 file or directory containing one.")
    args = parser.parse_args()

    input_path = resolve_input(args.input)
    log(f"Profiling: {input_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    total_bytes_read = 0

    with bz2.open(input_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            total_bytes_read += len(line.encode("utf-8"))
            if len(records) >= 100:
                break

    log(f"Read {len(records)} records ({total_bytes_read:,} bytes uncompressed).")

    # --- Estimate total thread count ---
    file_size = input_path.stat().st_size
    avg_record_bytes = total_bytes_read / max(len(records), 1)
    # bz2 achieves ~4-8x compression on JSON text; use 5x as heuristic
    estimated_uncompressed = file_size * 5
    estimated_total = int(estimated_uncompressed / avg_record_bytes)
    log(f"Rough estimate: ~{estimated_total:,} total threads (file={file_size:,} bytes compressed).")

    # --- Field occurrence counts ---
    field_counts: Counter = Counter()
    for rec in records:
        for key in rec:
            field_counts[key] += 1

    log("Field names seen across 100 records:")
    for field, count in field_counts.most_common():
        print(f"  {field}: {count}/100")

    # --- Probe specific fields ---
    body_field, body_hits = probe_field(records, BODY_CANDIDATES)
    title_field, title_hits = probe_field(records, TITLE_CANDIDATES)
    id_field, id_hits = probe_field(records, ID_CANDIDATES)
    count_field, count_hits = probe_field(records, COUNT_CANDIDATES)

    log(f"Body field detected:         {body_field!r}  (hits per candidate: {dict(body_hits)})")
    log(f"Title field detected:        {title_field!r}  (hits per candidate: {dict(title_hits)})")
    log(f"ID field detected:           {id_field!r}  (hits per candidate: {dict(id_hits)})")
    log(f"Comment-count field detected:{count_field!r}  (hits per candidate: {dict(count_hits)})")

    bq = body_quality(records, body_field)
    log(f"Body quality ({body_field}): {bq}")

    # --- Example records ---
    log("--- 3 example records (title/body truncated to 200 chars) ---")
    for i, rec in enumerate(records[:3]):
        title_val = rec.get(title_field or "", "")[:200] if title_field else ""
        body_val = rec.get(body_field or "", "")[:200] if body_field else ""
        print(f"  [{i+1}] title={title_val!r}")
        print(f"       body={body_val!r}")

    # --- Write schema probe ---
    probe = {
        "input_path": str(input_path),
        "records_sampled": len(records),
        "estimated_total_threads": estimated_total,
        "body_field": body_field,
        "title_field": title_field,
        "id_field": id_field,
        "comment_count_field": count_field,
        "body_quality": bq,
        "all_fields": dict(field_counts),
    }
    SCHEMA_PROBE_PATH.write_text(json.dumps(probe, indent=2))
    log(f"Schema probe written to: {SCHEMA_PROBE_PATH}")


if __name__ == "__main__":
    main()
