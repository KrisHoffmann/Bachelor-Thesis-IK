"""
extract_op_selftexts.py
=======================
Extract OP title + selftext for economics-filtered threads and produce a
sentence-level JSONL ready for stage6_inference.py.

Inputs
------
  /mnt/c/cmv_data/3778298/threads.jsonl.bz2
      Full CMV thread dump. Each line is a JSON object with at minimum:
        id        str  e.g. "t3_3d3hj8"  (matches thread_id in the corpus)
        title     str  OP title text
        selftext  str  OP body text (may be empty or "[removed]")

  final_dataset_creation/outputs/topic_filter/cmv_economics_thread_ids.txt
      One thread_id per line (t3_xxx format). Only these threads are extracted.

Outputs (written to final_dataset_creation/outputs/op_selftexts/)
------
  op_sentences_for_classifier.jsonl
      One sentence per line, schema matching stage6_inference.py input contract:
        sentence_id     str  "<thread_id>:<sentence_index>"
        comment_id      str  thread_id  (OP treated as its own "comment")
        thread_id       str  e.g. "t3_3d3hj8"
        sentence_text   str  individual sentence from title + selftext
        sentence_index  int  0-based within this OP

  op_thread_meta.jsonl
      One line per thread (for diagnostics):
        thread_id, title, selftext_raw, n_sentences, kept (bool)

Run from the repo root:
    python scripts/extract_op_selftexts.py [--smoke]
"""

import argparse
import bz2
import json
import re
import pathlib
import sys

THREADS_BZ2  = pathlib.Path("/mnt/c/cmv_data/3778298/threads.jsonl.bz2")
THREAD_IDS_FILE = pathlib.Path(
    "final_dataset_creation/outputs/topic_filter/cmv_economics_thread_ids.txt"
)
OUTPUT_DIR = pathlib.Path(
    "final_dataset_creation/outputs/op_selftexts"
)

REMOVED_MARKERS = {"[removed]", "[deleted]", ""}


def split_sentences(text: str) -> list[str]:
    """
    Minimal sentence splitter that avoids a heavy NLP dependency.
    Splits on '.', '!', '?' followed by whitespace or end-of-string,
    then strips and drops empty fragments.
    This matches the granularity used in the comment segmentation pipeline
    (Stage 3 uses a similar rule-based approach for short texts).
    """
    # Normalise whitespace / Reddit markdown artifacts
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    # Split at sentence-ending punctuation
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def combine_op_text(title: str, selftext: str) -> str:
    """Concatenate title and selftext into a single passage."""
    title = (title or "").strip()
    body  = (selftext or "").strip()
    if body in REMOVED_MARKERS:
        body = ""
    if title and body:
        return title + ". " + body
    return title or body


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OP selftexts for CLT inference")
    parser.add_argument("--smoke", action="store_true",
                        help="Process only first 200 matching threads (for testing)")
    args = parser.parse_args()

    # Load target thread IDs into a set for O(1) lookup
    if not THREAD_IDS_FILE.exists():
        sys.exit(f"ERROR: thread IDs file not found: {THREAD_IDS_FILE}")
    target_ids = set(THREAD_IDS_FILE.read_text().splitlines())
    target_ids = {t.strip() for t in target_ids if t.strip()}
    print(f"Target thread IDs loaded: {len(target_ids):,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_sentences = OUTPUT_DIR / "op_sentences_for_classifier.jsonl"
    out_meta      = OUTPUT_DIR / "op_thread_meta.jsonl"

    n_seen = n_matched = n_sentences = n_skipped_empty = 0
    smoke_limit = 200 if args.smoke else None

    with (
        bz2.open(THREADS_BZ2, "rt", encoding="utf-8") as fin,
        open(out_sentences, "w", encoding="utf-8") as fout_s,
        open(out_meta,      "w", encoding="utf-8") as fout_m,
    ):
        for raw_line in fin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            n_seen += 1
            if n_seen % 10000 == 0:
                print(f"  scanned {n_seen:,} threads, matched {n_matched:,} so far …",
                      flush=True)

            try:
                rec = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            thread_id = rec.get("id", "").strip()
            if thread_id not in target_ids:
                continue

            n_matched += 1
            title    = rec.get("title", "") or ""
            selftext = rec.get("selftext", "") or ""
            text     = combine_op_text(title, selftext)

            sentences = split_sentences(text)
            kept = bool(sentences)

            if not kept:
                n_skipped_empty += 1

            # Write meta record
            fout_m.write(json.dumps({
                "thread_id":    thread_id,
                "title":        title,
                "selftext_raw": selftext[:500],   # truncated for readability
                "n_sentences":  len(sentences),
                "kept":         kept,
            }, ensure_ascii=False) + "\n")

            # Write one sentence row per sentence
            for idx, sent in enumerate(sentences):
                fout_s.write(json.dumps({
                    "sentence_id":    f"{thread_id}:{idx}",
                    "comment_id":     thread_id,   # OP treated as its own comment
                    "thread_id":      thread_id,
                    "sentence_text":  sent,
                    "sentence_index": idx,
                }, ensure_ascii=False) + "\n")
                n_sentences += 1

            if smoke_limit and n_matched >= smoke_limit:
                print(f"Smoke mode: stopping after {smoke_limit} matched threads.")
                break

    print()
    print(f"Threads scanned  : {n_seen:,}")
    print(f"Threads matched  : {n_matched:,}  (target: {len(target_ids):,})")
    print(f"Threads skipped  : {n_skipped_empty:,}  (empty/removed selftext, no title)")
    unmatched = len(target_ids) - n_matched
    if unmatched > 0:
        print(f"WARNING: {unmatched:,} target thread IDs not found in threads.jsonl.bz2")
    print(f"Sentences written: {n_sentences:,}")
    print(f"Output sentences : {out_sentences}")
    print(f"Output meta      : {out_meta}")
    print()
    print("Next step — run CLT inference on Habrok:")
    print("  sbatch slurm/stage6_op_selftext.sbatch")


if __name__ == "__main__":
    main()
