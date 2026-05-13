"""
Stage 5.5 — Sentence export for the BERT cascade classifier.

Reads the spaCy-segmented comment file produced by Stage 3
(outputs/comments_segmented.jsonl, 598,272 comments) and explodes each
comment's `sentences` list into one JSONL row per sentence.

Repo-root assumption: run from the repository root, i.e.
    python scripts/stage5_5_sentence_export.py
Paths below are relative to that root.

Input:
    final_dataset_creation/dataset_construction/outputs/comments_segmented.jsonl

Outputs:
    final_dataset_creation/dataset_construction/outputs/sentences_for_classifier.jsonl
    final_dataset_creation/dataset_construction/consort/stage5_5_counts.json
"""

import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DS_ROOT = REPO_ROOT / "final_dataset_creation" / "dataset_construction"

INPUT_PATH  = DS_ROOT / "outputs" / "comments_segmented.jsonl"
OUTPUT_PATH = DS_ROOT / "outputs" / "sentences_for_classifier.jsonl"
COUNTS_PATH = DS_ROOT / "consort" / "stage5_5_counts.json"

LOG_EVERY = 50_000


def main() -> None:
    if not INPUT_PATH.exists():
        sys.exit(f"ERROR: input not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_comments = 0
    n_skipped_empty = 0
    n_sentences = 0
    t0 = time.time()

    with open(INPUT_PATH, encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for raw_line in fin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            comment = json.loads(raw_line)
            n_comments += 1

            sentences = comment.get("sentences")
            if not sentences:
                n_skipped_empty += 1
                if n_comments % LOG_EVERY == 0:
                    log.info("comments read: %d  sentences written: %d", n_comments, n_sentences)
                    sys.stdout.flush()
                continue

            n_in_comment = len(sentences)
            comment_id = comment["comment_id"]
            thread_id  = comment.get("thread_id", "")
            author     = comment.get("author", "")
            y          = comment.get("y", 0)

            for idx, sentence_text in enumerate(sentences):
                row = {
                    "sentence_id":            f"{comment_id}:{idx}",
                    "comment_id":             comment_id,
                    "thread_id":              thread_id,
                    "author":                 author,
                    "y":                      y,
                    "sentence_text":          sentence_text,
                    "sentence_index":         idx,
                    "n_sentences_in_comment": n_in_comment,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_sentences += 1

            if n_comments % LOG_EVERY == 0:
                elapsed = time.time() - t0
                log.info(
                    "comments read: %d  sentences written: %d  elapsed: %.0fs",
                    n_comments, n_sentences, elapsed,
                )
                sys.stdout.flush()

    elapsed = time.time() - t0
    mean_spc = n_sentences / max(n_comments - n_skipped_empty, 1)

    counts = {
        "n_comments":              n_comments,
        "n_comments_skipped_empty": n_skipped_empty,
        "n_sentences":             n_sentences,
        "mean_sentences_per_comment": round(mean_spc, 4),
        "elapsed_seconds":         round(elapsed, 1),
    }

    with open(COUNTS_PATH, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)

    log.info("=== Stage 5.5 complete ===")
    log.info("  comments read      : %d", n_comments)
    log.info("  comments skipped   : %d (empty/missing sentences)", n_skipped_empty)
    log.info("  sentences written  : %d", n_sentences)
    log.info("  mean sent/comment  : %.2f", mean_spc)
    log.info("  elapsed            : %.1f s", elapsed)
    log.info("  output             : %s", OUTPUT_PATH)
    log.info("  counts             : %s", COUNTS_PATH)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
