"""
Stage 5.5 — Export sentence-level JSONL for CLT classifier (Stage 6).

Input:  outputs/comments_segmented.jsonl (Stage 3 output)
Output: outputs/sentences_for_classifier.jsonl

What's done:
  - Reads each comment record (which has a "sentences" list from Stage 3).
  - Explodes to one row per (comment_id, sentence_index, sentence_text).
  - Preserves thread_id for downstream diagnostics.

Output schema (Stage 6 input contract):
  {
    "comment_id":     str  — unique comment identifier
    "sentence_index": int  — 0-based index within the comment
    "sentence_text":  str  — sentence text
    "thread_id":      str  — thread identifier (e.g. "t3_abc123")
  }

Stage 6 (run_clt_classifier.py) must consume this file and produce:
  outputs/sentence_predictions.parquet
  with columns: [comment_id, sentence_index, label_T, label_S, label_Soc, label_H]
  where each label ∈ {-1, 0, 1}  (-1 = not salient on that dimension)

CONSORT keys written:
  sentences_for_classifier
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import ConsortCounter, get_logger, stream_jsonl, write_jsonl_stream


def main():
    parser = argparse.ArgumentParser(description="Stage 5.5: explode comments to sentence-level JSONL for classifier")
    parser.add_argument("--input", default="outputs/comments_segmented.jsonl")
    parser.add_argument("--output", default="outputs/sentences_for_classifier.jsonl")
    parser.add_argument("--counts-json", default="consort/stage55_counts.json")
    parser.add_argument("--smoke", action="store_true", help="Process only first 1000 comments")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = get_logger("prepare_for_classifier", args.log_level)
    consort = ConsortCounter()

    comments_processed = 0
    total_sentences = 0

    with write_jsonl_stream(args.output) as writer:
        for rec in stream_jsonl(args.input):
            if args.smoke and comments_processed >= 1000:
                break
            comment_id = rec.get("comment_id")
            thread_id = rec.get("thread_id")
            sentences = rec.get("sentences") or []

            for idx, sentence_text in enumerate(sentences):
                writer.write({
                    "comment_id": comment_id,
                    "sentence_index": idx,
                    "sentence_text": sentence_text,
                    "thread_id": thread_id,
                })
                total_sentences += 1

            comments_processed += 1
            if comments_processed % 10000 == 0:
                logger.info("Exploded %d comments → %d sentences", comments_processed, total_sentences)

    consort.set("comments_exploded", comments_processed)
    consort.set("sentences_for_classifier", total_sentences)
    consort.write(args.counts_json)

    logger.info(
        "Done. %d comments → %d sentences written to %s",
        comments_processed, total_sentences, args.output,
    )
    logger.info("READY FOR STAGE 6 (classifier shootout). Output: %s", args.output)


if __name__ == "__main__":
    main()
