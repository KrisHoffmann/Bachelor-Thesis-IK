"""
Stage 3 — Sentence segmentation with spaCy.

Input:  outputs/comments_filtered.jsonl (Stage 2 output)
Output: outputs/comments_segmented.jsonl — same records plus "sentences" field

What's done:
  - Loads spaCy model (en_core_web_trf if available, else en_core_web_sm).
  - Processes comment bodies with nlp.pipe() in batches.
  - Adds record["sentences"] = [s.text for s in doc.sents].

Routing decision:
  Run on Hábrók GPU if comment count > ~200k; CPU otherwise.
  See README "Stage 3 routing decision".

Flags:
  --device {cpu,cuda}   default: cpu
  --batch-size N        default: 64 (increase on GPU, e.g. 256)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import ConsortCounter, get_logger, load_spacy_model, stream_jsonl, write_jsonl_stream


def main():
    parser = argparse.ArgumentParser(description="Stage 3: sentence segmentation")
    parser.add_argument("--input", default="outputs/comments_filtered.jsonl")
    parser.add_argument("--output", default="outputs/comments_segmented.jsonl")
    parser.add_argument("--counts-json", default="consort/stage3_counts.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true", help="Process only first 1000 records")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = get_logger("segment_sentences", args.log_level)
    consort = ConsortCounter()

    nlp = load_spacy_model(args.device)
    # Disable components not needed for sentence segmentation
    disabled = [c for c in ["ner", "lemmatizer"] if c in nlp.pipe_names]
    if disabled:
        nlp.disable_pipes(*disabled)

    logger.info("Streaming comments from %s", args.input)
    records = []
    for rec in stream_jsonl(args.input):
        if args.smoke and len(records) >= 1000:
            break
        records.append(rec)

    logger.info("Loaded %d comments; running nlp.pipe(batch_size=%d)", len(records), args.batch_size)

    bodies = [rec.get("body") or "" for rec in records]
    processed = 0

    with write_jsonl_stream(args.output) as writer:
        for rec, doc in zip(records, nlp.pipe(bodies, batch_size=args.batch_size)):
            rec["sentences"] = [s.text for s in doc.sents]
            writer.write(rec)
            processed += 1
            if processed % 10000 == 0:
                logger.info("Segmented %d / %d comments", processed, len(records))

    consort.set("comments_segmented", processed)
    consort.write(args.counts_json)
    logger.info("Done. Wrote %d segmented comments to %s", processed, args.output)


if __name__ == "__main__":
    main()
