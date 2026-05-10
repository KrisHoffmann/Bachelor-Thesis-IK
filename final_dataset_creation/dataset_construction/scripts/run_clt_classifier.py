"""
Stage 6 — CLT classifier inference.

BLOCKED — wait for shootout winner.
Hábrók sbatch wrapper at slurm/stage6_classifier.sbatch (also stub).

Input contract:
  outputs/sentences_for_classifier.jsonl
  Schema: {comment_id, sentence_index, sentence_text, thread_id}

Output contract:
  outputs/sentence_predictions.parquet
  Columns: [comment_id, sentence_index, label_T, label_S, label_Soc, label_H]
  Each label ∈ {-1, 0, 1}
    -1 = sentence not salient on this dimension
     0 = salient, low CLT (concrete/action focus)
     1 = salient, high CLT (abstract/principle focus)

Do not implement until classifier shootout is complete.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Stage 6: CLT classifier — STUB")
    parser.add_argument("--input", default="outputs/sentences_for_classifier.jsonl")
    parser.add_argument("--output", default="outputs/sentence_predictions.parquet")
    parser.add_argument("--model-path", required=False, help="Path to trained classifier model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    raise NotImplementedError(
        "Stage 6 is BLOCKED — wait for shootout winner. "
        "Input contract: outputs/sentences_for_classifier.jsonl. "
        "Output contract: outputs/sentence_predictions.parquet "
        "with columns [comment_id, sentence_index, label_T, label_S, label_Soc, label_H] "
        "where each label ∈ {-1, 0, 1} (-1 = not salient on dimension)."
    )


if __name__ == "__main__":
    main()
