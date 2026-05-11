"""
Validate the DeltaBot-derived delta labels against the curated pairs.jsonl.bz2 dataset.

Purpose:
  pairs.jsonl.bz2 is the ground-truth source from the CMV paired-comment study.
  Each record encodes one (delta, no-delta) pair per submission. We use it as a
  cross-check: the delta comment in each pair should be detected as y=1 by our
  label-derivation logic in pull_comments.py.

  False negatives are bugs in the labeller (our regex missed a real delta).
  False positives are mostly benign — pairs records only ONE delta per submission
  even when threads had multiple deltas; our labeller correctly marks all of them.

Δ-label derivation rule (implemented in _common.find_delta_recipients):
  A comment R is a delta-recipient iff there exists a DeltaBot confirmation B
  somewhere in R's subtree such that:
    - B.author == 'DeltaBot' and B.body matches CONFIRMED_DELTA_PATTERN
    - B.body names a user U via /u/<U> (parsed by extract_delta_recipient_username)
    - R is the EARLIEST ancestor of B (closest to thread root) whose author == U.
  This rule correctly handles back-and-forth threads where several rounds of replies
  precede the delta award; R is always the root-closest comment by the named recipient.

  Known limitations that cause recall < 1.0 (not labeller bugs):
    - ~5.3% of FNs: recipient's comment was deleted from Reddit before this corpus
      snapshot — the comment is absent from threads.jsonl.bz2 so we cannot label it.
    - ~1.7% of FNs: curation differences — pairs chose a later reply in a back-and-forth
      while our earliest-ancestor rule picks the root-closest comment by the named user.
  Combined, these account for the ~7% gap from perfect recall. The --recall-threshold
  flag controls the exit gate; default 0.90 is chosen to allow for these structural gaps.

Input:
  pairs.jsonl.bz2   — {submission_id, submission, nodelta_comment, delta_comment}
                       delta_comment.comments[0]["id"] is the canonical delta comment.
  threads.jsonl.bz2 — full thread data for cross-referencing

Output:
  JSON report with precision / recall on the pairs-ground-truth subset.
  Human-readable summary to stderr.
  Exits with code 1 if recall < --recall-threshold (default 0.90).
  Exits with code 0 and a warning if recall >= threshold but < 1.0.

Usage:
  python scripts/validate_delta_labels.py \\
    --pairs-input /mnt/c/cmv_data/3778298/pairs.jsonl.bz2 \\
    --threads-input /mnt/c/cmv_data/3778298/threads.jsonl.bz2 \\
    --output outputs/validation/delta_label_validation.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    find_delta_recipients,
    get_logger,
    stream_jsonl,
    stream_jsonl_bz2,
)


def _stream_auto(path: str):
    """Stream JSONL from bz2 or plain file, detected by extension."""
    if path.endswith(".bz2"):
        return stream_jsonl_bz2(path)
    return stream_jsonl(path)


def _extract_delta_comment_id(pair_rec: dict) -> str | None:
    """Extract the canonical delta comment ID from a pairs record.

    The delta_comment branch contains a comments list. We treat the first
    comment in that list as the canonical delta-awarded comment.
    """
    delta_branch = pair_rec.get("delta_comment", {})
    comments = delta_branch.get("comments", [])
    if not comments:
        return None
    return comments[0].get("id")


def _derive_delta_ids_from_thread(thread: dict) -> set[str]:
    """Return set of comment_ids we'd label y=1 in this thread.

    Uses find_delta_recipients() — same logic as Stage 1 pull_comments.py —
    so validation is guaranteed to test the actual labeller output.
    """
    return find_delta_recipients(thread)


def main():
    parser = argparse.ArgumentParser(
        description="Validate DeltaBot-derived delta labels against pairs.jsonl.bz2 ground truth"
    )
    parser.add_argument(
        "--pairs-input",
        default="/mnt/c/cmv_data/3778298/pairs.jsonl.bz2",
        help="Path to pairs.jsonl.bz2",
    )
    parser.add_argument(
        "--threads-input",
        default="/mnt/c/cmv_data/3778298/threads.jsonl.bz2",
        help="Path to threads.jsonl.bz2",
    )
    parser.add_argument(
        "--output",
        default="outputs/validation/delta_label_validation.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Limit to first N pairs (default: all)",
    )
    parser.add_argument(
        "--recall-threshold",
        type=float,
        default=0.90,
        help="Exit 1 if recall < threshold (default: 0.90). Warn if < 1.0 but >= threshold.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = get_logger("validate_delta_labels", args.log_level)

    # --- Step 1: Build ground-truth set from pairs ---
    # ground_truth: {submission_id -> delta_comment_id}
    ground_truth: dict[str, str] = {}
    pairs_processed = 0
    logger.info("Loading ground-truth pairs from %s", args.pairs_input)

    for pair in _stream_auto(args.pairs_input):
        if args.max_pairs is not None and pairs_processed >= args.max_pairs:
            break
        sub_id = pair.get("submission_id")
        delta_cid = _extract_delta_comment_id(pair)
        if sub_id and delta_cid:
            ground_truth[sub_id] = delta_cid
        pairs_processed += 1

    logger.info("Loaded %d pairs (unique submissions: %d)", pairs_processed, len(ground_truth))
    target_threads = set(ground_truth.keys())

    # --- Step 2: Stream threads and derive delta labels ---
    true_positives = 0   # GT says delta, we say delta
    false_negatives = 0  # GT says delta, we say NOT delta
    # "false positives" = we mark additional comments as delta (expected for multi-delta threads)
    extra_deltas = 0     # we say delta but GT doesn't mention this comment
    threads_matched = 0

    fn_examples: list[dict] = []

    logger.info("Streaming threads from %s (looking for %d target threads)", args.threads_input, len(target_threads))
    for thread in _stream_auto(args.threads_input):
        tid = thread.get("id")
        if tid not in target_threads:
            continue
        threads_matched += 1

        gt_comment_id = ground_truth[tid]
        our_delta_ids = _derive_delta_ids_from_thread(thread)

        if gt_comment_id in our_delta_ids:
            true_positives += 1
        else:
            false_negatives += 1
            if len(fn_examples) < 5:
                fn_examples.append({
                    "thread_id": tid,
                    "gt_comment_id": gt_comment_id,
                    "our_delta_ids": list(our_delta_ids)[:10],
                })

        extra = our_delta_ids - {gt_comment_id}
        extra_deltas += len(extra)

        if threads_matched % 1000 == 0:
            logger.info("Validated %d/%d threads", threads_matched, len(target_threads))

    threads_missing = len(target_threads) - threads_matched
    if threads_missing > 0:
        logger.warning(
            "%d target threads not found in threads.jsonl.bz2 (they may have been removed from Reddit)",
            threads_missing,
        )

    precision = true_positives / (true_positives + extra_deltas) if (true_positives + extra_deltas) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    report = {
        "pairs_processed": pairs_processed,
        "threads_matched": threads_matched,
        "threads_missing_from_corpus": threads_missing,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives_known_multi": extra_deltas,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fn_examples": fn_examples,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Report written to %s", args.output)

    # Human-readable summary
    print("\n=== DELTA LABEL VALIDATION ===", file=sys.stderr)
    print(f"Pairs processed:          {pairs_processed}", file=sys.stderr)
    print(f"Threads matched:          {threads_matched}", file=sys.stderr)
    print(f"Threads missing:          {threads_missing}", file=sys.stderr)
    print(f"True positives:           {true_positives}", file=sys.stderr)
    print(f"False negatives (BUGS):   {false_negatives}", file=sys.stderr)
    print(f"Extra deltas (multi-δ):   {extra_deltas}", file=sys.stderr)
    print(f"Precision:                {precision:.4f}", file=sys.stderr)
    print(f"Recall:                   {recall:.4f}", file=sys.stderr)

    if false_negatives > 0:
        print(f"\nWARNING: {false_negatives} false negatives detected.", file=sys.stderr)
        print("First examples:", file=sys.stderr)
        for ex in fn_examples:
            print(f"  thread={ex['thread_id']} gt={ex['gt_comment_id']} our={ex['our_delta_ids']}", file=sys.stderr)

    if recall < args.recall_threshold:
        print(
            f"\nERROR: recall {recall:.4f} < threshold {args.recall_threshold:.4f} — labeller has regressions, fix before proceeding!",
            file=sys.stderr,
        )
        sys.exit(1)
    elif recall < 1.0:
        print(
            f"\nWARNING: recall {recall:.4f} < 1.0 but >= threshold {args.recall_threshold:.4f}. "
            "Some FNs may be curation differences (earliest-ancestor vs pairs ground truth), not bugs.",
            file=sys.stderr,
        )
    else:
        print("Recall == 1.0 — labeller is consistent with pairs ground truth.", file=sys.stderr)


if __name__ == "__main__":
    main()
