"""
Stage 1 — Flatten threads.jsonl.bz2 into per-comment JSONL, filter to economics threads,
and derive delta labels from the corrected R→A→B tree structure.

Input:
  threads.jsonl.bz2   — one thread (submission) per line with nested comment forest.
    Top-level keys: id, name, selftext, title, author, created_utc, num_comments,
                    delta (submission-level), comments: [...]
  Each comment:        id, parent_id, link_id, level, body, author, created_utc,
                        score, controversiality, author_flair_text, edited, gilded,
                        urls, children: [...]
  thread_ids file     — one t3_xxx ID per line (economics thread allowlist)

Output:
  JSONL with one flat record per comment:
    {comment_id, thread_id, parent_id, author, body, created_utc, score, level,
     y, is_bot, is_award_gesture}

Delta label derivation (CORRECTED):
  The CMV delta-award tree structure is:
    R (delta-recipient — the persuasive argument)
    └─ A (award gesture — the persuaded user's ∆ reply)
       └─ B (DeltaBot confirmation: "Confirmed: N delta(s) awarded to /u/...")

  y=1 is assigned to R (the persuasive argument), NOT to A (the award gesture).
  Implementation: find_delta_recipients() does a per-thread pre-pass collecting
  all recipient comment ids; each comment is labelled y=1 iff its id is in that set.

is_award_gesture flag:
  True for comments A where any direct child is a DeltaBot confirmation. These are
  the OP's "you convinced me, ∆" replies. They should NOT be y=1 (they are not
  persuasive arguments), and filter_comments.py will exclude them from the corpus.

Bot rows:
  Bot rows are emitted by default with is_bot=True so that filter_comments.py can
  count and remove them in a single auditable CONSORT step. Use --no-emit-bots to
  suppress them.

Schema field constants are confirmed against threads.jsonl.bz2. No TODO needed.

CONSORT keys written:
  threads_in_economics_filter, threads_matched_in_corpus, threads_missing_from_corpus,
  raw_comments_total, raw_comments_bot, raw_comments_nonbot, raw_comments_delta_awarded
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    ConsortCounter,
    find_delta_recipients,
    get_logger,
    is_award_gesture,
    is_bot_author,
    stream_jsonl,
    stream_jsonl_bz2,
    walk_comment_tree,
    write_jsonl_stream,
)


def _stream_auto(path: str):
    """Stream JSONL from bz2 or plain file, detected by extension."""
    if path.endswith(".bz2"):
        return stream_jsonl_bz2(path)
    return stream_jsonl(path)

# Confirmed against threads.jsonl.bz2 schema. No TODO needed.
COMMENT_BODY_FIELD = "body"
COMMENT_AUTHOR_FIELD = "author"
COMMENT_ID_FIELD = "id"
COMMENT_PARENT_FIELD = "parent_id"
COMMENT_CHILDREN_FIELD = "children"
THREAD_ID_FIELD = "id"
THREAD_COMMENTS_FIELD = "comments"


def flatten_thread(thread: dict) -> list[dict]:
    """Walk all comments in a thread and return flat records with corrected delta labels."""
    thread_id = thread[THREAD_ID_FIELD]
    # Pre-pass: collect all delta-recipient comment ids for this thread
    recipients = find_delta_recipients(thread)

    flat = []
    for top_comment in thread.get(THREAD_COMMENTS_FIELD, []):
        for comment, _depth in walk_comment_tree(top_comment):
            cid = comment.get(COMMENT_ID_FIELD)
            flat.append({
                "comment_id": cid,
                "thread_id": thread_id,
                "parent_id": comment.get(COMMENT_PARENT_FIELD),
                "author": comment.get(COMMENT_AUTHOR_FIELD),
                "body": comment.get(COMMENT_BODY_FIELD),
                "created_utc": comment.get("created_utc"),
                "score": comment.get("score"),
                "level": comment.get("level"),
                "y": 1 if cid in recipients else 0,
                "is_bot": is_bot_author(comment.get(COMMENT_AUTHOR_FIELD)),
                "is_award_gesture": is_award_gesture(comment),
            })
    return flat


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: flatten threads.jsonl.bz2, filter to economics threads, derive delta labels"
    )
    parser.add_argument(
        "--threads-input",
        default="/mnt/c/cmv_data/3778298/threads.jsonl.bz2",
        help="Path to threads.jsonl.bz2",
    )
    parser.add_argument(
        "--thread-ids-input",
        default="../outputs/topic_filter/cmv_economics_thread_ids.txt",
        help="File with one t3_xxx thread ID per line (economics allowlist)",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--counts-json", required=True, help="CONSORT counts output path")
    parser.add_argument(
        "--emit-bots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit bot rows with is_bot=True (default: True). Use --no-emit-bots to suppress.",
    )
    parser.add_argument("--smoke", action="store_true", help="Process first 100 thread records and exit")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = get_logger("pull_comments", args.log_level)
    consort = ConsortCounter()

    # Load thread ID allowlist
    thread_id_path = Path(args.thread_ids_input)
    if not thread_id_path.exists():
        logger.error("Thread IDs file not found: %s", thread_id_path)
        sys.exit(1)
    thread_id_set: set[str] = set()
    with open(thread_id_path, "r", encoding="utf-8") as f:
        for line in f:
            tid = line.strip()
            if tid:
                thread_id_set.add(tid)
    logger.info("Loaded %d economics thread IDs from %s", len(thread_id_set), thread_id_path)
    consort.set("threads_in_economics_filter", len(thread_id_set))

    threads_matched = 0
    raw_total = 0
    raw_bot = 0
    raw_nonbot = 0
    raw_delta = 0

    with write_jsonl_stream(args.output) as writer:
        for i, thread in enumerate(_stream_auto(args.threads_input)):
            if args.smoke and i >= 100:
                break

            tid = thread.get(THREAD_ID_FIELD)
            if tid not in thread_id_set:
                continue

            threads_matched += 1
            flat_records = flatten_thread(thread)

            for rec in flat_records:
                raw_total += 1
                if rec["is_bot"]:
                    raw_bot += 1
                else:
                    raw_nonbot += 1
                if rec["y"] == 1:
                    raw_delta += 1

                if rec["is_bot"] and not args.emit_bots:
                    continue
                writer.write(rec)

            if threads_matched % 500 == 0:
                logger.info("Matched %d economics threads so far", threads_matched)

    threads_missing = len(thread_id_set) - threads_matched
    consort.set("threads_matched_in_corpus", threads_matched)
    consort.set("threads_missing_from_corpus", threads_missing)
    consort.set("raw_comments_total", raw_total)
    consort.set("raw_comments_bot", raw_bot)
    consort.set("raw_comments_nonbot", raw_nonbot)
    consort.set("raw_comments_delta_awarded", raw_delta)
    consort.set("comments_written", writer.count)
    consort.write(args.counts_json)

    logger.info(
        "Done. Threads matched: %d (missing from corpus: %d) | "
        "comments total: %d | bot: %d | nonbot: %d | delta: %d | written: %d",
        threads_matched, threads_missing,
        raw_total, raw_bot, raw_nonbot, raw_delta, writer.count,
    )


if __name__ == "__main__":
    main()
