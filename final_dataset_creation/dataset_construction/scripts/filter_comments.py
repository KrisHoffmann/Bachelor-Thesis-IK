"""
Stage 2 — Filter comments_raw.jsonl.

Input:  outputs/comments_raw.jsonl (Stage 1 output)
        Fields: comment_id, thread_id, parent_id, author, body, created_utc,
                score, level, y, is_bot, is_award_gesture

Output: outputs/comments_filtered.jsonl

Filters applied in order (each step records count to CONSORT):
  1. bots_and_automod   — rows where is_bot == True (set by Stage 1)
  2. self_deltas        — y==1 AND the commenter also wrote the award gesture
                          under their own comment (i.e., awarded delta to themselves)
  3. award_gestures     — rows where is_award_gesture == True (the "∆" replies from
                          persuaded users; they are not persuasive arguments and must
                          not appear as cases or controls)
  4. deleted_or_removed — body in {None, "", "[deleted]", "[removed]"}
  5. short_comments     — fewer than 20 whitespace-separated tokens
  6. quote_only         — >80% of non-empty lines start with ">"

Self-delta detection (updated for corrected labeller):
  With the corrected labeller, y=1 is the persuasive argument R. A self-delta is
  the case where R's author also wrote the award gesture A under R (i.e., the person
  awarded a delta to their own argument). Detection:

  Pass 1: build {comment_id -> (author, is_award_gesture)} from post-bot records.
          Also build {parent_id -> [child_comment_ids]} (using bare parent_id after
          stripping t1_/t3_ prefix).
  Pass 2: for each y==1 row R, look up R's children. If any child is_award_gesture
          AND has the same author as R, this is a self-delta.

  DeltaBot rejects self-awarding in practice, so self-deltas should be rare. They
  are excluded defensively to keep the corpus clean.

Bots are filtered FIRST so the self-delta lookup does not consider bot-authored rows.
Award gestures are filtered AFTER self-deltas because the self-delta check needs the
is_award_gesture flag of children — those rows must still be present during Pass 2.

CONSORT keys written:
  input_to_filter, after_bots, removed_bots,
  after_self_deltas, removed_self_deltas,
  after_award_gestures, removed_award_gestures,
  after_deleted_removed, removed_deleted_or_removed,
  after_short, removed_short,
  after_quote_only, removed_quote_only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import ConsortCounter, get_logger, stream_jsonl, write_jsonl_stream

DELETED_BODIES = {"[deleted]", "[removed]", "", None}
QUOTE_THRESHOLD = 0.80
SHORT_TOKEN_THRESHOLD = 20


def _strip_prefix(parent_id: str | None) -> str | None:
    """Strip t1_/t3_ prefix from parent_id to get bare comment_id."""
    if not parent_id:
        return None
    for prefix in ("t1_", "t3_", "t2_"):
        if parent_id.startswith(prefix):
            return parent_id[len(prefix):]
    return parent_id


def is_deleted(record: dict) -> bool:
    return record.get("body") in DELETED_BODIES


def is_short(record: dict) -> bool:
    body = record.get("body") or ""
    return len(body.split()) < SHORT_TOKEN_THRESHOLD


def is_quote_only(record: dict) -> bool:
    body = record.get("body") or ""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if not lines:
        return True
    quoted = sum(
        1 for ln in lines
        if ln.lstrip().startswith(">") or ln.lstrip().startswith("&gt;")
    )
    return (quoted / len(lines)) > QUOTE_THRESHOLD

def main():
    parser = argparse.ArgumentParser(description="Stage 2: filter comments")
    parser.add_argument("--input", default="outputs/comments_raw.jsonl")
    parser.add_argument("--output", default="outputs/comments_filtered.jsonl")
    parser.add_argument("--counts-json", default="consort/stage2_counts.json")
    parser.add_argument("--smoke", action="store_true", help="Process only first 1000 records")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = get_logger("filter_comments", args.log_level)
    consort = ConsortCounter()

    # --- Load all records for two-pass self-delta detection ---
    all_records = []
    for rec in stream_jsonl(args.input):
        if args.smoke and len(all_records) >= 1000:
            break
        all_records.append(rec)
    consort.set("input_to_filter", len(all_records))

    # --- Step 1: remove bots (is_bot flag set by Stage 1) ---
    step1 = [r for r in all_records if not r.get("is_bot", False)]
    removed_bots = len(all_records) - len(step1)
    logger.info("Step 1 (bots): removed %d → %d remaining", removed_bots, len(step1))
    consort.set("removed_bots", removed_bots)
    consort.set("after_bots", len(step1))

    # --- Step 2: self-delta removal ---
    # Build lookup tables from post-bot records (award gestures still present here).
    # {comment_id -> (author, is_award_gesture)}
    id_to_info: dict[str, tuple[str, bool]] = {}
    # {bare_parent_id -> [child_comment_ids]}
    children_of: dict[str, list[str]] = {}
    for r in step1:
        cid = r.get("comment_id")
        author = r.get("author")
        ag = r.get("is_award_gesture", False)
        if cid and author:
            id_to_info[cid] = (author, ag)
        bare_parent = _strip_prefix(r.get("parent_id"))
        if bare_parent and cid:
            children_of.setdefault(bare_parent, []).append(cid)

    def is_self_delta(record: dict) -> bool:
        if record.get("y") != 1:
            return False
        cid = record.get("comment_id")
        r_author = record.get("author")
        if not cid or not r_author:
            return False
        # Check if any child of R is an award gesture authored by R's author
        for child_id in children_of.get(cid, []):
            child_info = id_to_info.get(child_id)
            if child_info and child_info[1] and child_info[0] == r_author:
                return True
        return False

    step2 = [r for r in step1 if not is_self_delta(r)]
    removed_self_delta = len(step1) - len(step2)
    logger.info("Step 2 (self-deltas): removed %d → %d remaining", removed_self_delta, len(step2))
    consort.set("removed_self_deltas", removed_self_delta)
    consort.set("after_self_deltas", len(step2))

    # --- Step 3: award gestures ---
    step3 = [r for r in step2 if not r.get("is_award_gesture", False)]
    removed_ag = len(step2) - len(step3)
    logger.info("Step 3 (award gestures): removed %d → %d remaining", removed_ag, len(step3))
    consort.set("removed_award_gestures", removed_ag)
    consort.set("after_award_gestures", len(step3))

    # --- Step 4: deleted/removed ---
    step4 = [r for r in step3 if not is_deleted(r)]
    removed_deleted = len(step3) - len(step4)
    logger.info("Step 4 (deleted/removed): removed %d → %d remaining", removed_deleted, len(step4))
    consort.set("removed_deleted_or_removed", removed_deleted)
    consort.set("after_deleted_removed", len(step4))

    # --- Step 5: short comments ---
    step5 = [r for r in step4 if not is_short(r)]
    removed_short = len(step4) - len(step5)
    logger.info("Step 5 (short <20 tokens): removed %d → %d remaining", removed_short, len(step5))
    consort.set("removed_short", removed_short)
    consort.set("after_short", len(step5))

    # --- Step 6: quote-only ---
    step6 = [r for r in step5 if not is_quote_only(r)]
    removed_quote = len(step5) - len(step6)
    logger.info("Step 6 (quote-only >80%%): removed %d → %d remaining", removed_quote, len(step6))
    consort.set("removed_quote_only", removed_quote)
    consort.set("after_quote_only", len(step6))

    with write_jsonl_stream(args.output) as writer:
        for rec in step6:
            writer.write(rec)

    consort.write(args.counts_json)
    logger.info("Wrote %d filtered comments to %s", len(step6), args.output)


if __name__ == "__main__":
    main()
