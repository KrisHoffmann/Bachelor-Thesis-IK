"""
Generate mock_sentence_predictions.parquet consistent with mock_threads.jsonl.

Called by smoke tests before running post-classifier stages. Runs a dry
pipeline of Stage 1 + Stage 2 on the fixture data to determine which comment
IDs survive, then produces realistic sentence-level label predictions.

Uses the same _common.py helpers as the actual pipeline (find_delta_recipients,
is_award_gesture, is_bot_author) so the dry run is always in sync.

Usage:
  python tests/setup_fixtures.py
(from dataset_construction/ directory)

Outputs:
  tests/fixtures/mock_sentence_predictions.parquet

Expected survivors with current fixtures (see create_fixtures.py for design):
  c1  (t3_econA1, y=1, bob)
  c2  (t3_econA1, y=0, carol)
  c3  (t3_econB2, y=1, eve)
  c4  (t3_econB2, y=1, frank)
  c6  (t3_econD4, y=0, iris)
  c7  (t3_econF6, y=1, iris)   ← back-and-forth: earliest ancestor of DeltaBot named by /u/iris
  c7a (t3_econF6, y=0, henry)  ← 20-token reply, survives short filter
  c7b (t3_econF6, y=0, iris)   ← 20-token follow-up, survives short filter; NOT y=1 (c7 is earlier)

All-NaN comment: c6 (t3_econD4, y=0, no stratum impact when dropped)
After Stage 7: 7 comments (c1, c2, c3, c4, c7, c7a, c7b)
Stage 9 strata: 2 (t3_econA1: c1+c2; t3_econF6: c7+c7a or c7+c7b depending on seed)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from _common import (
    find_delta_recipients,
    is_award_gesture,
    is_bot_author,
    stream_jsonl,
    walk_comment_tree,
)

FIXTURES = Path(__file__).parent / "fixtures"
THREADS_FILE = FIXTURES / "mock_threads.jsonl"
THREAD_IDS_FILE = FIXTURES / "mock_econ_thread_ids.txt"

DELETED_BODIES = {"[deleted]", "[removed]", "", None}
QUOTE_THRESHOLD = 0.80
SHORT_TOKEN_THRESHOLD = 20


def _strip_prefix(pid):
    if not pid:
        return None
    for p in ("t1_", "t3_", "t2_"):
        if pid.startswith(p):
            return pid[len(p):]
    return pid


def _is_quote_only(body):
    lines = [ln for ln in (body or "").splitlines() if ln.strip()]
    if not lines:
        return True
    return sum(1 for ln in lines if ln.lstrip().startswith(">")) / len(lines) > QUOTE_THRESHOLD


def run_stage1(thread_ids):
    """Flatten threads to flat comment records using the corrected labeller."""
    records = []
    for thread in stream_jsonl(str(THREADS_FILE)):
        if thread["id"] not in thread_ids:
            continue
        recipients = find_delta_recipients(thread)
        for top in thread.get("comments", []):
            for comment, _ in walk_comment_tree(top):
                cid = comment["id"]
                records.append({
                    "comment_id": cid,
                    "thread_id": thread["id"],
                    "parent_id": comment.get("parent_id"),
                    "author": comment.get("author"),
                    "body": comment.get("body"),
                    "y": 1 if cid in recipients else 0,
                    "is_bot": is_bot_author(comment.get("author")),
                    "is_award_gesture": is_award_gesture(comment),
                })
    return records


def run_stage2(records):
    """Apply Stage 2 filters (mirrors filter_comments.py logic), return surviving records."""
    # Step 1: bots
    step1 = [r for r in records if not r.get("is_bot", False)]

    # Step 2: self-deltas — y=1 comment R where author also wrote award gesture child
    id_to_info = {
        r["comment_id"]: (r["author"], r.get("is_award_gesture", False))
        for r in step1 if r["comment_id"] and r["author"]
    }
    children_of: dict[str, list[str]] = {}
    for r in step1:
        bare = _strip_prefix(r.get("parent_id"))
        if bare and r["comment_id"]:
            children_of.setdefault(bare, []).append(r["comment_id"])

    def is_self_delta(r):
        if r.get("y") != 1:
            return False
        for child_id in children_of.get(r["comment_id"], []):
            info = id_to_info.get(child_id)
            if info and info[1] and info[0] == r["author"]:
                return True
        return False

    step2 = [r for r in step1 if not is_self_delta(r)]

    # Step 3: award gestures
    step3 = [r for r in step2 if not r.get("is_award_gesture", False)]

    # Step 4: deleted/removed
    step4 = [r for r in step3 if r.get("body") not in DELETED_BODIES]

    # Step 5: short
    step5 = [r for r in step4 if len((r.get("body") or "").split()) >= SHORT_TOKEN_THRESHOLD]

    # Step 6: quote-only
    step6 = [r for r in step5 if not _is_quote_only(r.get("body") or "")]
    return step6


def build_predictions(surviving_records):
    """Build sentence-level mock predictions covering all surviving comments."""
    import pandas as pd

    # Label patterns to exercise different aggregation paths
    label_patterns = [
        (1, 0, -1, -1),    # T+S salient
        (-1, -1, -1, -1),  # all -1 (zero-salience sentence)
        (1, 1, 1, 1),      # all salient
        (0, 1, -1, 0),     # mixed
        (-1, 0, 1, -1),    # S+Soc salient
    ]

    # c6 is t3_econD4, y=0 — no stratum impact when dropped by Stage 7.
    # t3_econA1 strata survive: c1(delta)+c2(control) → 1 stratum.
    # t3_econB2 has c3+c4 (both delta, no controls) → excluded from matching.
    all_nan_id = "c6"
    surviving_ids = {r["comment_id"] for r in surviving_records}
    if all_nan_id not in surviving_ids:
        # Fallback: pick a y=0 comment that is not the sole control in a stratum
        y0_ids = [r["comment_id"] for r in surviving_records if r["y"] == 0]
        all_nan_id = y0_ids[-1] if y0_ids else (surviving_records[0]["comment_id"] if surviving_records else None)
        print(f"WARNING: c6 not in survivors, fell back to all_nan_id={all_nan_id}")

    rows = []
    for i, rec in enumerate(surviving_records):
        cid = rec["comment_id"]
        n_sentences = 2 + i % 3
        for sent_idx in range(n_sentences):
            if cid == all_nan_id:
                lt, ls, lsoc, lh = -1, -1, -1, -1
            else:
                lt, ls, lsoc, lh = label_patterns[(i + sent_idx) % len(label_patterns)]
            rows.append({
                "comment_id": cid,
                "sentence_index": sent_idx,
                "label_T": lt,
                "label_S": ls,
                "label_Soc": lsoc,
                "label_H": lh,
            })

    df = pd.DataFrame(rows)
    out_path = FIXTURES / "mock_sentence_predictions.parquet"
    df.to_parquet(out_path, index=False)

    surviving_count = len(surviving_records)
    n_all_nan = 1 if all_nan_id else 0
    print(f"Written {len(df)} sentence rows across {df['comment_id'].nunique()} comments to {out_path}")
    print(f"All-NaN comment: {all_nan_id} (will be dropped by Stage 7)")
    print(f"Expected Stage 7 output: {surviving_count - n_all_nan} comments")
    return all_nan_id, surviving_count


def main():
    thread_ids = set()
    with open(THREAD_IDS_FILE) as f:
        for line in f:
            tid = line.strip()
            if tid:
                thread_ids.add(tid)

    stage1_records = run_stage1(thread_ids)
    print(f"Stage 1: {len(stage1_records)} comments from {len(thread_ids)} thread IDs")
    for r in stage1_records:
        print(
            f"  id={r['comment_id']} author={r['author']} y={r['y']}"
            f" is_bot={r['is_bot']} is_ag={r['is_award_gesture']}"
            f" body_tokens={len((r.get('body') or '').split())}"
        )

    surviving = run_stage2(stage1_records)
    print(f"Stage 2: {len(surviving)} comments survive filtering")
    for r in surviving:
        print(f"  id={r['comment_id']} author={r['author']} y={r['y']}")

    all_nan_id, total = build_predictions(surviving)
    print(f"\nExpected smoke test counts:")
    print(f"  After Stage 1 (with --emit-bots): {len(stage1_records)}")
    print(f"  After Stage 2: {len(surviving)}")
    print(f"  After Stage 7 (drop all-NaN={all_nan_id}): {total - 1}")


if __name__ == "__main__":
    main()
