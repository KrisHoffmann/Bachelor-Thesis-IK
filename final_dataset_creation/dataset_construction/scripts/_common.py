"""
Shared utilities for Phase B dataset construction pipeline.

Provides: stream_jsonl_bz2, stream_jsonl, write_jsonl, write_jsonl_stream,
ConsortCounter, get_logger, load_spacy_model,
CONFIRMED_DELTA_PATTERN, is_delta_confirmation, is_bot_author,
walk_comment_tree, find_delta_recipients, is_award_gesture.
"""

import bz2
import json
import logging
import re
import sys
from pathlib import Path
from typing import Iterator


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def stream_jsonl_bz2(path: str) -> Iterator[dict]:
    """Stream records from a .jsonl.bz2 file one at a time."""
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stream_jsonl(path: str) -> Iterator[dict]:
    """Stream records from a plain .jsonl file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: list[dict], path: str) -> None:
    """Write a list of dicts to a .jsonl file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_jsonl_stream(path: str):
    """Context manager that returns a write callable for streaming JSONL output."""
    class _Writer:
        def __init__(self, p):
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            self._f = open(p, "w", encoding="utf-8")
            self.count = 0

        def write(self, rec: dict):
            self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.count += 1

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self._f.close()

    return _Writer(path)


class ConsortCounter:
    """Accumulate CONSORT flow counts and write to JSON."""

    def __init__(self):
        self._counts: dict[str, int] = {}

    def set(self, key: str, value: int) -> None:
        self._counts[key] = value

    def increment(self, key: str, by: int = 1) -> None:
        self._counts[key] = self._counts.get(key, 0) + by

    def get(self, key: str, default: int = 0) -> int:
        return self._counts.get(key, default)

    def as_dict(self) -> dict:
        return dict(self._counts)

    def write(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._counts, f, indent=2)


def load_spacy_model(device: str = "cpu"):
    """Load spaCy model, preferring en_core_web_trf; fall back to en_core_web_sm."""
    import spacy
    logger = get_logger("load_spacy_model")

    if device == "cuda":
        spacy.prefer_gpu()
        logger.info("GPU preference set for spaCy")

    try:
        nlp = spacy.load("en_core_web_trf")
        logger.info("Loaded spaCy model: en_core_web_trf")
    except OSError:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model: en_core_web_sm (trf not available)")

    return nlp


# ---------------------------------------------------------------------------
# Delta-label derivation helpers
# ---------------------------------------------------------------------------

# Matches DeltaBot confirmation replies. Body must start with "Confirmed:" after
# stripping, followed by digit count and "delta(s) awarded".
CONFIRMED_DELTA_PATTERN = re.compile(
    r"^Confirmed:\s+\d+\s+delta(s)?\s+awarded", re.IGNORECASE
)

# Captures the username from DeltaBot's confirmation body.
# Handles two DeltaBot formats:
#   new: "Confirmed: N delta(s) awarded to /u/USERNAME …"
#   old: "Confirmed: N delta(s) awarded to USERNAME …" (no /u/ prefix)
# Reddit usernames: 3-20 chars, letters/digits/underscores/hyphens.
# Termination: whitespace, "(", "[", or end-of-string.
DELTA_RECIPIENT_PATTERN = re.compile(
    r"^Confirmed:\s+\d+\s+delta(?:s)?\s+awarded\s+to\s+(?:/u/)?([A-Za-z0-9_\-]+)",
    re.IGNORECASE,
)


def extract_delta_recipient_username(deltabot_body: str | None) -> str | None:
    """Parse the recipient username from a DeltaBot confirmation body.

    Returns None if the body doesn't match or username can't be parsed.
    """
    if not deltabot_body:
        return None
    m = DELTA_RECIPIENT_PATTERN.match(deltabot_body.strip())
    if m:
        return m.group(1)
    return None


def is_delta_confirmation(comment: dict) -> bool:
    """Return True if this comment is a DeltaBot delta-confirmation reply.

    Only matches the affirmative "Confirmed: N delta(s) awarded …" pattern.
    Rejected/revoked/removed replies do NOT match.
    """
    if comment.get("author") != "DeltaBot":
        return False
    body = (comment.get("body") or "").strip()
    return bool(CONFIRMED_DELTA_PATTERN.match(body))


# ---------------------------------------------------------------------------
# Bot-author detection
# ---------------------------------------------------------------------------

# Rule: author == "AutoModerator"  OR
#       re.search(r'bot', re.sub(r'[_0-9]+$', '', author), re.IGNORECASE)
#
# Caveat: the substring "bot" inside other words (e.g. "ABottledCoke",
# "robotron") may produce false positives. This is intentional — it is safer
# to over-exclude than to include bot noise. Document any affected authors in
# the CONSORT report.
_BOT_RE = re.compile(r"bot", re.IGNORECASE)
_TRAILING_RE = re.compile(r"[_0-9]+$")


def is_bot_author(author: str | None) -> bool:
    """Return True if `author` looks like a bot account."""
    if not author:
        return False
    if author == "AutoModerator":
        return True
    stripped = _TRAILING_RE.sub("", author)
    return bool(_BOT_RE.search(stripped))


# ---------------------------------------------------------------------------
# Comment-tree walker
# ---------------------------------------------------------------------------

def walk_comment_tree(comment: dict, depth: int = 0) -> Iterator[tuple[dict, int]]:
    """Yield (comment_dict, depth) for the root comment and all descendants."""
    yield comment, depth
    for child in comment.get("children", []):
        yield from walk_comment_tree(child, depth + 1)


# ---------------------------------------------------------------------------
# Delta-recipient identification (corrected tree traversal)
# ---------------------------------------------------------------------------

def is_award_gesture(comment: dict) -> bool:
    """Return True if this comment is an 'award gesture' — i.e., it has a
    direct DeltaBot confirmation child.

    Award gestures are the persuaded user's replies containing "∆"; they are
    NOT delta-recipients. Stage 2 drops them so they never appear in the
    analysis corpus as cases or controls.
    """
    return any(
        child.get("author") == "DeltaBot" and is_delta_confirmation(child)
        for child in comment.get("children", [])
    )


def find_delta_recipients(thread_record: dict) -> set[str]:
    """Walk the comment forest and return comment_ids of delta-recipients.

    Rule (verified against pairs.jsonl.bz2 ground truth):
      A comment R is a delta-recipient iff there exists a DeltaBot confirmation
      B somewhere in R's subtree such that:
        - B.author == 'DeltaBot'
        - B.body matches CONFIRMED_DELTA_PATTERN
        - B.body names a user U via /u/<U> (parsed by extract_delta_recipient_username)
        - R is the EARLIEST ancestor of B (closest to thread root) whose author == U.

    This rule correctly handles back-and-forth threads where several rounds of
    replies precede the delta award: R is always the first (root-closest) comment
    by the named recipient in B's ancestor chain, regardless of nesting depth.

    If DeltaBot names a user not found in the ancestor chain (data anomaly), a
    warning is logged and that confirmation is skipped.

    Returns set of bare comment ids (matching the 'id' field, without t1_ prefix).
    """
    recipients: set[str] = set()
    logger = get_logger("find_delta_recipients")

    def _walk(c: dict, ancestors: list[dict]) -> None:
        if c.get("author") == "DeltaBot" and is_delta_confirmation(c):
            named_user = extract_delta_recipient_username(c.get("body"))
            if named_user is None:
                logger.warning(
                    "DeltaBot confirmation matched header but failed username parse: "
                    "comment_id=%s body[:80]=%r",
                    c.get("id"), (c.get("body") or "")[:80],
                )
            else:
                # Walk ancestors from root toward DeltaBot; pick the first
                # (earliest = closest to root) whose author matches named_user.
                match = None
                for ancestor in ancestors:
                    if ancestor.get("author") == named_user:
                        match = ancestor
                        break
                if match is None:
                    logger.warning(
                        "DeltaBot named /u/%s but no ancestor matches "
                        "(comment_id=%s, ancestor authors=%s)",
                        named_user, c.get("id"),
                        [a.get("author") for a in ancestors],
                    )
                else:
                    recipients.add(match["id"])

        new_ancestors = ancestors + [c]
        for child in c.get("children", []):
            _walk(child, new_ancestors)

    for top in thread_record.get("comments", []):
        _walk(top, [])
    return recipients
