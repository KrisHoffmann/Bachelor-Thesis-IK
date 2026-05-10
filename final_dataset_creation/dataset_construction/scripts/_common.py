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
    NOT delta-recipients. The recipient is the comment one level above the
    award gesture (the persuasive argument).
    """
    return any(is_delta_confirmation(child) for child in comment.get("children", []))


def find_delta_recipients(thread_record: dict) -> set[str]:
    """Walk the comment forest and return comment_ids of delta-recipients.

    A comment R is a delta-recipient iff there exists a child A of R such
    that A has a child B where is_delta_confirmation(B) is True.

    The CMV award flow is: R (persuasive argument) → A (award gesture with ∆)
    → B (DeltaBot confirmation). The recipient is R, not A.

    Returns set of bare comment ids (matching the 'id' field, without t1_ prefix).
    """
    recipients: set[str] = set()

    def _walk(c: dict) -> None:
        for child_a in c.get("children", []):
            for child_b in child_a.get("children", []):
                if is_delta_confirmation(child_b):
                    recipients.add(c["id"])
                    break
            if c["id"] in recipients:
                break
        for child in c.get("children", []):
            _walk(child)

    for top in thread_record.get("comments", []):
        _walk(top)
    return recipients
