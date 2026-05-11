"""
Stage 5 — Compute linguistic control variables.

Input:  outputs/comments_segmented.jsonl (Stage 3 output)
Output: outputs/comments_controls.parquet  (default, override with --output)
        consort/stage5_counts.json

Continuous controls computed per comment:
  noun_to_verb_ratio     — spaCy POS: count(NOUN) / count(VERB); NaN if no verbs
  type_token_ratio       — unique lowercased word tokens / total word tokens
  flesch_kincaid         — textstat.flesch_kincaid_grade(body)
  hedge_density          — hedge count / total word tokens; hedge list from
                           data/hedges_hyland_2005.txt
  mean_sentence_length   — mean word tokens per sentence (via doc.sents)
  mean_word_length       — mean chars per word token (excluding punctuation)
  punctuation_density    — punct chars / total chars
  paragraph_count        — len(re.split(r'\\n\\s*\\n', cleaned_body))

Processing order per comment: (1) html.unescape (iterated until stable, max 3
passes — the upstream JSONL contains a mixture of single-, double-, and
triple-encoded entities from Reddit's API), (2) markdown link anchor
extraction, (3) URL stripping, (4) spaCy tokenization. Reddit blockquote
markers (>) are retained as part of the body text, since blockquoted citations
function as referential anchors in argumentative discourse and the surrounding
rebuttal prose is genuine commenter content.

Bookkeeping (not control variables):
  num_sentences          — sentence count from Stage 3 "sentences" field
  num_tokens             — total non-space spaCy tokens (incl. punctuation)
  num_word_tokens        — non-space, non-punctuation spaCy tokens
  num_urls               — URL count from original body (before cleaning)

Normality diagnostics (Jarque-Bera) and any log1p transforms are intentionally
deferred to the R analysis pipeline (analysis/load_matched_sample.R), where they
can be applied to the matched analysis sample post-Stage-9.  Running JB on the
full ~598k pre-matched population would reject normality on virtually every
variable from statistical power alone.

TODO: Stage 5.5 currently reads JSONL; update it to read the parquet output of
this script once Stage 5.5 is implemented.
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path

import textstat

sys.path.insert(0, str(Path(__file__).parent))
from _common import ConsortCounter, get_logger, load_spacy_model, stream_jsonl

CONTINUOUS_CONTROLS = [
    "noun_to_verb_ratio",
    "type_token_ratio",
    "flesch_kincaid",
    "hedge_density",
    "mean_sentence_length",
    "mean_word_length",
    "punctuation_density",
    "paragraph_count",
]
BOOKKEEPING = ["num_sentences", "num_tokens", "num_word_tokens", "num_urls"]
PUNCT_CHARS = set(".,;:!?\"'()-[]{}…—–/\\")
URL_PATTERN = re.compile(
    r'https?://[^\s\)\]]+|www\.[^\s\)\]]+',
    re.IGNORECASE,
)
MARKDOWN_LINK_PATTERN = re.compile(
    r'\[([^\]]+)\]\((https?://[^\s\)]+|www\.[^\s\)]+)\)',
    re.IGNORECASE,
)


def _unescape(text: str) -> str:
    """Decode HTML entities iteratively until stable (max 3 passes).

    The upstream JSONL contains a mixture of single-, double-, and
    triple-encoded entities from Reddit's API pipeline.  A single
    html.unescape() call is insufficient for double/triple-encoded strings
    such as '&amp;gt;' (needs two passes) or '&amp;amp;gt;' (needs three).
    Capped at 3 to avoid pathological behaviour on malformed input.
    """
    for _ in range(3):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    return text


def load_hedges(hedge_file: str) -> tuple[set, set]:
    """Load hedges from file, split into single-word and multiword sets.

    Hedges are matched lexically without sense disambiguation. Some Hyland
    (2005) hedges are polysemous (e.g., 'about' as approximator vs. preposition;
    'feel' as cognitive hedge vs. physical perception). False-positive matches
    in non-hedging contexts are treated as random measurement error in the
    downstream regression. This is consistent with standard practice in
    corpus-linguistic hedge analysis.
    """
    single_word: set[str] = set()
    multiword: set[str] = set()
    with open(hedge_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if line and not line.startswith("#"):
                if " " in line:
                    multiword.add(line)
                else:
                    single_word.add(line)
    return single_word, multiword


def compute_raw_controls(rec: dict, doc, hedges_tuple: tuple[set, set]) -> dict:
    body = _unescape(rec.get("body") or "")
    sentences = [_unescape(s) for s in (rec.get("sentences") or [])]

    # num_urls counted on decoded body (pre-URL-strip); all controls use cleaned_body.
    num_urls = len(URL_PATTERN.findall(body))
    # Step 1: replace [anchor](url) -> anchor to preserve link text
    cleaned_body = MARKDOWN_LINK_PATTERN.sub(r'\1', body)
    # Step 2: remove any remaining bare URLs
    cleaned_body = URL_PATTERN.sub('', cleaned_body)

    # doc was built from cleaned_body (see pass1)
    tokens = [t for t in doc if not t.is_space]
    num_tokens = len(tokens)
    word_tokens = [t for t in tokens if not t.is_punct]
    num_word_tokens = len(word_tokens)

    # noun_to_verb_ratio
    nouns = sum(1 for t in tokens if t.pos_ == "NOUN")
    verbs = sum(1 for t in tokens if t.pos_ == "VERB")
    noun_to_verb_ratio = (nouns / verbs) if verbs > 0 else float("nan")

    # type_token_ratio — unique lowercased word tokens / total word tokens
    if num_word_tokens > 0:
        unique = len({t.text.lower() for t in word_tokens})
        type_token_ratio = unique / num_word_tokens
    else:
        type_token_ratio = float("nan")

    # flesch_kincaid
    flesch_kincaid = textstat.flesch_kincaid_grade(cleaned_body) if cleaned_body.strip() else float("nan")

    # Lexical match only; no sense disambiguation. See load_hedges() docstring.
    # hedge_density — tokenization-based matching
    single_word_hedges, multiword_hedges = hedges_tuple
    body_lower = cleaned_body.lower()
    # Single-word hedges: count tokens whose lowercased text is in the hedge set
    hedge_count = sum(1 for t in word_tokens if t.text.lower() in single_word_hedges)
    # Multiword hedges: regex with word boundaries to avoid substring false-positives
    for hedge in multiword_hedges:
        hedge_count += len(re.findall(r"\b" + re.escape(hedge) + r"\b", body_lower))
    hedge_density = hedge_count / num_word_tokens if num_word_tokens > 0 else float("nan")

    # mean_sentence_length — word tokens per sentence via doc.sents
    # Note: num_sentences below uses the Stage 3 "sentences" field (canonical
    # segmentation); sent_lengths here uses spaCy's doc.sents for token
    # consistency.  The two may differ slightly.
    sent_lengths = [
        len([t for t in sent if not t.is_space and not t.is_punct])
        for sent in doc.sents
    ]
    mean_sentence_length = (
        sum(sent_lengths) / len(sent_lengths) if sent_lengths else float("nan")
    )

    # mean_word_length
    if word_tokens:
        mean_word_length = sum(len(t.text) for t in word_tokens) / len(word_tokens)
    else:
        mean_word_length = float("nan")

    # punctuation_density
    total_chars = len(cleaned_body)
    punct_count = sum(1 for c in cleaned_body if c in PUNCT_CHARS)
    punctuation_density = punct_count / total_chars if total_chars > 0 else float("nan")

    # paragraph_count
    paragraphs = re.split(r"\n\s*\n", cleaned_body)
    paragraph_count = len([p for p in paragraphs if p.strip()])

    # num_sentences from Stage 3 segmentation
    num_sentences = len(sentences)

    return {
        "comment_id": rec.get("comment_id"),
        "noun_to_verb_ratio": noun_to_verb_ratio,
        "type_token_ratio": type_token_ratio,
        "flesch_kincaid": flesch_kincaid,
        "hedge_density": hedge_density,
        "mean_sentence_length": mean_sentence_length,
        "mean_word_length": mean_word_length,
        "punctuation_density": punctuation_density,
        "paragraph_count": float(paragraph_count),
        "num_sentences": num_sentences,
        "num_tokens": num_tokens,
        "num_word_tokens": num_word_tokens,
        "num_urls": num_urls,
    }


def pass1(args, logger):
    import pandas as pd

    nlp = load_spacy_model(args.device)

    # Disable pipes we don't need; keep senter (or parser) for doc.sents
    pipe_names = set(nlp.pipe_names)
    always_disable = [c for c in ["ner", "lemmatizer"] if c in pipe_names]
    # Disable parser only if senter is available — senter provides sentence
    # boundaries needed for mean_sentence_length via doc.sents
    if "senter" in pipe_names and "parser" in pipe_names:
        always_disable.append("parser")
    if always_disable:
        nlp.select_pipes(disable=always_disable)

    hedge_file = Path(args.hedge_file)
    if not hedge_file.exists():
        logger.error("Hedge file not found: %s", hedge_file)
        sys.exit(1)
    hedges_tuple = load_hedges(str(hedge_file))
    sw, mw = hedges_tuple
    logger.info(
        "Loaded %d single-word and %d multiword hedges from %s",
        len(sw), len(mw), hedge_file,
    )

    logger.info("Pass 1: computing raw controls from %s", args.input)

    # Read all records into memory, then batch-process cleaned bodies through nlp.pipe()
    records = []
    for i, rec in enumerate(stream_jsonl(args.input)):
        if args.smoke and i >= 1000:
            break
        records.append(rec)

    # Pre-compute cleaned_body per record so nlp.pipe() never sees raw HTML
    # entities or URLs.  Processing order: (1) html.unescape, (2) markdown
    # link anchor extraction, (3) URL stripping, (4) spaCy.
    # compute_raw_controls() re-derives cleaned_body from the decoded body
    # using the same order; the two derivations are identical.
    def _clean(body: str) -> str:
        return URL_PATTERN.sub('', MARKDOWN_LINK_PATTERN.sub(r'\1', _unescape(body)))

    bodies = [_clean(rec.get("body") or "") for rec in records]

    rows = []
    for i, (rec, doc) in enumerate(
        zip(records, nlp.pipe(bodies, batch_size=50))
    ):
        row = compute_raw_controls(rec, doc, hedges_tuple)
        rows.append(row)
        if (i + 1) % 10000 == 0:
            logger.info("Processed %d comments", i + 1)

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    logger.info("Pass 1 complete. Wrote %d rows to %s", len(df), args.output)
    return len(df)


def main():
    parser = argparse.ArgumentParser(description="Stage 5: compute linguistic controls")
    parser.add_argument("--input", default="outputs/comments_segmented.jsonl")
    parser.add_argument("--output", default="outputs/comments_controls.parquet")
    parser.add_argument("--hedge-file", default="data/hedges_hyland_2005.txt")
    parser.add_argument("--counts-json", default="consort/stage5_counts.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Process only first 1000 records")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = get_logger("compute_controls", args.log_level)
    consort = ConsortCounter()

    n = pass1(args, logger)
    consort.set("comments_with_controls", n)

    consort.write(args.counts_json)
    logger.info("Stage 5 complete.")


if __name__ == "__main__":
    main()
