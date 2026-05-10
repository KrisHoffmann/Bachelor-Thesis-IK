"""
Stage 5 — Compute linguistic control variables (two-pass).

Input:  outputs/comments_segmented.jsonl (Stage 3 output)
Output:
  Pass 1 → outputs/controls_raw.parquet
  Pass 2 → outputs/controls_final.parquet
           consort/jb_diagnostics.json

=== PASS 1: Compute raw control values per comment ===
Continuous controls:
  noun_to_verb_ratio     — spaCy POS: count(NOUN) / count(VERB); NaN if no verbs
  type_token_ratio       — unique lowercased tokens / total tokens
  flesch_kincaid         — textstat.flesch_kincaid_grade(body)
  hedge_density          — hedge count / total tokens; hedge list from data/hedges_hyland_2005.txt
  mean_sentence_length   — mean tokens per sentence (from "sentences" field)
  mean_word_length       — mean chars per token (excluding punctuation)
  punctuation_density    — punct chars / total chars
  paragraph_count        — len(re.split(r'\\n\\s*\\n', body))

Bookkeeping (NOT log-transformed in Pass 2):
  num_sentences          — count of sentences (from "sentences" field)
  num_tokens             — total whitespace tokens

=== PASS 2: Normality check and conditional log1p transform ===
Loads controls_raw.parquet, runs scipy.stats.jarque_bera per continuous control column.
Log-transforms (np.log1p) any column with JB p-value < 0.05.
Writes controls_final.parquet and jb_diagnostics.json.

Caveat: JB-and-transform is applied here for convenience. Strictly, the binding decision
should be made on the matched analysis sample post-Stage-9. The regression script may
override these transformations.

spaCy model loading is shared with segment_sentences.py via _common.load_spacy_model().
"""

import argparse
import json
import re
import sys
from pathlib import Path

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
BOOKKEEPING = ["num_sentences", "num_tokens"]
PUNCT_CHARS = set(".,;:!?\"'()-[]{}…—–/\\")
JB_ALPHA = 0.05


def load_hedges(hedge_file: str) -> set:
    hedges = set()
    with open(hedge_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if line and not line.startswith("#"):
                hedges.add(line)
    return hedges


def compute_raw_controls(rec: dict, nlp, hedges: set) -> dict:
    body = rec.get("body") or ""
    sentences = rec.get("sentences") or []

    doc = nlp(body)
    tokens = [t for t in doc if not t.is_space]
    num_tokens = len(tokens)
    word_tokens = [t for t in tokens if not t.is_punct]

    # noun_to_verb_ratio
    nouns = sum(1 for t in tokens if t.pos_ == "NOUN")
    verbs = sum(1 for t in tokens if t.pos_ == "VERB")
    noun_to_verb_ratio = (nouns / verbs) if verbs > 0 else float("nan")

    # type_token_ratio
    if num_tokens > 0:
        unique = len({t.text.lower() for t in tokens})
        type_token_ratio = unique / num_tokens
    else:
        type_token_ratio = float("nan")

    # flesch_kincaid
    import textstat
    flesch_kincaid = textstat.flesch_kincaid_grade(body) if body.strip() else float("nan")

    # hedge_density
    body_lower = body.lower()
    body_words = body_lower.split()
    hedge_count = 0
    for hedge in hedges:
        if " " in hedge:
            hedge_count += body_lower.count(hedge)
        else:
            hedge_count += body_words.count(hedge)
    hedge_density = hedge_count / num_tokens if num_tokens > 0 else float("nan")

    # mean_sentence_length
    if sentences:
        sent_lengths = [len(s.split()) for s in sentences]
        mean_sentence_length = sum(sent_lengths) / len(sent_lengths)
    else:
        mean_sentence_length = float("nan")

    # mean_word_length
    if word_tokens:
        mean_word_length = sum(len(t.text) for t in word_tokens) / len(word_tokens)
    else:
        mean_word_length = float("nan")

    # punctuation_density
    total_chars = len(body)
    punct_count = sum(1 for c in body if c in PUNCT_CHARS)
    punctuation_density = punct_count / total_chars if total_chars > 0 else float("nan")

    # paragraph_count
    paragraphs = re.split(r"\n\s*\n", body)
    paragraph_count = len([p for p in paragraphs if p.strip()])

    # num_sentences, num_tokens
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
    }


def pass1(args, logger):
    import pandas as pd

    nlp = load_spacy_model(args.device)
    disabled = [c for c in ["ner", "lemmatizer"] if c in nlp.pipe_names]
    if disabled:
        nlp.disable_pipes(*disabled)

    hedge_file = Path(args.hedge_file)
    if not hedge_file.exists():
        logger.error("Hedge file not found: %s", hedge_file)
        sys.exit(1)
    hedges = load_hedges(str(hedge_file))
    logger.info("Loaded %d hedges from %s", len(hedges), hedge_file)

    logger.info("Pass 1: computing raw controls from %s", args.input)
    rows = []
    for i, rec in enumerate(stream_jsonl(args.input)):
        if args.smoke and i >= 1000:
            break
        row = compute_raw_controls(rec, nlp, hedges)
        rows.append(row)
        if (i + 1) % 10000 == 0:
            logger.info("Processed %d comments", i + 1)

    df = pd.DataFrame(rows)
    Path(args.raw_output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.raw_output, index=False)
    logger.info("Pass 1 complete. Wrote %d rows to %s", len(df), args.raw_output)
    return len(df)


def pass2(args, logger):
    import numpy as np
    import pandas as pd
    from scipy.stats import jarque_bera

    logger.info("Pass 2: normality checks and conditional log1p transforms from %s", args.raw_output)
    df = pd.read_parquet(args.raw_output)

    jb_results = {}
    for col in CONTINUOUS_CONTROLS:
        if col not in df.columns:
            logger.warning("Column %s not found in raw parquet, skipping", col)
            continue
        series = df[col].dropna()
        if len(series) < 8:
            jb_results[col] = {"jb_statistic": None, "jb_pvalue": None, "transformed": False, "transform": None}
            continue
        stat, pval = jarque_bera(series)
        do_transform = pval < JB_ALPHA
        jb_results[col] = {
            "jb_statistic": float(stat),
            "jb_pvalue": float(pval),
            "transformed": bool(do_transform),
            "transform": "log1p" if do_transform else None,
        }
        if do_transform:
            df[col] = np.log1p(df[col].clip(lower=0))
            logger.info("log1p-transformed %s (JB p=%.4f)", col, pval)
        else:
            logger.info("No transform for %s (JB p=%.4f)", col, pval)

    Path(args.final_output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.final_output, index=False)
    logger.info("Wrote controls_final.parquet to %s", args.final_output)

    jb_path = Path(args.counts_json).parent / "jb_diagnostics.json"
    jb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jb_path, "w", encoding="utf-8") as f:
        json.dump(jb_results, f, indent=2)
    logger.info("JB diagnostics written to %s", jb_path)


def main():
    parser = argparse.ArgumentParser(description="Stage 5: compute linguistic controls (two-pass)")
    parser.add_argument("--input", default="outputs/comments_segmented.jsonl")
    parser.add_argument("--raw-output", default="outputs/controls_raw.parquet")
    parser.add_argument("--final-output", default="outputs/controls_final.parquet")
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

    pass2(args, logger)

    consort.write(args.counts_json)
    logger.info("Stage 5 complete.")


if __name__ == "__main__":
    main()
