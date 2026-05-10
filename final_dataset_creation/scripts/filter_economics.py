"""
Zero-shot topic filter: keep CMV threads about economics/economic policy.

Requires schema_probe.json produced by profile_corpus.py.
Model: MoritzLaurer/deberta-v3-large-zeroshot-v2.0
"""

from __future__ import annotations

import argparse
import bz2
import csv
import json
import os
import pathlib
import random
import sys
import time
from datetime import datetime
from typing import Any

OUTPUT_DIR = pathlib.Path("final_dataset_creation/outputs/topic_filter")
SCHEMA_PROBE_PATH = OUTPUT_DIR / "schema_probe.json"

MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
CANDIDATE_LABELS = ["economics", "an unrelated topic"]
POSITIVE_LABEL = "economics"
DELETED_MARKERS = {"[deleted]", "[removed]"}


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

def resolve_input(raw: str) -> pathlib.Path:
    p = pathlib.Path(raw)
    if p.is_dir():
        candidates = sorted(p.glob("*.bz2"))
        thread_candidates = [c for c in candidates if "thread" in c.name.lower()]
        chosen = thread_candidates[0] if thread_candidates else (candidates[0] if candidates else None)
        if chosen is None:
            sys.exit(f"[{ts()}] ERROR: No *.bz2 files found in directory {p}")
        log(f"Directory given — using: {chosen}")
        return chosen
    if not p.exists():
        sys.exit(f"[{ts()}] ERROR: Input path does not exist: {p}")
    return p


# ---------------------------------------------------------------------------
# Pre-filter
# ---------------------------------------------------------------------------

def is_deleted(val: Any) -> bool:
    return val is None or str(val).strip() in DELETED_MARKERS or str(val).strip() == ""


def prefilter(
    records: list[dict],
    body_field: str | None,
    title_field: str | None,
    count_field: str | None,
) -> tuple[list[dict], dict]:
    profile: dict[str, int] = {"raw_threads": len(records)}

    # Drop deleted/removed/empty body
    after_body = [
        r for r in records
        if body_field and not is_deleted(r.get(body_field))
    ]
    profile["after_remove_deleted"] = len(after_body)
    log(f"Pre-filter body check: {len(records)} -> {len(after_body)}")

    # Drop deleted title
    after_title = [
        r for r in after_body
        if not (title_field and str(r.get(title_field, "")).strip() in DELETED_MARKERS)
    ]
    log(f"Pre-filter title check: {len(after_body)} -> {len(after_title)}")

    # Drop low-comment threads
    if count_field:
        after_count = [
            r for r in after_title
            if int(r.get(count_field, 0) or 0) >= 2
        ]
    else:
        after_count = after_title
    profile["after_remove_short"] = len(after_count)
    profile["after_remove_automod"] = len(after_count)  # TODO: add automod filter
    log(f"Pre-filter comment-count check: {len(after_title)} -> {len(after_count)}")

    return after_count, profile


# ---------------------------------------------------------------------------
# Corpus streaming
# ---------------------------------------------------------------------------

def stream_records(input_path: pathlib.Path):
    with bz2.open(input_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_all_records(input_path: pathlib.Path, max_threads: int | None) -> list[dict]:
    records: list[dict] = []
    for rec in stream_records(input_path):
        records.append(rec)
        if max_threads and len(records) >= max_threads:
            break
    return records


# ---------------------------------------------------------------------------
# DeBERTa inference
# ---------------------------------------------------------------------------

def build_pipeline(hypothesis: str):
    from transformers import pipeline  # type: ignore

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        log(f"HF_HOME={hf_home}")

    log(f"Loading model: {MODEL_NAME}")
    clf = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=0 if _cuda_available() else -1,
    )
    log("Model loaded.")
    return clf


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except ImportError:
        return False


def gpu_mem_str() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"GPU {used:.1f}/{total:.1f} GB"
    except Exception:
        pass
    return ""


def classify_batch(
    clf,
    texts: list[str],
    hypothesis_template: str,
    threshold: float,
) -> list[tuple[float, str]]:
    """Returns list of (score_for_economics, label) per text.

    Uses two-label contrastive zero-shot NLI. The pipeline fills each
    candidate label into hypothesis_template (which must contain '{}'),
    runs NLI against the input text, and softmax-normalises the
    entailment scores across the two labels.
    """
    if "{}" not in hypothesis_template:
        raise ValueError(
            f"hypothesis_template must contain '{{}}' placeholder. Got: {hypothesis_template!r}"
        )

    results = clf(
        texts,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template=hypothesis_template,
        multi_label=False,
    )
    if isinstance(results, dict):
        results = [results]
    out = []
    for r in results:
        score_map = dict(zip(r["labels"], r["scores"]))
        econ_score = score_map.get(POSITIVE_LABEL, 0.0)
        label = POSITIVE_LABEL if econ_score >= threshold else "not economics"
        out.append((econ_score, label))
    return out

def run_inference(
    clf,
    records: list[dict],
    body_field: str | None,
    title_field: str | None,
    id_field: str | None,
    hypothesis_template: str,
    threshold: float,
    batch_size: int,
) -> list[tuple[dict, float, str]]:
    """Returns list of (record, score, label)."""
    results: list[tuple[dict, float, str]] = []
    total = len(records)
    start = time.time()

    for batch_start in range(0, total, batch_size):
        batch = records[batch_start: batch_start + batch_size]
        texts = []
        for r in batch:
            title = str(r.get(title_field or "", "") or "")
            body = str(r.get(body_field or "", "") or "")
            texts.append((title + " " + body).strip()[:2000])  # rough char cap before tokenizer

        try:
            batch_results = classify_batch(clf, texts, hypothesis_template, threshold)
        except Exception as exc:
            log(f"WARNING: batch {batch_start}-{batch_start+len(batch)} failed: {exc} — skipping")
            for rec in batch:
                results.append((rec, 0.0, "not economics"))
            continue

        for rec, (score, label) in zip(batch, batch_results):
            results.append((rec, score, label))

        processed = batch_start + len(batch)
        if processed % 1000 < batch_size or processed == total:
            elapsed = time.time() - start
            rate = processed / max(elapsed, 1)
            eta = (total - processed) / max(rate, 1)
            mem = gpu_mem_str()
            log(
                f"Progress: {processed}/{total} ({100*processed/total:.1f}%) "
                f"| ETA {eta/60:.1f} min | {mem}"
            )

    return results


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_smoke(
    filtered: list[dict],
    id_field: str | None,
) -> None:
    sample = filtered[:100]
    smoke_path = OUTPUT_DIR / "smoke_thread_ids.txt"
    ids = [str(r.get(id_field or "", "")) for r in sample]
    smoke_path.write_text("\n".join(ids))
    log(f"Smoke mode: wrote {len(ids)} thread IDs to {smoke_path}")


def mode_validation(
    filtered: list[dict],
    body_field: str | None,
    title_field: str | None,
    id_field: str | None,
    hypothesis_template: str,
    threshold: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    sample = rng.sample(filtered, min(50, len(filtered)))
    log(f"Validation: sampled {len(sample)} threads (seed={seed})")

    clf = build_pipeline(hypothesis_template)
    scored = run_inference(clf, sample, body_field, title_field, id_field, hypothesis_template, threshold, batch_size=8)

    out_path = OUTPUT_DIR / "topic_validation_sample.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["thread_id", "title", "gold_label", "model_score", "model_label"])
        writer.writeheader()
        for rec, score, label in scored:
            writer.writerow({
                "thread_id": rec.get(id_field or "", ""),
                "title": str(rec.get(title_field or "", ""))[:300],
                "gold_label": "",
                "model_score": f"{score:.4f}",
                "model_label": label,
            })
    log(f"Validation CSV written to: {out_path}")
    log("Next step: open the CSV, fill in the 'gold_label' column, then compute accuracy (see README).")


def mode_full(
    filtered: list[dict],
    body_field: str | None,
    title_field: str | None,
    id_field: str | None,
    hypothesis_template: str,
    threshold: float,
    batch_size: int,
    profile: dict,
    model_name: str,
) -> None:
    clf = build_pipeline(hypothesis_template)

    wall_start = time.time()
    scored = run_inference(clf, filtered, body_field, title_field, id_field, hypothesis_template, threshold, batch_size)
    wall_seconds = time.time() - wall_start

    kept = [(rec, score, label) for rec, score, label in scored if label == "economics"]
    log(f"Topic filter: {len(kept)}/{len(scored)} threads classified as 'economics'.")

    ids_path = OUTPUT_DIR / "cmv_economics_thread_ids.txt"
    ids_path.write_text("\n".join(str(rec.get(id_field or "", "")) for rec, _, _ in kept))
    log(f"Thread IDs written to: {ids_path}")

    profile_out = {
        **profile,
        "after_topic_filter": len(kept),
        "topic_threshold": threshold,
        "hypothesis_template": hypothesis_template,
        "model_name": model_name,
        "wall_clock_seconds": round(wall_seconds, 1),
    }
    profile_path = OUTPUT_DIR / "cmv_economics_profile.json"
    profile_path.write_text(json.dumps(profile_out, indent=2))
    log(f"Profile written to: {profile_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Filter CMV threads by economics topic.")
    parser.add_argument("--input", required=True, help="Path to .bz2 file or directory.")
    parser.add_argument(
        "--mode", required=True, choices=["smoke", "validation", "full"],
        help="smoke=pre-filter only; validation=50-thread sample; full=all threads.",
    )
    parser.add_argument("--max_threads", type=int, default=None, help="Cap total threads loaded.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default 0.5).")
    parser.add_argument(
        "--hypothesis",
        default="This text is about {}.",
        help=(
            "Hypothesis template for zero-shot NLI. Must contain '{}' "
            "where each candidate label will be inserted. "
            "Default: 'This text is about {}.'."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size (default 8).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for validation sample.")
    args = parser.parse_args()

    # Load schema probe
    if not SCHEMA_PROBE_PATH.exists():
        sys.exit(
            f"[{ts()}] ERROR: {SCHEMA_PROBE_PATH} not found.\n"
            "Run profile_corpus.py --input <path> first to generate it."
        )
    probe = json.loads(SCHEMA_PROBE_PATH.read_text())
    body_field: str | None = probe.get("body_field")
    title_field: str | None = probe.get("title_field")
    id_field: str | None = probe.get("id_field")
    count_field: str | None = probe.get("comment_count_field")
    log(f"Schema probe loaded: body={body_field}, title={title_field}, id={id_field}, count={count_field}")

    input_path = resolve_input(args.input)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Loading records from: {input_path}")
    records = load_all_records(input_path, args.max_threads)
    log(f"Loaded {len(records):,} records.")

    filtered, profile = prefilter(records, body_field, title_field, count_field)
    log(f"After pre-filter: {len(filtered):,} threads.")

    if args.mode == "smoke":
        mode_smoke(filtered, id_field)
        return

    if args.mode == "validation":
        mode_validation(
            filtered, body_field, title_field, id_field,
            args.hypothesis, args.threshold, args.seed,
        )
        return

    if args.mode == "full":
        mode_full(
            filtered, body_field, title_field, id_field,
            args.hypothesis, args.threshold, args.batch_size,
            profile, MODEL_NAME,
        )
        return


if __name__ == "__main__":
    main()
