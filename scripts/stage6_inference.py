"""
Stage 6 — BERT cascade inference on sentences_for_classifier.jsonl.

Repo-root assumption: run from the repository root, i.e.
    python -u scripts/stage6_inference.py --input ... --output ...
Paths to model checkpoints are passed explicitly via --stage1-ckpt / --stage2-ckpt.

Adapted from peer_handoff/inference.py.  The model loading code and
Stage 1 → Stage 2 dispatch are kept identical; additions are:
  - JSONL I/O instead of JSON (streaming read, streaming write)
  - --max-sentences for smoke / profile sub-runs
  - --log-every and flush-on-every-log-line to avoid SLURM buffering
  - VRAM reporting at first batch and every --log-every batches
  - null Stage 2 fields for non-salient sentences
  - --stage1-out reserved for a future disk-intermediate mode
  - --chunk-size for periodic intermediate saves (Defense 1)
  - --resume to continue from a partial output file (Defense 1)
  - per-batch try/except with OOM tracking (Defense 2)

Processing flow (chunk loop):
  Models are loaded once into VRAM before the chunk loop begins.
  For each chunk of --chunk-size input sentences:
    1. Run Stage 1 (salience) over the chunk in batches.
    2. Filter to salient sentences within the chunk.
    3. Run Stage 2 (CLT dimensions) over the salient subset in batches.
    4. Merge Stage 1 + Stage 2 results back into the chunk rows.
    5. Write JSONL output rows for the chunk; fsync.
  Repeat until all input sentences are processed.

OUTPUT SCHEMA
-------------
Each output row contains all input fields plus:

  salience_pred    int or null   1 = salient, 0 = not salient (Stage 1 argmax);
                                 null if the Stage 1 batch failed.
  salience_logits  list or null  [logit_class0, logit_class1] (raw Stage 1 logits);
                                 null if the Stage 1 batch failed.

For salient sentences (salience_pred == 1), Stage 2 runs and adds per-dimension
fields. For non-salient sentences, Stage 2 is not run and these fields are null
(distinguishing "not run" from "ran and predicted N/A"). Also null if the Stage 2
batch failed.

  {dim}_pred    int or null
  {dim}_logits  list[float, float, float] or null   (raw Stage 2 logits, 3 classes)

where {dim} is one of: temporal, spatial, social, hypothetical.

Label encoding (Stage 2, all four dimensions):
  Logit argmax index 0  →  pred value -1  (N/A — construal level not applicable)
  Logit argmax index 1  →  pred value  0  (Near — low construal level)
  Logit argmax index 2  →  pred value  1  (Far  — high construal level)

Mapping: INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}  (argmax-minus-1 encoding)

Source: peer_handoff/src/data.py LABEL_MAP / INV_LABEL_MAP (lines 41-42),
which encodes the raw annotation values {-1, 0, 1} as class indices {0, 1, 2}
for training. Inference inverts this with INV_LABEL_MAP to recover the original
annotation-space values. The config (peer_handoff/configs/bert_stage2_final.yaml)
does not specify a label map; the mapping lives exclusively in data.py.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# peer_handoff/src must be on the path so CLTStage2Model can be imported.
# When running from the repo root, peer_handoff/src is a sibling of scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "peer_handoff"))
from src.model import CLTStage2Model

DIMS = ("temporal", "spatial", "social", "hypothetical")

# argmax-minus-1: class index {0,1,2} → annotation value {-1,0,1} (N/A, Near, Far)
INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}

# Abort if this many consecutive OOM errors occur in a single stage's batch loop.
_OOM_ABORT_THRESHOLD = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def flush() -> None:
    sys.stdout.flush()
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vram_info(device: torch.device) -> str:
    if device.type != "cuda":
        return "no GPU"
    alloc = torch.cuda.memory_allocated(device) / 1e9
    reserv = torch.cuda.memory_reserved(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    return f"VRAM alloc={alloc:.2f}GB reserved={reserv:.2f}GB total={total:.2f}GB"


def make_batches(items: list, batch_size: int) -> list[list]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def iter_chunks(
    input_path: Path,
    chunk_size: int,
    skip_rows: int,
    max_sentences: int | None,
) -> Iterator[list[dict]]:
    """Yield successive chunks of up to chunk_size records from a JSONL file.

    Skips the first skip_rows non-empty lines (for --resume).
    Stops after yielding max_sentences total records (across all chunks) if set.
    Input ordering is stable — sequential read, no shuffling.
    """
    total_yielded = 0
    skipped = 0
    chunk: list[dict] = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if skipped < skip_rows:
                skipped += 1
                continue

            if max_sentences is not None and total_yielded >= max_sentences:
                break

            chunk.append(json.loads(line))
            total_yielded += 1

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

            if max_sentences is not None and total_yielded >= max_sentences:
                break

    if chunk:
        yield chunk


def count_jsonl_rows(path: Path) -> int:
    """Count non-empty lines in a JSONL file (used for --resume)."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# ---------------------------------------------------------------------------
# Stage 1 — salience
# ---------------------------------------------------------------------------

def load_stage1_model(
    ckpt_dir: str,
    device: torch.device,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    log.info("[Stage 1] Loading model from %s ...", ckpt_dir); flush()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.to(device)
    model.eval()
    log.info("[Stage 1] Model loaded. %s", vram_info(device)); flush()
    return model, tokenizer


def run_stage1_on_chunk(
    records: list[dict],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    batch_size: int,
    device: torch.device,
    log_every: int,
    chunk_label: str,
    failure_counts: dict,
) -> tuple[list[bool | None], list[list[float] | None]]:
    """Run Stage 1 salience classifier over one chunk.

    Returns (salient_flags, logits_list), both length len(records).
    Entries are None for any batch that raised an exception.

    failure_counts is mutated in place:
      failure_counts["s1_batches"] += 1 per failed batch
      failure_counts["s1_sentences"] += batch size per failed batch
    """
    texts = [r["sentence_text"] for r in records]
    batches = make_batches(texts, batch_size)

    all_preds:  list[int | None]         = []
    all_logits: list[list[float] | None] = []

    log.info(
        "[Stage 1] %s  %d sentences, %d batches of up to %d",
        chunk_label, len(texts), len(batches), batch_size,
    ); flush()

    t_batch_start = time.time()
    consecutive_oom = 0

    with torch.no_grad():
        for i, batch_texts in enumerate(batches):
            sentence_ids = [
                records[i * batch_size + k].get("sentence_id", f"idx={i * batch_size + k}")
                for k in range(len(batch_texts))
            ]
            try:
                enc = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=128,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                logits = model(**enc).logits
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                batch_logits = logits.cpu().tolist()
                all_preds.extend(preds)
                all_logits.extend(batch_logits)
                consecutive_oom = 0

            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                consecutive_oom += 1
                failure_counts["s1_batches"] += 1
                failure_counts["s1_sentences"] += len(batch_texts)
                log.error(
                    "[Stage 1] OOM batch %d/%d (%d consecutive) — ids: %s — %s",
                    i + 1, len(batches), consecutive_oom, sentence_ids, exc,
                ); flush()
                all_preds.extend([None] * len(batch_texts))
                all_logits.extend([None] * len(batch_texts))
                if consecutive_oom >= _OOM_ABORT_THRESHOLD:
                    raise RuntimeError(
                        f"[Stage 1] {_OOM_ABORT_THRESHOLD} consecutive OOM errors — "
                        "GPU state is broken, aborting."
                    ) from exc

            except Exception as exc:
                consecutive_oom = 0
                failure_counts["s1_batches"] += 1
                failure_counts["s1_sentences"] += len(batch_texts)
                log.error(
                    "[Stage 1] batch %d/%d failed — ids: %s — %s: %s",
                    i + 1, len(batches), sentence_ids, type(exc).__name__, exc,
                ); flush()
                all_preds.extend([None] * len(batch_texts))
                all_logits.extend([None] * len(batch_texts))

            if i == 0 or (i + 1) % log_every == 0 or (i + 1) == len(batches):
                elapsed = time.time() - t_batch_start
                log.info(
                    "[Stage 1] %s  batch %d/%d  wall=%.1fs  %s",
                    chunk_label, i + 1, len(batches), elapsed, vram_info(device),
                ); flush()

    salient_flags = [bool(p) if p is not None else None for p in all_preds]
    n_sal = sum(1 for s in salient_flags if s)
    log.info(
        "[Stage 1] %s  done. %d/%d salient (%.1f%%)",
        chunk_label, n_sal, len(texts), 100 * n_sal / max(len(texts), 1),
    ); flush()
    return salient_flags, all_logits


# ---------------------------------------------------------------------------
# Stage 2 — CLT dimensions
# ---------------------------------------------------------------------------

def load_stage2_model(
    ckpt_path: str,
    device: torch.device,
    model_name: str = "bert-base-uncased",
) -> tuple[CLTStage2Model, AutoTokenizer]:
    log.info("[Stage 2] Loading model from %s ...", ckpt_path); flush()
    model = CLTStage2Model(model_name)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    log.info("[Stage 2] Model loaded. %s", vram_info(device)); flush()
    return model, tokenizer


def run_stage2_on_chunk(
    salient_records: list[dict],
    salient_indices: list[int],
    model: CLTStage2Model,
    tokenizer: AutoTokenizer,
    batch_size: int,
    device: torch.device,
    log_every: int,
    chunk_label: str,
    failure_counts: dict,
) -> dict[int, dict]:
    """Run Stage 2 CLT dimension classifier over the salient sentences in one chunk.

    Returns a dict mapping original-within-chunk index → {dim: {"pred", "logits"}}.
    Entries for a failed batch have pred=None, logits=None for all dims.

    failure_counts is mutated in place:
      failure_counts["s2_batches"] += 1 per failed batch
      failure_counts["s2_sentences"] += batch size per failed batch
    """
    texts = [r["sentence_text"] for r in salient_records]
    batches = make_batches(texts, batch_size)

    all_preds:  dict[str, list[int | None]]         = {d: [] for d in DIMS}
    all_logits: dict[str, list[list[float] | None]] = {d: [] for d in DIMS}

    log.info(
        "[Stage 2] %s  %d salient sentences, %d batches of up to %d",
        chunk_label, len(texts), len(batches), batch_size,
    ); flush()

    t_batch_start = time.time()
    consecutive_oom = 0

    with torch.no_grad():
        for i, batch_texts in enumerate(batches):
            sentence_ids = [
                salient_records[i * batch_size + k].get(
                    "sentence_id", f"idx={salient_indices[i * batch_size + k]}"
                )
                for k in range(len(batch_texts))
            ]
            try:
                enc = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=128,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                logits_dict = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                for dim in DIMS:
                    raw_logits = logits_dict[dim]
                    idx = torch.argmax(raw_logits, dim=-1).cpu().tolist()
                    all_preds[dim].extend(INV_LABEL_MAP[p] for p in idx)
                    all_logits[dim].extend(raw_logits.cpu().tolist())
                consecutive_oom = 0

            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                consecutive_oom += 1
                failure_counts["s2_batches"] += 1
                failure_counts["s2_sentences"] += len(batch_texts)
                log.error(
                    "[Stage 2] OOM batch %d/%d (%d consecutive) — ids: %s — %s",
                    i + 1, len(batches), consecutive_oom, sentence_ids, exc,
                ); flush()
                for dim in DIMS:
                    all_preds[dim].extend([None] * len(batch_texts))
                    all_logits[dim].extend([None] * len(batch_texts))
                if consecutive_oom >= _OOM_ABORT_THRESHOLD:
                    raise RuntimeError(
                        f"[Stage 2] {_OOM_ABORT_THRESHOLD} consecutive OOM errors — "
                        "GPU state is broken, aborting."
                    ) from exc

            except Exception as exc:
                consecutive_oom = 0
                failure_counts["s2_batches"] += 1
                failure_counts["s2_sentences"] += len(batch_texts)
                log.error(
                    "[Stage 2] batch %d/%d failed — ids: %s — %s: %s",
                    i + 1, len(batches), sentence_ids, type(exc).__name__, exc,
                ); flush()
                for dim in DIMS:
                    all_preds[dim].extend([None] * len(batch_texts))
                    all_logits[dim].extend([None] * len(batch_texts))

            if i == 0 or (i + 1) % log_every == 0 or (i + 1) == len(batches):
                elapsed = time.time() - t_batch_start
                log.info(
                    "[Stage 2] %s  batch %d/%d  wall=%.1fs  %s",
                    chunk_label, i + 1, len(batches), elapsed, vram_info(device),
                ); flush()

    result: dict[int, dict] = {}
    for j, orig_idx in enumerate(salient_indices):
        result[orig_idx] = {
            dim: {
                "pred":   all_preds[dim][j],
                "logits": all_logits[dim][j],
            }
            for dim in DIMS
        }

    log.info("[Stage 2] %s  done.", chunk_label); flush()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLT cascade inference — Stage 1 salience + Stage 2 CLT dimensions.",
    )
    parser.add_argument("--input",         required=True,  help="Input JSONL path (sentences_for_classifier.jsonl).")
    parser.add_argument("--output",        required=True,  help="Output JSONL path.")
    parser.add_argument("--stage1-ckpt",   required=True,  help="Stage 1 HuggingFace checkpoint directory.")
    parser.add_argument("--stage2-ckpt",   required=True,  help="Stage 2 .pt state-dict file.")
    parser.add_argument("--batch-size",    type=int, default=32,    help="Inference batch size (default 32).")
    parser.add_argument("--chunk-size",    type=int, default=50000, help="Input sentences per intermediate save (default 50000).")
    parser.add_argument("--max-sentences", type=int, default=None,  help="Process only the first N sentences (default: all).")
    parser.add_argument("--log-every",     type=int, default=100,   help="Log progress every N batches (default 100).")
    parser.add_argument("--device",        default=None,            help="'cuda', 'cpu', or device index. Default: cuda if available.")
    parser.add_argument("--resume",        action="store_true",     help="Resume from existing output: skip already-written rows, append to output file.")
    parser.add_argument("--stage1-out",    default=None,            help="Reserved: path for a future disk-intermediate Stage 1 output. Unused.")
    args = parser.parse_args()

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("Device: %s", device); flush()
    if device.type == "cuda":
        log.info(
            "  GPU: %s  (%.1f GB total)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        ); flush()

    # Validate checkpoint paths
    if not Path(args.stage1_ckpt).is_dir():
        sys.exit(f"ERROR: --stage1-ckpt directory not found: {args.stage1_ckpt}")
    if not Path(args.stage2_ckpt).is_file():
        sys.exit(f"ERROR: --stage2-ckpt file not found: {args.stage2_ckpt}")

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: --input file not found: {args.input}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --resume: count rows already written, skip that many input rows, append.
    # Fresh run: truncate the output file so stale data from a prior run is gone.
    skip_rows = 0
    file_mode = "w"
    if args.resume:
        if out_path.exists():
            skip_rows = count_jsonl_rows(out_path)
            file_mode = "a"
            log.info(
                "[Resume] Output file has %d rows — skipping that many input rows and appending.",
                skip_rows,
            ); flush()
        else:
            log.info("[Resume] Output file does not exist yet — starting fresh."); flush()
    else:
        # Truncate now so the file is empty before any chunks are written.
        out_path.open("w").close()

    # Load both models once — expensive VRAM load happens here, not per-chunk.
    s1_model, s1_tokenizer = load_stage1_model(args.stage1_ckpt, device)
    s2_model, s2_tokenizer = load_stage2_model(args.stage2_ckpt, device)

    # Failure counters accumulated across all chunks.
    failure_counts = {
        "s1_batches": 0, "s1_sentences": 0,
        "s2_batches": 0, "s2_sentences": 0,
    }

    # Run stats accumulated for the final summary.
    total_sentences  = 0
    total_salient    = 0
    total_written    = 0
    chunk_index      = 0

    # Chunk loop — Defense 1: write + fsync after every chunk.
    log.info(
        "Starting chunk loop (chunk_size=%d, skip_rows=%d, max_sentences=%s) ...",
        args.chunk_size, skip_rows, args.max_sentences,
    ); flush()

    with open(out_path, file_mode, encoding="utf-8") as fout:
        for chunk in iter_chunks(input_path, args.chunk_size, skip_rows, args.max_sentences):
            chunk_index += 1
            chunk_label = f"[chunk {chunk_index}]"
            log.info("%s  %d sentences (global written so far: %d)", chunk_label, len(chunk), total_written); flush()

            # Stage 1 over this chunk.
            salient_flags, s1_logits = run_stage1_on_chunk(
                records=chunk,
                model=s1_model,
                tokenizer=s1_tokenizer,
                batch_size=args.batch_size,
                device=device,
                log_every=args.log_every,
                chunk_label=chunk_label,
                failure_counts=failure_counts,
            )

            # Filter to salient sentences within this chunk.
            # Indices here are local (0-based within the chunk).
            salient_indices = [i for i, s in enumerate(salient_flags) if s]
            salient_records = [chunk[i] for i in salient_indices]

            # Stage 2 over salient sentences in this chunk.
            stage2_results: dict[int, dict] = {}
            if salient_records:
                stage2_results = run_stage2_on_chunk(
                    salient_records=salient_records,
                    salient_indices=salient_indices,
                    model=s2_model,
                    tokenizer=s2_tokenizer,
                    batch_size=args.batch_size,
                    device=device,
                    log_every=args.log_every,
                    chunk_label=chunk_label,
                    failure_counts=failure_counts,
                )
            else:
                log.info("%s  [Stage 2] Skipped — no salient sentences.", chunk_label); flush()

            # Merge and write JSONL for this chunk.
            chunk_salient = 0
            for i, rec in enumerate(chunk):
                out = dict(rec)

                s1_flag  = salient_flags[i]
                s1_logit = s1_logits[i]
                out["salience_pred"]   = int(s1_flag) if s1_flag is not None else None
                out["salience_logits"] = (
                    [round(v, 6) for v in s1_logit] if s1_logit is not None else None
                )

                if i in stage2_results:
                    s2 = stage2_results[i]
                    for dim in DIMS:
                        pred   = s2[dim]["pred"]
                        logits = s2[dim]["logits"]
                        out[f"{dim}_pred"]   = pred
                        out[f"{dim}_logits"] = (
                            [round(v, 6) for v in logits] if logits is not None else None
                        )
                    if s1_flag:
                        chunk_salient += 1
                else:
                    # Non-salient or Stage 1 failed: Stage 2 not run.
                    for dim in DIMS:
                        out[f"{dim}_pred"]   = None
                        out[f"{dim}_logits"] = None

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            # Flush + fsync so the OS actually commits to disk before next chunk.
            fout.flush()
            os.fsync(fout.fileno())

            total_sentences += len(chunk)
            total_salient   += chunk_salient
            total_written   += len(chunk)

            log.info(
                "%s  complete — %d sentences written, %d salient  (global: %d written)",
                chunk_label, len(chunk), chunk_salient, total_written,
            ); flush()

    # Final summary.
    total_batches = (
        (total_sentences + args.batch_size - 1) // args.batch_size
        + (total_salient  + args.batch_size - 1) // args.batch_size
    )
    failed_batches = failure_counts["s1_batches"] + failure_counts["s2_batches"]

    log.info("=== Stage 6 complete ==="); flush()
    log.info("  total sentences : %d", total_sentences + skip_rows); flush()
    log.info("  processed now   : %d  (skipped %d via --resume)", total_sentences, skip_rows); flush()
    log.info("  salient (new)   : %d  (%.1f%%)", total_salient, 100 * total_salient / max(total_sentences, 1)); flush()
    log.info("  output          : %s  (%d rows total)", out_path, total_written + skip_rows); flush()
    log.info(
        "  batch failures  : S1=%d (%d sentences)  S2=%d (%d sentences)",
        failure_counts["s1_batches"], failure_counts["s1_sentences"],
        failure_counts["s2_batches"], failure_counts["s2_sentences"],
    ); flush()

    if total_batches > 0 and failed_batches / max(total_batches, 1) > 0.01:
        log.warning(
            "WARNING: >1%% of batches failed (%d/%d) — likely a systematic problem, "
            "not random bad inputs. Inspect stderr above.",
            failed_batches, total_batches,
        ); flush()

    try:
        # Per-dimension distribution summary over newly-processed salient sentences.
        # Reads back from the output file to cover all chunks uniformly.
        if total_salient > 0:
            dim_counts: dict[str, dict[int, int]] = {
                dim: {-1: 0, 0: 0, 1: 0} for dim in DIMS
            }
            rows_checked = 0
            with open(out_path, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("salience_pred") != 1:
                        continue
                    rows_checked += 1
                    for dim in DIMS:
                        v = row.get(f"{dim}_pred")
                        if v in (-1, 0, 1):
                            dim_counts[dim][v] += 1
            for dim in DIMS:
                c = dim_counts[dim]
                log.info("  %-14s N/A=%d  Near=%d  Far=%d", dim, c[-1], c[0], c[1]); flush()
    except Exception as exc:
        log.warning("Post-write summary failed (output is unaffected): %s", exc); flush()


if __name__ == "__main__":
    main()
