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

I/O between Stage 1 and Stage 2 is currently in-memory (single pass):
all sentences are read, Stage 1 runs over them, then Stage 2 runs over
the salient subset, and results are merged before writing. This is
appropriate for the smoke test and profile run. Whether to switch to a
disk-intermediate mode (write Stage 1 output, re-read for Stage 2) will
be decided after the profile run reveals memory pressure, if any.

OUTPUT SCHEMA
-------------
Each output row contains all input fields plus:

  salience_pred    int   1 = salient, 0 = not salient (Stage 1 argmax)
  salience_logits  list  [logit_class0, logit_class1] (raw Stage 1 logits)

For salient sentences (salience_pred == 1), Stage 2 runs and adds per-dimension
fields. For non-salient sentences, Stage 2 is not run and these fields are null
(distinguishing "not run" from "ran and predicted N/A"):

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
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# peer_handoff/src must be on the path so CLTStage2Model can be imported.
# When running from the repo root, peer_handoff/src is a sibling of scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "peer_handoff"))
from src.model import CLTStage2Model

DIMS = ("temporal", "spatial", "social", "hypothetical")

# argmax-minus-1: class index {0,1,2} → annotation value {-1,0,1} (N/A, Near, Far)
INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}

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


def load_sentences(input_path: Path, max_sentences: int | None) -> list[dict]:
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_sentences is not None and len(records) >= max_sentences:
                break
    return records


# ---------------------------------------------------------------------------
# Stage 1 — salience
# ---------------------------------------------------------------------------

def run_stage1(
    records: list[dict],
    ckpt_dir: str,
    batch_size: int,
    device: torch.device,
    log_every: int,
) -> tuple[list[bool], list[list[float]]]:
    """Returns (salient_flags, logits_list).

    salient_flags[i] is True if sentence i is predicted salient.
    logits_list[i] is [logit_class0, logit_class1] as Python floats.
    """
    log.info("[Stage 1] Loading model from %s ...", ckpt_dir); flush()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.to(device)
    model.eval()
    log.info("[Stage 1] Model loaded. %s", vram_info(device)); flush()

    texts = [r["sentence_text"] for r in records]
    batches = make_batches(texts, batch_size)
    all_preds: list[int] = []
    all_logits: list[list[float]] = []

    log.info(
        "[Stage 1] %d sentences, %d batches of up to %d",
        len(texts), len(batches), batch_size,
    ); flush()

    t_batch_start = time.time()
    with torch.no_grad():
        for i, batch_texts in enumerate(batches):
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

            if i == 0 or (i + 1) % log_every == 0 or (i + 1) == len(batches):
                elapsed = time.time() - t_batch_start
                log.info(
                    "[Stage 1] batch %d/%d  wall=%.1fs  %s",
                    i + 1, len(batches), elapsed, vram_info(device),
                ); flush()

    salient_flags = [bool(p) for p in all_preds]
    n_sal = sum(salient_flags)
    log.info(
        "[Stage 1] Done. %d/%d salient (%.1f%%)",
        n_sal, len(texts), 100 * n_sal / max(len(texts), 1),
    ); flush()
    return salient_flags, all_logits


# ---------------------------------------------------------------------------
# Stage 2 — CLT dimensions
# ---------------------------------------------------------------------------

def run_stage2(
    salient_records: list[dict],
    salient_indices: list[int],
    ckpt_path: str,
    batch_size: int,
    device: torch.device,
    log_every: int,
    model_name: str = "bert-base-uncased",
) -> dict[int, dict]:
    """Returns a dict mapping original record index → {dim: (pred_int, logits_list)}.

    pred_int is in {-1 (N/A), 0 (Near), 1 (Far)}.
    logits_list is [float, float, float] for the three classes.
    """
    log.info("[Stage 2] Loading model from %s ...", ckpt_path); flush()
    model = CLTStage2Model(model_name)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    log.info("[Stage 2] Model loaded. %s", vram_info(device)); flush()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = [r["sentence_text"] for r in salient_records]
    batches = make_batches(texts, batch_size)

    all_preds:  dict[str, list[int]]         = {d: [] for d in DIMS}
    all_logits: dict[str, list[list[float]]] = {d: [] for d in DIMS}

    log.info(
        "[Stage 2] %d salient sentences, %d batches of up to %d",
        len(texts), len(batches), batch_size,
    ); flush()

    t_batch_start = time.time()
    with torch.no_grad():
        for i, batch_texts in enumerate(batches):
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

            if i == 0 or (i + 1) % log_every == 0 or (i + 1) == len(batches):
                elapsed = time.time() - t_batch_start
                log.info(
                    "[Stage 2] batch %d/%d  wall=%.1fs  %s",
                    i + 1, len(batches), elapsed, vram_info(device),
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

    log.info("[Stage 2] Done."); flush()
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
    parser.add_argument("--batch-size",    type=int, default=32,   help="Inference batch size (default 32).")
    parser.add_argument("--max-sentences", type=int, default=None, help="Process only the first N sentences (default: all).")
    parser.add_argument("--log-every",     type=int, default=100,  help="Log progress every N batches (default 100).")
    parser.add_argument("--device",        default=None,           help="'cuda', 'cpu', or device index. Default: cuda if available.")
    parser.add_argument("--stage1-out",    default=None,           help="Reserved: path for a future disk-intermediate Stage 1 output. Unused.")
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

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: --input file not found: {args.input}")

    log.info("Loading sentences from %s (max=%s) ...", input_path, args.max_sentences); flush()
    t0 = time.time()
    records = load_sentences(input_path, args.max_sentences)
    log.info("Loaded %d sentences in %.1fs", len(records), time.time() - t0); flush()

    # Stage 1
    salient_flags, s1_logits = run_stage1(
        records=records,
        ckpt_dir=args.stage1_ckpt,
        batch_size=args.batch_size,
        device=device,
        log_every=args.log_every,
    )

    salient_indices = [i for i, s in enumerate(salient_flags) if s]
    salient_records = [records[i] for i in salient_indices]

    # Stage 2
    stage2_results: dict[int, dict] = {}
    if salient_records:
        stage2_results = run_stage2(
            salient_records=salient_records,
            salient_indices=salient_indices,
            ckpt_path=args.stage2_ckpt,
            batch_size=args.batch_size,
            device=device,
            log_every=args.log_every,
        )
    else:
        log.info("[Stage 2] Skipped — no sentences predicted salient."); flush()

    # Write output — one row per input sentence, including non-salient
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing output to %s ...", out_path); flush()

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            out = dict(rec)
            out["salience_pred"]   = int(salient_flags[i])
            out["salience_logits"] = [round(v, 6) for v in s1_logits[i]]

            if i in stage2_results:
                s2 = stage2_results[i]
                for dim in DIMS:
                    out[f"{dim}_pred"]   = s2[dim]["pred"]
                    out[f"{dim}_logits"] = [round(v, 6) for v in s2[dim]["logits"]]
            else:
                # Non-salient: Stage 2 was not run. null distinguishes
                # "not run" from "ran and predicted N/A (-1)".
                for dim in DIMS:
                    out[f"{dim}_pred"]   = None
                    out[f"{dim}_logits"] = None

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    n_sal = sum(salient_flags)
    log.info("=== Stage 6 complete ==="); flush()
    log.info("  total sentences : %d", len(records)); flush()
    log.info("  salient         : %d  (%.1f%%)", n_sal, 100 * n_sal / max(len(records), 1)); flush()
    log.info("  non-salient     : %d", len(records) - n_sal); flush()
    log.info("  output          : %s", out_path); flush()
    try:
        if stage2_results:
            for dim in DIMS:
                vals = [stage2_results[i][dim]["pred"] for i in salient_indices]
                counts = {v: vals.count(v) for v in (-1, 0, 1)}
                log.info("  %-14s N/A=%d  Near=%d  Far=%d", dim, counts[-1], counts[0], counts[1]); flush()
    except Exception as exc:
        log.warning("Post-write summary failed (output is unaffected): %s", exc); flush()


if __name__ == "__main__":
    main()
