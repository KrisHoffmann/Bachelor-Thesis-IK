"""
CLT Cascade Inference Script
=============================
Runs the trained BERT-base cascade (Stage 1 + Stage 2) on a JSON file of sentences.

Stage 1 — salience classifier (binary):
    Predicts whether each sentence is salient (relevant to the author's argument).

Stage 2 — CLT dimension classifier (four parallel heads):
    For salient sentences only, predicts Temporal / Spatial / Social / Hypothetical
    Construal Level on a Near / Far / N/A scale.

Output encoding:
    salient          : bool   (True = salient)
    temporal         : int    (-1 = N/A, 0 = Near, 1 = Far)
    spatial          : int    (-1 = N/A, 0 = Near, 1 = Far)
    social           : int    (-1 = N/A, 0 = Near, 1 = Far)
    hypothetical     : int    (-1 = N/A, 0 = Near, 1 = Far)

Non-salient sentences are assigned -1 for all four Stage 2 dimensions
(Stage 2 was not run on them — consistent with how the cascade was trained).

WARNING — Temporal and Spatial predictions are unreliable on small datasets.
See README.md for details.

Input JSON schema
-----------------
A JSON array of objects. Each object MUST contain:
    sentence_text : str   — the sentence to classify

Any additional fields (e.g. post_id, sentence_idx) are preserved in the output.

Example:
    [
      {"sentence_text": "Future generations will benefit from this.", "post_id": "p1", "sentence_idx": 0},
      {"sentence_text": "I had lunch.", "post_id": "p1", "sentence_idx": 1}
    ]

CLI usage
---------
    python inference.py \\
        --input example_input.json \\
        --stage1_model models/stage1_model \\
        --stage2_model models/stage2_best_model.pt \\
        --output predictions.json \\
        [--batch_size 32]

Model weight layout (drop into peer_handoff/models/)
----------------------------------------------------
    models/
    ├── stage1_model/           # HuggingFace checkpoint directory
    │   ├── config.json
    │   ├── model.safetensors   (or pytorch_model.bin)
    │   ├── tokenizer.json
    │   └── tokenizer_config.json
    └── stage2_best_model.pt    # raw PyTorch state dict (.pt file)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Allow `python inference.py` from inside peer_handoff/ with src/ as a sibling
sys.path.insert(0, str(Path(__file__).parent))
from src.model import CLTStage2Model

DIMS = ("temporal", "spatial", "social", "hypothetical")

# Stage 2 class-index → raw label: {0→-1 (N/A), 1→0 (Near), 2→1 (Far)}
INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_input(path: str) -> list[dict]:
    """Load and validate input JSON. Crashes with a clear message on bad format."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"ERROR: input file not found: {path}")
    with open(p, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            sys.exit(f"ERROR: input file is not valid JSON: {e}")
    if not isinstance(data, list):
        sys.exit("ERROR: input JSON must be a list of objects (got a non-list top-level value).")
    if len(data) == 0:
        sys.exit("ERROR: input JSON list is empty.")
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            sys.exit(f"ERROR: record {i} is not a JSON object.")
        if "sentence_text" not in rec:
            sys.exit(
                f"ERROR: record {i} is missing required field 'sentence_text'. "
                f"Got keys: {list(rec.keys())}"
            )
        if not isinstance(rec["sentence_text"], str) or not rec["sentence_text"].strip():
            sys.exit(f"ERROR: record {i} has an empty or non-string 'sentence_text'.")
    return data


def check_model_paths(stage1_model: str, stage2_model: str) -> None:
    if not Path(stage1_model).is_dir():
        sys.exit(
            f"ERROR: stage1_model directory not found: {stage1_model}\n"
            "Expected a HuggingFace checkpoint directory (config.json, model weights, tokenizer).\n"
            "See README.md for the expected layout under peer_handoff/models/."
        )
    required_s1 = ["config.json", "tokenizer_config.json"]
    missing = [f for f in required_s1 if not (Path(stage1_model) / f).exists()]
    if missing:
        sys.exit(
            f"ERROR: stage1_model directory is missing expected files: {missing}\n"
            f"Found: {os.listdir(stage1_model)}"
        )
    if not Path(stage2_model).is_file():
        sys.exit(
            f"ERROR: stage2_model file not found: {stage2_model}\n"
            "Expected a PyTorch state dict (.pt file) saved by torch.save(model.state_dict(), ...).\n"
            "See README.md for the expected layout under peer_handoff/models/."
        )


def make_batches(items: list, batch_size: int) -> list[list]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


# ---------------------------------------------------------------------------
# Stage 1 — salience inference
# ---------------------------------------------------------------------------

def run_stage1(
    records: list[dict],
    model_dir: str,
    batch_size: int,
    device: torch.device,
) -> list[bool]:
    """Run Stage 1 salience classifier. Returns a bool list (True = salient)."""
    print(f"\n[Stage 1] Loading model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    texts = [r["sentence_text"] for r in records]
    batches = make_batches(texts, batch_size)
    all_preds: list[int] = []

    print(f"[Stage 1] Running inference on {len(texts)} sentences "
          f"({len(batches)} batches of up to {batch_size}) ...")

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
            all_preds.extend(preds)
            if (i + 1) % 10 == 0 or (i + 1) == len(batches):
                print(f"  batch {i+1}/{len(batches)}")

    salient_flags = [bool(p) for p in all_preds]
    n_salient = sum(salient_flags)
    print(f"[Stage 1] Done. {n_salient}/{len(texts)} sentences predicted salient "
          f"({100*n_salient/len(texts):.1f}%).")
    return salient_flags


# ---------------------------------------------------------------------------
# Stage 2 — CLT dimension inference
# ---------------------------------------------------------------------------

def run_stage2(
    salient_records: list[dict],
    salient_indices: list[int],
    model_path: str,
    batch_size: int,
    device: torch.device,
    model_name: str = "bert-base-uncased",
) -> dict[int, dict[str, int]]:
    """Run Stage 2 on salient sentences.

    Returns a dict mapping original record index → {dim: raw_label} where
    raw_label is in {-1 (N/A), 0 (Near), 1 (Far)}.

    Stage 2 model loading pattern (from src/train.py line 232):
        torch.save(model.state_dict(), run_dir / "best_model.pt")
    and loaded back with:
        model.load_state_dict(torch.load(path, map_location=device))
    CLTStage2Model is instantiated with just the model_name string; the
    state dict is loaded on top. The encoder is AutoModel (not
    AutoModelForSequenceClassification), and [CLS] pooling is used.
    """
    print(f"\n[Stage 2] Loading model from {model_path} ...")
    model = CLTStage2Model(model_name)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = [r["sentence_text"] for r in salient_records]
    batches = make_batches(texts, batch_size)

    print(f"[Stage 2] Running inference on {len(texts)} salient sentences "
          f"({len(batches)} batches of up to {batch_size}) ...")

    all_preds: dict[str, list[int]] = {d: [] for d in DIMS}

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
                idx = torch.argmax(logits_dict[dim], dim=-1).cpu().tolist()
                all_preds[dim].extend(INV_LABEL_MAP[p] for p in idx)
            if (i + 1) % 10 == 0 or (i + 1) == len(batches):
                print(f"  batch {i+1}/{len(batches)}")

    # Map back to original indices
    result: dict[int, dict[str, int]] = {}
    for j, orig_idx in enumerate(salient_indices):
        result[orig_idx] = {dim: all_preds[dim][j] for dim in DIMS}

    print(f"[Stage 2] Done.")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLT cascade inference: Stage 1 salience + Stage 2 CLT dimensions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",        required=True,  help="Path to input JSON file.")
    parser.add_argument("--stage1_model", required=True,  help="Path to Stage 1 HF checkpoint directory.")
    parser.add_argument("--stage2_model", required=True,  help="Path to Stage 2 .pt state-dict file.")
    parser.add_argument("--output",       required=True,  help="Path for output JSON file.")
    parser.add_argument("--batch_size",   type=int, default=32, help="Inference batch size (default: 32).")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # Validate paths
    check_model_paths(args.stage1_model, args.stage2_model)

    # Load input
    records = load_input(args.input)
    print(f"\nLoaded {len(records)} sentences from {args.input}")

    # Stage 1
    salient_flags = run_stage1(records, args.stage1_model, args.batch_size, device)

    salient_indices = [i for i, s in enumerate(salient_flags) if s]
    salient_records = [records[i] for i in salient_indices]

    # Stage 2 (only if at least one sentence is salient)
    stage2_results: dict[int, dict[str, int]] = {}
    if salient_records:
        stage2_results = run_stage2(
            salient_records=salient_records,
            salient_indices=salient_indices,
            model_path=args.stage2_model,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        print("\n[Stage 2] Skipped — no sentences predicted salient.")

    # Assemble output: preserve all original fields, add predictions
    na_dims = {dim: -1 for dim in DIMS}
    output = []
    for i, rec in enumerate(records):
        out = dict(rec)  # preserve all input fields
        out["salient"] = salient_flags[i]
        if i in stage2_results:
            out.update(stage2_results[i])
        else:
            out.update(na_dims)
        output.append(out)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nPredictions written to {out_path}  ({len(output)} records)")

    # Summary
    n_sal = sum(salient_flags)
    print(f"\nSummary:")
    print(f"  Total sentences : {len(output)}")
    print(f"  Salient         : {n_sal}  ({100*n_sal/len(output):.1f}%)")
    print(f"  Non-salient     : {len(output) - n_sal}")
    if stage2_results:
        for dim in DIMS:
            vals = [stage2_results[i][dim] for i in salient_indices]
            counts = {v: vals.count(v) for v in (-1, 0, 1)}
            print(f"  {dim:<14} N/A={counts[-1]}  Near={counts[0]}  Far={counts[1]}")


if __name__ == "__main__":
    main()
