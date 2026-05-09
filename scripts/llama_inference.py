"""
LLaMA-3.2-3B-Instruct prompted classifier — Stage 1 (salience) and Stage 2 (CLT dims).

parse_failure_fallback
  Stage 1: 0 (non-salient) — see configs/llama_stage1.yaml
  Stage 2: {"temporal":-1,"spatial":-1,"social":-1,"hypothetical":-1} (all N/A) — see configs/llama_stage2.yaml

Calibration algorithm (--split calibration)
  1. Load dev.json as a list of raw JSON records.
  2. Extract all sentence_ids (task_id values) in file order.
  3. De-duplicate while preserving order to get unique_task_ids.
  4. Seed a random.Random(42) instance (NOT the global random state).
  5. Shuffle unique_task_ids with that seeded RNG.
  6. Take the first 30 task_ids from the shuffled list.
  7. Return all records whose task_id is in that set, in original file order.
  This is deterministic across Python versions because random.Random(seed).shuffle
  is stable within CPython and we operate on task_ids (ints), not objects.

Smoke split (--split smoke)
  Returns the first 5 records of dev.json sorted ascending by (task_id, sentence_idx).
  Deterministic; no randomness involved.

Dev-set protection
  Running --split dev requires --confirm-final-run AND a prompt_template_version
  that has NOT been previously logged in outputs/llama_{stage}_dev_predictions.json.
  This prevents accidental dev-set hammering and forces a version bump per full run.

Test-set protection
  Running --split test requires --allow-test-set. Hard-fail otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── repo root on sys.path so src imports work regardless of cwd ──────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.data import DIMS
from src.evaluate import compute_metrics_dict, evaluate_stage2

# ── constants ────────────────────────────────────────────────────────────────
CALIBRATION_SEED = 42
CALIBRATION_N = 30
SMOKE_N = 5


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_raw_records(data_dir: Path, split_name: str) -> list[dict]:
    """Load raw JSON records for a given split name (train/dev/test)."""
    actual = "dev" if split_name in ("calibration", "smoke") else split_name
    path = data_dir / f"{actual}.json"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _smoke_records(records: list[dict]) -> list[dict]:
    """Return first 5 records of dev sorted ascending by (task_id, sentence_idx)."""
    sorted_records = sorted(records, key=lambda r: (r["task_id"], r["sentence_idx"]))
    return sorted_records[:SMOKE_N]


def _calibration_records(records: list[dict]) -> list[dict]:
    """Return the fixed 30-sentence calibration subset of dev.

    Algorithm (see module docstring for the authoritative description):
    1. Collect unique task_ids in file order (first occurrence wins).
    2. Shuffle with seeded RNG(42).
    3. Take first 30 task_ids.
    4. Return all records with a matching task_id, preserving original order.
    """
    seen: dict[int, None] = {}
    for r in records:
        seen.setdefault(r["task_id"], None)
    unique_task_ids = list(seen.keys())

    rng = random.Random(CALIBRATION_SEED)
    rng.shuffle(unique_task_ids)
    selected_ids = set(unique_task_ids[:CALIBRATION_N])

    return [r for r in records if r["task_id"] in selected_ids]


def _filter_records(records: list[dict], split_name: str) -> list[dict]:
    if split_name == "smoke":
        return _smoke_records(records)
    if split_name == "calibration":
        return _calibration_records(records)
    return records  # train / dev / test — full file


def _check_dev_protection(
    stage: int,
    prompt_template_version: str,
    confirm_final_run: bool,
    output_dir: Path,
) -> None:
    if not confirm_final_run:
        raise RuntimeError(
            "--split dev requires --confirm-final-run. "
            "The full dev evaluation runs ONCE per prompt_template_version and "
            "its result is final. Pass --confirm-final-run to acknowledge this."
        )
    pred_path = output_dir / f"llama_stage{stage}_dev_predictions.json"
    if pred_path.exists():
        with open(pred_path, encoding="utf-8") as f:
            existing = json.load(f)
        prev_version = existing.get("prompt_template_version") if isinstance(existing, dict) else None
        if prev_version is None and isinstance(existing, list) and existing:
            prev_version = existing[0].get("prompt_template_version")
        if prev_version == prompt_template_version:
            raise RuntimeError(
                f"Dev predictions for prompt_template_version='{prompt_template_version}' "
                f"already exist in {pred_path}. "
                "Bump prompt_template_version in the config before running again."
            )


def _check_test_protection(allow_test_set: bool) -> None:
    if not allow_test_set:
        raise RuntimeError(
            "--split test requires --allow-test-set. "
            "The test set is touched ONCE, after the 9-model selection is final."
        )


# ── prompt building ───────────────────────────────────────────────────────────

def _build_few_shot_block_stage1(
    cfg: dict,
    raw_train: list[dict],
    template: str,
) -> str:
    """Build few-shot block for Stage 1 from hard-coded task_ids in config."""
    index: dict[tuple[int, int], dict] = {
        (r["task_id"], r["sentence_idx"]): r for r in raw_train
    }
    lines: list[str] = []
    for ex in cfg["few_shot_task_ids"]:
        key = (ex["task_id"], ex["sentence_idx"])
        if key not in index:
            raise KeyError(
                f"Few-shot example (task_id={ex['task_id']}, "
                f"sentence_idx={ex['sentence_idx']}) not found in train.json"
            )
        rec = index[key]
        rendered = template.format(
            sentence=rec["sentence_text"],
            label=ex["label"],
        )
        lines.append(rendered.rstrip())
    return "\n".join(lines) + "\n" if lines else ""


def _build_few_shot_block_stage2(
    cfg: dict,
    raw_train: list[dict],
    template: str,
) -> str:
    """Build few-shot block for Stage 2 from hard-coded task_ids in config."""
    index: dict[tuple[int, int], dict] = {
        (r["task_id"], r["sentence_idx"]): r for r in raw_train
    }
    lines: list[str] = []
    for ex in cfg["few_shot_task_ids"]:
        key = (ex["task_id"], ex["sentence_idx"])
        if key not in index:
            raise KeyError(
                f"Few-shot example (task_id={ex['task_id']}, "
                f"sentence_idx={ex['sentence_idx']}) not found in train.json"
            )
        rec = index[key]
        rendered = template.format(
            sentence=rec["sentence_text"],
            temporal=ex["temporal"],
            spatial=ex["spatial"],
            social=ex["social"],
            hypothetical=ex["hypothetical"],
        )
        lines.append(rendered.rstrip())
    return "\n".join(lines) + "\n" if lines else ""


def _render_stage1_prompt(cfg: dict, few_shot_block: str, sentence: str) -> str:
    return cfg["user_prompt_template"].format(
        few_shot_block=few_shot_block,
        sentence=sentence,
    ).rstrip()


def _render_stage2_prompt(cfg: dict, few_shot_block: str, sentence: str) -> str:
    return cfg["user_prompt_template"].format(
        few_shot_block=few_shot_block,
        sentence=sentence,
    ).rstrip()


def _apply_chat_template(
    tokenizer,
    system_prompt: str,
    user_message: str,
) -> str:
    """Format messages through the model's chat template, return decoded string."""
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ── parsers ───────────────────────────────────────────────────────────────────

def _parse_stage1(raw_text: str, fallback: int) -> tuple[int | None, str, str | None]:
    """Parse Stage 1 model output. Returns (prediction, status, error_str)."""
    stripped = raw_text.strip()
    # Accept the first '0' or '1' in the output.
    for char in stripped:
        if char in ("0", "1"):
            return int(char), "ok", None
    return None, "fail_format", f"no '0' or '1' found in: {stripped!r}"


def _parse_stage2(
    raw_text: str,
    fallback: dict,
) -> tuple[dict | None, str, str | None]:
    """Parse Stage 2 JSON output. Returns (prediction_dict, status, error_str).

    prediction_dict values are RAW {-1, 0, 1} integers.
    """
    stripped = raw_text.strip()
    # Find the first '{...}' in the output
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "fail_format", f"no JSON object found in: {stripped!r}"

    json_str = stripped[start : end + 1]
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, "fail_format", f"JSON decode error: {e}  raw: {json_str!r}"

    result: dict[str, int] = {}
    for dim in DIMS:
        if dim not in parsed:
            return None, "fail_format", f"missing dimension '{dim}' in: {parsed}"
        val = parsed[dim]
        if not isinstance(val, int):
            try:
                val = int(val)
            except (ValueError, TypeError):
                return None, "fail_value", f"non-int value for '{dim}': {val!r}"
        if val not in (-1, 0, 1):
            return None, "fail_value", f"value {val} for '{dim}' not in {{-1,0,1}}"
        result[dim] = val

    return result, "ok", None


# ── generation ────────────────────────────────────────────────────────────────

def _generate_one(
    model,
    tokenizer,
    rendered_prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, float]:
    """Run single greedy generation. Returns (decoded_output_text, elapsed_seconds)."""
    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,  # ignored when do_sample=False; set to silence warnings
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    # Decode only the newly generated tokens
    new_ids = output_ids[0, input_len:]
    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return raw_text, elapsed


# ── logging ───────────────────────────────────────────────────────────────────

def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _write_predictions_stage1(
    path: Path,
    records: list[dict],
    prompt_template_version: str,
) -> None:
    """Write Stage 1 prediction file.

    Format: {"prompt_template_version": ..., "predictions": [{"task_id", "sentence_idx", "prediction_int"}]}
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt_template_version": prompt_template_version,
        "predictions": records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_predictions_stage2(
    path: Path,
    records: list[dict],
    prompt_template_version: str,
) -> None:
    """Write Stage 2 prediction file.

    Format: {"prompt_template_version": ..., "predictions": [{"task_id", "sentence_idx", "predictions": {dim: raw_int}}]}
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt_template_version": prompt_template_version,
        "predictions": records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def run_stage1(
    cfg: dict,
    split: str,
    limit: int | None,
    model,
    tokenizer,
    device: torch.device,
    data_dir: Path,
    output_dir: Path,
    raw_train: list[dict],
    confirm_final_run: bool,
    allow_test_set: bool,
) -> None:
    if split == "test":
        _check_test_protection(allow_test_set)
    if split == "dev":
        _check_dev_protection(1, cfg["prompt_template_version"], confirm_final_run, output_dir)

    raw_log_path = output_dir / f"llama_stage1_{split}_raw.jsonl"
    pred_path = output_dir / f"llama_stage1_{split}_predictions.json"

    all_records = _load_raw_records(data_dir, split)
    records = _filter_records(all_records, split)
    if limit is not None:
        records = records[:limit]

    few_shot_block = _build_few_shot_block_stage1(
        cfg, raw_train, cfg["few_shot_example_template"]
    )

    fallback: int = cfg["parse_failure_fallback"]
    max_new_tokens: int = cfg["max_new_tokens"]
    system_prompt: str = cfg["system_prompt"]
    version: str = cfg["prompt_template_version"]

    predictions_out: list[dict] = []
    gold_labels: list[int] = []
    pred_labels: list[int] = []

    n_fail = 0
    print(f"\n[Stage 1] split={split}  n={len(records)}  version={version}")
    for i, rec in enumerate(records):
        sentence = rec["sentence_text"]
        gold_int = 1 if rec["salient"] else 0

        user_msg = _render_stage1_prompt(cfg, few_shot_block, sentence)
        rendered_prompt = _apply_chat_template(tokenizer, system_prompt, user_msg)

        raw_text, elapsed = _generate_one(
            model, tokenizer, rendered_prompt, max_new_tokens, device
        )

        prediction, status, error = _parse_stage1(raw_text, fallback)
        if prediction is None:
            n_fail += 1
            prediction = fallback

        log_rec = {
            "sentence_id": rec["task_id"],
            "stage": 1,
            "prompt_template_version": version,
            "rendered_prompt": rendered_prompt,
            "raw_output_text": raw_text,
            "parsed_prediction": prediction if status == "ok" else None,
            "parse_status": status,
            "parse_error": error,
            "generation_seconds": elapsed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        _append_jsonl(raw_log_path, log_rec)

        predictions_out.append(
            {
                "task_id": rec["task_id"],
                "sentence_idx": rec["sentence_idx"],
                "prediction_int": prediction,
            }
        )
        gold_labels.append(gold_int)
        pred_labels.append(prediction)

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            print(f"  [{i+1}/{len(records)}] parse_fails={n_fail}", flush=True)

    _write_predictions_stage1(pred_path, predictions_out, version)
    print(f"\nStage 1 predictions → {pred_path}")
    print(f"Raw log             → {raw_log_path}")
    print(f"Parse failures: {n_fail}/{len(records)}")

    metrics = compute_metrics_dict(pred_labels, gold_labels)
    print(f"\nStage 1 metrics (split={split}):")
    print(f"  macro_f1 = {metrics['macro_f1']:.4f}")
    print(f"  accuracy = {metrics['accuracy']:.4f}")
    print(f"  per_class_f1 (non-sal, sal) = {metrics['per_class_f1']}")
    print(f"  krippendorff_alpha = {metrics['krippendorff_alpha']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  confusion_matrix:\n{cm}")


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def _load_stage1_predictions(path: str) -> dict[tuple[int, int], int]:
    """Load Stage 1 prediction file. Returns {(task_id, sentence_idx): prediction_int}."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    preds = payload["predictions"] if isinstance(payload, dict) else payload
    return {(p["task_id"], p["sentence_idx"]): p["prediction_int"] for p in preds}


def _build_stage2_inputs(
    use_gold_salience: bool,
    stage1_predictions_path: str | None,
    data_dir: Path,
    split: str,
) -> tuple[list[str], dict[str, list[int]], list[dict]]:
    """Return (sentences, gold_raw, source_records) all index-aligned.

    The i-th element of each returned structure corresponds to the same
    source record. sentence text is never used as a dictionary key.

    gold_raw values are RAW {-1, 0, 1} integers (not encoded indices).
    source_records entries carry at minimum 'task_id' and 'sentence_idx'.
    """
    all_records = _load_raw_records(data_dir, split)
    filtered_records = _filter_records(all_records, split)

    if use_gold_salience:
        # Build aligned arrays directly from filtered raw records.
        # Read dimension labels straight from each record (None → -1).
        # This avoids any dependency on hf_dataset ordering.
        salient_records = [r for r in filtered_records if r.get("salient")]
        sentences: list[str] = []
        gold_raw: dict[str, list[int]] = {dim: [] for dim in DIMS}
        source_recs: list[dict] = []
        for r in salient_records:
            sentences.append(r["sentence_text"])
            source_recs.append(r)
            for dim in DIMS:
                raw_val = r.get(dim)
                if raw_val is None:
                    raw_val = -1
                gold_raw[dim].append(raw_val)
    else:
        if stage1_predictions_path is None:
            raise ValueError(
                "--stage1-predictions is required when --use-gold-salience is not set."
            )
        stage1_preds = _load_stage1_predictions(stage1_predictions_path)
        salient_records = [
            r for r in filtered_records
            if stage1_preds.get((r["task_id"], r["sentence_idx"]), 0) == 1
        ]
        sentences = [r["sentence_text"] for r in salient_records]
        gold_raw = {dim: [] for dim in DIMS}
        source_recs = []
        for r in salient_records:
            source_recs.append(r)
            for dim in DIMS:
                raw_val = r.get(dim)
                if raw_val is None:
                    raw_val = -1
                gold_raw[dim].append(raw_val)

    return sentences, gold_raw, source_recs


def run_stage2(
    cfg: dict,
    split: str,
    limit: int | None,
    model,
    tokenizer,
    device: torch.device,
    data_dir: Path,
    output_dir: Path,
    raw_train: list[dict],
    confirm_final_run: bool,
    allow_test_set: bool,
    use_gold_salience: bool,
    stage1_predictions_path: str | None,
) -> None:
    if split == "test":
        _check_test_protection(allow_test_set)
    if split == "dev":
        _check_dev_protection(2, cfg["prompt_template_version"], confirm_final_run, output_dir)

    raw_log_path = output_dir / f"llama_stage2_{split}_raw.jsonl"
    pred_path = output_dir / f"llama_stage2_{split}_predictions.json"

    sentences, gold_raw, source_recs = _build_stage2_inputs(
        use_gold_salience, stage1_predictions_path, data_dir, split
    )

    if limit is not None:
        sentences = sentences[:limit]
        gold_raw = {dim: gold_raw[dim][:limit] for dim in DIMS}
        source_recs = source_recs[:limit]

    few_shot_block = _build_few_shot_block_stage2(
        cfg, raw_train, cfg["few_shot_example_template"]
    )

    fallback: dict = cfg["parse_failure_fallback"]
    max_new_tokens: int = cfg["max_new_tokens"]
    system_prompt: str = cfg["system_prompt"]
    version: str = cfg["prompt_template_version"]

    predictions_out: list[dict] = []
    pred_raw: dict[str, list[int]] = {dim: [] for dim in DIMS}

    n_fail = 0
    print(f"\n[Stage 2] split={split}  n={len(sentences)}  version={version}")
    print(f"  gold_salience={'gold' if use_gold_salience else 'stage1_predictions'}")

    for i, (sentence, rec) in enumerate(zip(sentences, source_recs)):
        user_msg = _render_stage2_prompt(cfg, few_shot_block, sentence)
        rendered_prompt = _apply_chat_template(tokenizer, system_prompt, user_msg)

        raw_text, elapsed = _generate_one(
            model, tokenizer, rendered_prompt, max_new_tokens, device
        )

        prediction, status, error = _parse_stage2(raw_text, fallback)
        if prediction is None:
            n_fail += 1
            prediction = dict(fallback)

        log_rec = {
            "sentence_id": rec["task_id"],
            "stage": 2,
            "prompt_template_version": version,
            "rendered_prompt": rendered_prompt,
            "raw_output_text": raw_text,
            "parsed_prediction": prediction if status == "ok" else None,
            "parse_status": status,
            "parse_error": error,
            "generation_seconds": elapsed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        _append_jsonl(raw_log_path, log_rec)

        predictions_out.append(
            {
                "task_id": rec["task_id"],
                "sentence_idx": rec["sentence_idx"],
                "predictions": prediction,
            }
        )
        for dim in DIMS:
            pred_raw[dim].append(prediction[dim])

        if (i + 1) % 10 == 0 or (i + 1) == len(sentences):
            print(f"  [{i+1}/{len(sentences)}] parse_fails={n_fail}", flush=True)

    _write_predictions_stage2(pred_path, predictions_out, version)
    print(f"\nStage 2 predictions → {pred_path}")
    print(f"Raw log             → {raw_log_path}")
    print(f"Parse failures: {n_fail}/{len(sentences)}")

    metrics = evaluate_stage2(pred_raw, gold_raw)
    print(f"\nStage 2 metrics (split={split}):")
    for dim in DIMS:
        dm = metrics[dim]
        print(
            f"  {dim:<14} macro_f1={dm['macro_f1']:.4f}  "
            f"collapsed={dm['collapsed']}  "
            f"per_class_f1={[f'{v:.3f}' for v in dm['per_class_f1']]}"
        )
    print(f"  mean_macro_f1  = {metrics['mean_macro_f1']:.4f}")


# ── model loading ─────────────────────────────────────────────────────────────

def _load_model_and_tokenizer(model_name: str):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Run: export HF_TOKEN=$(cat ~/.hf_token)"
        )

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        os.environ["HF_HOME"] = hf_home  # already set; explicit for clarity

    print(f"Loading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Loading model: {model_name}  device={device}  dtype={dtype}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    print("Model loaded.", flush=True)
    return model, tokenizer, device


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLaMA-3.2-3B-Instruct CLT classifier (Stage 1 and Stage 2)."
    )
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument(
        "--stage", type=int, choices=[1, 2], required=True, help="1=salience, 2=CLT dims."
    )
    p.add_argument(
        "--split",
        choices=["dev", "test", "calibration", "smoke"],
        required=True,
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Process only first N sentences."
    )
    p.add_argument(
        "--confirm-final-run",
        action="store_true",
        help="Required when --split dev to acknowledge this is the final evaluation.",
    )
    p.add_argument(
        "--allow-test-set",
        action="store_true",
        help="Required when --split test. Touch test ONCE after model selection.",
    )
    # Stage 2 salience source
    p.add_argument(
        "--use-gold-salience",
        action="store_true",
        default=False,
        help=(
            "Stage 2: evaluate on gold-salient sentences (filtering raw records "
            "where salient=True). DEFAULT for dev evaluation — matches the encoder "
            "cascade evaluation path. Mutually exclusive with --stage1-predictions."
        ),
    )
    p.add_argument(
        "--stage1-predictions",
        default=None,
        help=(
            "Stage 2 production mode: path to Stage 1 predictions JSON. "
            "Run only on sentences predicted salient by Stage 1."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.stage == 2 and args.use_gold_salience and args.stage1_predictions:
        raise ValueError("--use-gold-salience and --stage1-predictions are mutually exclusive.")
    if args.stage == 2 and not args.use_gold_salience and not args.stage1_predictions:
        # Default to gold salience when neither flag given, with a warning.
        print(
            "WARNING: neither --use-gold-salience nor --stage1-predictions was given. "
            "Defaulting to --use-gold-salience for comparability with encoder cascades.",
            flush=True,
        )
        args.use_gold_salience = True

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_root = _REPO_ROOT
    data_dir = repo_root / cfg.get("data_dir", "data/processed")
    output_dir = repo_root / cfg.get("output_dir", "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load train records once (for few-shot example lookup)
    raw_train = _load_raw_records(data_dir, "train")

    model, tokenizer, device = _load_model_and_tokenizer(cfg["model_name"])

    if args.stage == 1:
        run_stage1(
            cfg=cfg,
            split=args.split,
            limit=args.limit,
            model=model,
            tokenizer=tokenizer,
            device=device,
            data_dir=data_dir,
            output_dir=output_dir,
            raw_train=raw_train,
            confirm_final_run=args.confirm_final_run,
            allow_test_set=args.allow_test_set,
        )
    else:
        run_stage2(
            cfg=cfg,
            split=args.split,
            limit=args.limit,
            model=model,
            tokenizer=tokenizer,
            device=device,
            data_dir=data_dir,
            output_dir=output_dir,
            raw_train=raw_train,
            confirm_final_run=args.confirm_final_run,
            allow_test_set=args.allow_test_set,
            use_gold_salience=args.use_gold_salience,
            stage1_predictions_path=args.stage1_predictions,
        )


if __name__ == "__main__":
    main()
