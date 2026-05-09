#!/bin/bash
# Submit this job and confirm it succeeds before submitting habrok_llama_full.slurm.
#
# Purpose: cheap end-to-end integration test on 5 sentences.
# Exercises: model load + HF auth, chat template, generation, parser,
#            JSONL logging, prediction-file schema, metric-function call.
# Metric values on n=5 are meaningless — we care only that every code path
# runs without error and exits 0.
#
# Usage:
#   sbatch scripts/habrok_llama_smoke.slurm
#
#SBATCH --job-name=llama_smoke
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/home5/%u/clt_thesis_logs/llama_smoke_%j.out
#SBATCH --error=/home5/%u/clt_thesis_logs/llama_smoke_%j.err

set -euo pipefail

# ── logging dir ──────────────────────────────────────────────────────────────
mkdir -p "/home5/${USER}/clt_thesis_logs"

echo "=== LLaMA smoke test ==="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Date:    $(date -u)"
echo "User:    ${USER}"
echo ""

# ── environment ───────────────────────────────────────────────────────────────
module purge
module load Python/3.11.5-GCCcore-13.2.0

VENV_PATH="/scratch/${USER}/venvs/llama"
if [[ ! -d "${VENV_PATH}" ]]; then
    echo "ERROR: venv not found at ${VENV_PATH}" >&2
    echo "Create it with: python -m venv ${VENV_PATH} && source ${VENV_PATH}/bin/activate && pip install --upgrade pip && pip install -r ${REPO_ROOT}/requirements.txt && pip install huggingface_hub sentencepiece" >&2
    exit 1
fi
source "${VENV_PATH}/bin/activate"

export HF_HOME="/scratch/${USER}/hf_cache"
mkdir -p "${HF_HOME}"

HF_TOKEN_FILE="${HOME}/.hf_token"
if [[ ! -f "${HF_TOKEN_FILE}" ]]; then
    echo "ERROR: HF token file not found at ${HF_TOKEN_FILE}" >&2
    exit 1
fi
export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"

# ── repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT="/scratch/${USER}/clt_thesis/Bachelor-Thesis-IK"
if [[ ! -d "${REPO_ROOT}" ]]; then
    echo "ERROR: repo not found at ${REPO_ROOT}" >&2
    exit 1
fi
cd "${REPO_ROOT}"

# ── output dir ────────────────────────────────────────────────────────────────
mkdir -p "/scratch/${USER}/clt_outputs"
# Symlink outputs/ into scratch so JSONL logs land there (scratch has space)
if [[ ! -L outputs ]] && [[ ! -d outputs ]]; then
    ln -s "/scratch/${USER}/clt_outputs" outputs
fi
mkdir -p outputs

echo "=== Stage 1 smoke (5 sentences) ==="
python scripts/llama_inference.py \
    --config configs/llama_stage1.yaml \
    --stage 1 \
    --split smoke
echo "Stage 1 smoke exit code: $?"

echo ""
echo "=== Stage 2 smoke (5 sentences, --use-gold-salience) ==="
# Force --use-gold-salience so Stage 2 is exercised even if Stage 1 predicts
# all sentences as non-salient on this tiny 5-sentence slice.
python scripts/llama_inference.py \
    --config configs/llama_stage2.yaml \
    --stage 2 \
    --split smoke \
    --use-gold-salience
echo "Stage 2 smoke exit code: $?"

echo ""
echo "=== Smoke test COMPLETE ==="
echo "Check the following files exist and are non-empty:"
echo "  outputs/llama_stage1_smoke_raw.jsonl"
echo "  outputs/llama_stage1_smoke_predictions.json"
echo "  outputs/llama_stage2_smoke_raw.jsonl"
echo "  outputs/llama_stage2_smoke_predictions.json"
echo ""
echo "Inspect a few raw_output_text values to confirm the model is generating"
echo "real text (not empty strings or repeated tokens)."
echo ""
echo "When satisfied, submit the full job:"
echo "  sbatch scripts/habrok_llama_full.slurm"
