#!/bin/bash
# Full LLaMA evaluation job — submit ONLY after habrok_llama_smoke.slurm succeeds.
#
# Run order (each as a separate Python invocation so one failure is isolated):
#   (a) Stage 1 calibration    — 30 sentences, no guard flags required
#   (b) Stage 1 full dev       — 176 sentences, gated on --confirm-final-run
#   (c) Stage 2 calibration    — 30 sentences, --use-gold-salience
#   (d) Stage 2 full dev       — 176 salient sentences, --use-gold-salience
#                                gated on --confirm-final-run
#
# Calibration discipline
#   Runs (a) and (c) first. Inspect outputs/llama_stage*_calibration_raw.jsonl
#   before deciding whether to unlock (b) and (d).
#
#   To run ONLY calibration (recommended first pass):
#     Set CONFIRM_FINAL_RUN=0 (the default) and submit.
#     The dev steps will skip with a clear message.
#
#   To run the final full-dev evaluation:
#     Set CONFIRM_FINAL_RUN=1 and resubmit.
#     This will fail (RuntimeError) if prompt_template_version is unchanged
#     from a previous full-dev run — bump the version in the config first.
#
# Set this to 1 only when you are ready to commit the full dev evaluation.
#SBATCH --job-name=llama_full
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home5/%u/clt_thesis_logs/llama_full_%j.out
#SBATCH --error=/home5/%u/clt_thesis_logs/llama_full_%j.err

# First executable line — SLURM has finished reading directives by here.
CONFIRM_FINAL_RUN="${CONFIRM_FINAL_RUN:-0}"

set -uo pipefail
# NOTE: 'set -e' is intentionally omitted so that a failure in one step
# does not prevent the remaining steps from running. Each step's exit code
# is checked explicitly and printed.

# ── logging dir ──────────────────────────────────────────────────────────────
mkdir -p "/home5/${USER}/clt_thesis_logs"

echo "=== LLaMA full evaluation ==="
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Node:             $(hostname)"
echo "Date:             $(date -u)"
echo "User:             ${USER}"
echo "CONFIRM_FINAL_RUN=${CONFIRM_FINAL_RUN}"
echo ""

# ── environment ───────────────────────────────────────────────────────────────
module purge
module load Python/3.11.5-GCCcore-13.2.0

VENV_PATH="/scratch/${USER}/venvs/llama"
if [[ ! -d "${VENV_PATH}" ]]; then
    echo "ERROR: venv not found at ${VENV_PATH}" >&2
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
if [[ ! -L outputs ]] && [[ ! -d outputs ]]; then
    ln -s "/scratch/${USER}/clt_outputs" outputs
fi
mkdir -p outputs

# ── helper to run a step and report exit code ─────────────────────────────────
OVERALL_EXIT=0
run_step() {
    local step_name="$1"
    shift
    echo ""
    echo "━━━ ${step_name} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Command: python $*"
    echo "Started: $(date -u)"
    python "$@"
    local rc=$?
    echo "Finished: $(date -u)  exit_code=${rc}"
    if [[ "${rc}" -ne 0 ]]; then
        echo "WARNING: ${step_name} exited with code ${rc}" >&2
        OVERALL_EXIT="${rc}"
    fi
    return "${rc}"
}

# ── (a) Stage 1 calibration ───────────────────────────────────────────────────
run_step "Stage 1 calibration (30 sentences)" \
    scripts/llama_inference.py \
    --config configs/llama_stage1.yaml \
    --stage 1 \
    --split calibration
S1_CAL_RC=$?

# ── (b) Stage 1 full dev ──────────────────────────────────────────────────────
if [[ "${CONFIRM_FINAL_RUN}" == "1" ]]; then
    run_step "Stage 1 full dev (176 sentences — FINAL)" \
        scripts/llama_inference.py \
        --config configs/llama_stage1.yaml \
        --stage 1 \
        --split dev \
        --confirm-final-run
    S1_DEV_RC=$?
else
    echo ""
    echo "━━━ Stage 1 full dev — SKIPPED (CONFIRM_FINAL_RUN=0) ━━━━━━━━━━━━━"
    echo "To run: resubmit with CONFIRM_FINAL_RUN=1"
    echo "  CONFIRM_FINAL_RUN=1 sbatch scripts/habrok_llama_full.slurm"
    S1_DEV_RC=0
fi

# ── (c) Stage 2 calibration ───────────────────────────────────────────────────
run_step "Stage 2 calibration (30 sentences, --use-gold-salience)" \
    scripts/llama_inference.py \
    --config configs/llama_stage2.yaml \
    --stage 2 \
    --split calibration \
    --use-gold-salience
S2_CAL_RC=$?

# ── (d) Stage 2 full dev ──────────────────────────────────────────────────────
if [[ "${CONFIRM_FINAL_RUN}" == "1" ]]; then
    run_step "Stage 2 full dev (gold-salient sentences — FINAL)" \
        scripts/llama_inference.py \
        --config configs/llama_stage2.yaml \
        --stage 2 \
        --split dev \
        --use-gold-salience \
        --confirm-final-run
    S2_DEV_RC=$?
else
    echo ""
    echo "━━━ Stage 2 full dev — SKIPPED (CONFIRM_FINAL_RUN=0) ━━━━━━━━━━━━━"
    echo "To run: resubmit with CONFIRM_FINAL_RUN=1"
    S2_DEV_RC=0
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Step summary ==="
echo "  (a) Stage 1 calibration : exit ${S1_CAL_RC}"
if [[ "${CONFIRM_FINAL_RUN}" == "1" ]]; then
    echo "  (b) Stage 1 full dev    : exit ${S1_DEV_RC}"
else
    echo "  (b) Stage 1 full dev    : SKIPPED"
fi
echo "  (c) Stage 2 calibration : exit ${S2_CAL_RC}"
if [[ "${CONFIRM_FINAL_RUN}" == "1" ]]; then
    echo "  (d) Stage 2 full dev    : exit ${S2_DEV_RC}"
else
    echo "  (d) Stage 2 full dev    : SKIPPED"
fi
echo ""
echo "Output files (check all exist and are non-empty):"
echo "  outputs/llama_stage1_calibration_raw.jsonl"
echo "  outputs/llama_stage1_calibration_predictions.json"
echo "  outputs/llama_stage2_calibration_raw.jsonl"
echo "  outputs/llama_stage2_calibration_predictions.json"
if [[ "${CONFIRM_FINAL_RUN}" == "1" ]]; then
    echo "  outputs/llama_stage1_dev_raw.jsonl"
    echo "  outputs/llama_stage1_dev_predictions.json"
    echo "  outputs/llama_stage2_dev_raw.jsonl"
    echo "  outputs/llama_stage2_dev_predictions.json"
fi
echo ""
echo "Job finished: $(date -u)  overall_exit=${OVERALL_EXIT}"
exit "${OVERALL_EXIT}"
