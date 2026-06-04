#!/bin/bash
#
# Stage 6 — CLT inference on OP selftexts (~11,142 threads, ~220k sentences).
#
# Wall time: 1 h is generous; expect 5–15 min on any modern GPU.
# Memory: 8 GB (same as smoke test; OP corpus is ~20x smaller than full comments).
#
# Usage (from Habrok scratch):
#   cd /scratch/s4546407/clt_thesis/Bachelor-Thesis-IK
#   sbatch slurm/stage6_op_selftext.sbatch
#
#SBATCH --job-name=stage6_op_selftext
#SBATCH --output=slurm/logs/stage6_op_selftext_%j.out
#SBATCH --error=slurm/logs/stage6_op_selftext_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

set -euo pipefail

REPO_ROOT="/scratch/s4546407/clt_thesis/Bachelor-Thesis-IK"

echo "=== Stage 6 OP selftext inference — start $(date) ==="
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Repo    : ${REPO_ROOT}"

cd "${REPO_ROOT}"
mkdir -p slurm/logs

module purge
module load Python/3.11.5-GCCcore-13.2.0
source /scratch/s4546407/venvs/llama/bin/activate

export HF_HOME="/scratch/s4546407/hf_cache"
mkdir -p "${HF_HOME}"

echo "Python  : $(which python)"
python -c "import torch; print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}')"
python -c "import transformers; print(f'transformers {transformers.__version__}')"

# ---------------------------------------------------------------------------
# Step 1 — Extract OP selftexts from the local bz2 dump
# ---------------------------------------------------------------------------
THREADS_BZ2="/mnt/c/cmv_data/3778298/threads.jsonl.bz2"
OUT_SENTENCES="final_dataset_creation/outputs/op_selftexts/op_sentences_for_classifier.jsonl"

# Note: on Habrok the bz2 will be on scratch, not /mnt/c — update path if needed.
# If you copied it to scratch:
#   THREADS_BZ2="/scratch/s4546407/clt_thesis/threads.jsonl.bz2"

if [ ! -f "${OUT_SENTENCES}" ]; then
    echo "Extracting OP sentences from ${THREADS_BZ2} ..."
    python scripts/extract_op_selftexts.py
    echo "Extraction complete. $(wc -l < ${OUT_SENTENCES}) sentences."
else
    echo "OP sentences already extracted ($(wc -l < ${OUT_SENTENCES}) rows). Skipping extraction."
fi

# ---------------------------------------------------------------------------
# Step 2 — Run CLT inference
# ---------------------------------------------------------------------------
STAGE1_CKPT="peer_handoff/models/stage1_model"
STAGE2_CKPT="peer_handoff/models/stage2_best_model.pt"
OUT_FILE="final_dataset_creation/outputs/op_selftexts/op_predictions.jsonl"

if [ ! -d "${STAGE1_CKPT}" ]; then
    echo "ERROR: Stage 1 checkpoint not found at ${STAGE1_CKPT}"; exit 1
fi
if [ ! -f "${STAGE2_CKPT}" ]; then
    echo "ERROR: Stage 2 checkpoint not found at ${STAGE2_CKPT}"; exit 1
fi

if [ -s "${OUT_FILE}" ]; then
    echo "ERROR: ${OUT_FILE} already exists and is non-empty."
    echo "  To resume: add --resume to the python invocation below and resubmit."
    echo "  To start fresh: rm ${OUT_FILE} && sbatch slurm/stage6_op_selftext.sbatch"
    exit 1
fi

echo ""
echo "Running CLT inference on $(wc -l < ${OUT_SENTENCES}) OP sentences ..."
python -u scripts/stage6_inference.py \
    --input       "${OUT_SENTENCES}" \
    --output      "${OUT_FILE}" \
    --stage1-ckpt "${STAGE1_CKPT}" \
    --stage2-ckpt "${STAGE2_CKPT}" \
    --batch-size  32 \
    --chunk-size  50000 \
    --log-every   50
    # --resume   # uncomment to continue a partial run

echo ""
echo "=== Stage 6 OP selftext inference — end $(date) ==="
echo "Output   : ${OUT_FILE}"
echo "Row count: $(wc -l < ${OUT_FILE})"
echo ""
echo "Quick schema check (first record):"
python -c "
import json
with open('${OUT_FILE}') as f:
    rec = json.loads(f.readline())
print('keys:', list(rec.keys()))
print('salience_pred:', rec.get('salience_pred'))
print('social_pred  :', rec.get('social_pred'))
"
echo ""
echo "Next step: run scripts/aggregate_op_indices.py to produce per-thread A_Soc_OP / A_H_OP"
