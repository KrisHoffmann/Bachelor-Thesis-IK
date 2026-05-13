#!/bin/bash
#
# Stage 6 — BERT cascade inference, smoke test (first 1000 sentences).
#
# GPU partition: gpushort
#   Rationale: BERT-base is ~440 MB; the two models never coexist in VRAM
#   (Stage 1 is loaded, run, then freed before Stage 2 loads). Peak VRAM
#   is ~2 GB at batch size 32, well within a V100 (16 GB) or any A100 slice.
#   gpushort is the right partition for a sub-5-minute test job — shorter
#   queue time than the general `gpu` partition and no wasted budget.
#   If gpushort is unavailable on your allocation, substitute `gpu` and
#   reduce --time to 00:30:00.
#
# Usage:
#   cd /scratch/s4546407/clt_thesis/Bachelor-Thesis-IK
#   sbatch slurm/stage6_smoke.sbatch
#
#SBATCH --job-name=stage6_smoke
#SBATCH --output=slurm/logs/stage6_smoke_%j.out
#SBATCH --error=slurm/logs/stage6_smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

set -euo pipefail

REPO_ROOT="/scratch/s4546407/clt_thesis/Bachelor-Thesis-IK"

echo "=== Stage 6 smoke test — start $(date) ==="
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Repo    : ${REPO_ROOT}"

cd "${REPO_ROOT}"
mkdir -p slurm/logs

module purge
module load Python/3.11.5-GCCcore-13.2.0
source /scratch/s4546407/venvs/phase_b/bin/activate

export HF_HOME="/scratch/s4546407/hf_cache"
mkdir -p "${HF_HOME}"

echo "Python  : $(which python)"
python -c "import torch; print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}')"
python -c "import transformers; print(f'transformers {transformers.__version__}')"

DS="final_dataset_creation/dataset_construction"

# Verify required files
if [ ! -f "${DS}/outputs/sentences_for_classifier.jsonl" ]; then
    echo "ERROR: ${DS}/outputs/sentences_for_classifier.jsonl not found"
    echo "Run Stage 5.5 first: python scripts/stage5_5_sentence_export.py"
    exit 1
fi
echo "Input rows (first 5): $(head -n 5 ${DS}/outputs/sentences_for_classifier.jsonl | wc -l)"

STAGE1_CKPT="peer_handoff/models/stage1_model"
STAGE2_CKPT="peer_handoff/models/stage2_best_model.pt"

if [ ! -d "${STAGE1_CKPT}" ]; then
    echo "ERROR: Stage 1 checkpoint not found at ${STAGE1_CKPT}"
    exit 1
fi
if [ ! -f "${STAGE2_CKPT}" ]; then
    echo "ERROR: Stage 2 checkpoint not found at ${STAGE2_CKPT}"
    exit 1
fi

mkdir -p "${DS}/outputs"

python -u scripts/stage6_inference.py \
    --input   "${DS}/outputs/sentences_for_classifier.jsonl" \
    --output  "${DS}/outputs/stage6_smoke_predictions.jsonl" \
    --stage1-ckpt "${STAGE1_CKPT}" \
    --stage2-ckpt "${STAGE2_CKPT}" \
    --max-sentences 1000 \
    --batch-size 32 \
    --log-every 10

echo ""
echo "=== Stage 6 smoke test — end $(date) ==="
echo "Output: ${DS}/outputs/stage6_smoke_predictions.jsonl"
echo "Row count: $(wc -l < ${DS}/outputs/stage6_smoke_predictions.jsonl)"
echo ""
echo "Quick schema check (first record keys):"
python -c "
import json, sys
with open('${DS}/outputs/stage6_smoke_predictions.jsonl') as f:
    rec = json.loads(f.readline())
print('keys:', list(rec.keys()))
print('salience_pred:', rec.get('salience_pred'))
print('temporal_pred:', rec.get('temporal_pred'))
print('social_pred   :', rec.get('social_pred'))
print('salience_logits length:', len(rec.get('salience_logits', [])))
"
