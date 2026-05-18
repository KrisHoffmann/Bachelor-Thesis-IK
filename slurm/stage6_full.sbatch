#!/bin/bash
#
# Stage 6 — BERT cascade inference, full corpus (~4,040,317 sentences).
#
# GPU partition: gpumedium
#   Rationale: BERT-base is ~440 MB; the two models never coexist in VRAM
#   (Stage 1 is loaded, run, then freed before Stage 2 loads). Peak VRAM
#   is ~2 GB at batch size 32, well within a V100 (16 GB) or any A100 slice.
#   gpumedium is used rather than pinning to a specific GPU type so the
#   scheduler can assign whatever is free first (L40S, A100, V100). Pinning
#   to e.g. --gres=gpu:a100:1 would reduce queue time variability but at the
#   cost of a longer wait when A100s are busy. Given the 12h wall-clock budget
#   and that BERT-base is not memory-constrained on any current Hábrók GPU,
#   permissive scheduling is preferred.
#
# Memory: 16G
#   The chunk loop holds up to --chunk-size (50 000) input records in memory
#   plus the merged output buffer before each fsync. Smoke test peaked at ~3 GB
#   on 1 000 sentences; 50k-sentence chunks are 50× larger in the worst case.
#   16 GB gives 5× headroom over that extrapolation without being wasteful.
#
# Usage:
#   cd /scratch/s4546407/clt_thesis/Bachelor-Thesis-IK
#   sbatch slurm/stage6_full.sbatch
#
#SBATCH --job-name=stage6_full
#SBATCH --output=slurm/logs/stage6_full_%j.out
#SBATCH --error=slurm/logs/stage6_full_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

set -euo pipefail

REPO_ROOT="/scratch/s4546407/clt_thesis/Bachelor-Thesis-IK"

echo "=== Stage 6 full corpus — start $(date) ==="
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

DS="final_dataset_creation/dataset_construction"

# Verify required files exist.
if [ ! -f "${DS}/outputs/sentences_for_classifier.jsonl" ]; then
    echo "ERROR: ${DS}/outputs/sentences_for_classifier.jsonl not found"
    echo "Run Stage 5.5 first: python scripts/stage5_5_sentence_export.py"
    exit 1
fi

# Verify input row count is in the expected ~4M range.
INPUT_ROWS=$(wc -l < "${DS}/outputs/sentences_for_classifier.jsonl")
echo "Input rows: ${INPUT_ROWS}"
if [ "${INPUT_ROWS}" -lt 4000000 ] || [ "${INPUT_ROWS}" -gt 4100000 ]; then
    echo "WARNING: input row count outside expected 4M-4.1M range. Continuing anyway."
fi

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

# Guard against silently overwriting or appending to an existing output file.
# To resume a crashed run: add --resume to the python invocation below and resubmit.
# To start fresh:          rm "${OUT_FILE}" && sbatch slurm/stage6_full.sbatch
OUT_FILE="${DS}/outputs/stage6_full_predictions.jsonl"
if [ -s "${OUT_FILE}" ]; then
    echo "ERROR: ${OUT_FILE} already exists and is non-empty."
    echo "  - To resume: edit sbatch to add --resume, then resubmit."
    echo "  - To start fresh: rm ${OUT_FILE} && sbatch slurm/stage6_full.sbatch"
    exit 1
fi

python -u scripts/stage6_inference.py \
    --input       "${DS}/outputs/sentences_for_classifier.jsonl" \
    --output      "${OUT_FILE}" \
    --stage1-ckpt "${STAGE1_CKPT}" \
    --stage2-ckpt "${STAGE2_CKPT}" \
    --batch-size  32 \
    --chunk-size  50000 \
    --log-every   100
    # --resume

echo ""
echo "=== Stage 6 full corpus — end $(date) ==="
echo "Output   : ${OUT_FILE}"
echo "Row count: $(wc -l < ${OUT_FILE})"
echo ""
echo "Quick schema check (first record keys):"
python -c "
import json, sys
with open('${OUT_FILE}') as f:
    rec = json.loads(f.readline())
print('keys:', list(rec.keys()))
print('salience_pred:', rec.get('salience_pred'))
print('temporal_pred:', rec.get('temporal_pred'))
print('social_pred   :', rec.get('social_pred'))
print('salience_logits length:', len(rec.get('salience_logits', [])))
"
echo ""
echo "Full-corpus distribution summary:"
python -c "
import json
from collections import Counter
out = '${OUT_FILE}'
total = 0
salience = Counter()
dim_dists = {'temporal': Counter(), 'spatial': Counter(), 'social': Counter(), 'hypothetical': Counter()}
with open(out) as f:
    for line in f:
        r = json.loads(line)
        total += 1
        salience[r['salience_pred']] += 1
        for d in dim_dists:
            v = r[f'{d}_pred']
            if v is not None:
                dim_dists[d][v] += 1
print(f'Total rows: {total}')
print(f'Salience  : {dict(salience)}')
for d, c in dim_dists.items():
    print(f'{d:14s}: {dict(c)}')
"
