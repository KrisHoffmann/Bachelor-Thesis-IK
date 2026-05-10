#!/usr/bin/env bash
# Stages 1, 1.5 (validate_delta_labels), 2 — fast, I/O-bound stages.
# Stage 4 (author features) has been removed from the pipeline.
# Stage 3 runs on Hábrók via slurm/stage3_segmentation.sbatch.
# Stage 5/5.5 are in run_pipeline_post_segmentation.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Configure paths ---
THREADS_PATH=${THREADS_PATH:-/mnt/c/cmv_data/3778298/threads.jsonl.bz2}
PAIRS_PATH=${PAIRS_PATH:-/mnt/c/cmv_data/3778298/pairs.jsonl.bz2}
THREAD_IDS=${THREAD_IDS:-../outputs/topic_filter/cmv_economics_thread_ids.txt}

echo "========================================"
echo " Phase B — Pre-Classifier Pipeline"
echo " Threads:        $THREADS_PATH"
echo " Pairs (QA):     $PAIRS_PATH"
echo " Thread IDs:     $THREAD_IDS"
echo "========================================"

# --- Stage 1: Flatten threads, derive delta labels (corrected R→A→B labeller) ---
echo ""
echo "[STAGE 1] starting — pull_comments.py"
python scripts/pull_comments.py \
    --threads-input "$THREADS_PATH" \
    --thread-ids-input "$THREAD_IDS" \
    --output outputs/comments_raw.jsonl \
    --counts-json consort/stage1_counts.json \
    --emit-bots
echo "[STAGE 1] complete"

python -c "
import json
with open('consort/stage1_counts.json') as f:
    d = json.load(f)
print(f'  Threads matched: {d[\"threads_matched_in_corpus\"]}')
print(f'  Raw comments total: {d[\"raw_comments_total\"]}')
print(f'  Raw delta-awarded (recipients): {d[\"raw_comments_delta_awarded\"]}')
print(f'  Bot comments: {d[\"raw_comments_bot\"]}')
"

# --- Stage 1.5: Validate delta labeller against pairs ground truth ---
echo ""
echo "[STAGE 1.5] starting — validate_delta_labels.py"
python scripts/validate_delta_labels.py \
    --pairs-input "$PAIRS_PATH" \
    --threads-input "$THREADS_PATH" \
    --output outputs/validation/delta_label_validation.json \
    --log-level INFO

RECALL=$(python -c "
import json
with open('outputs/validation/delta_label_validation.json') as f:
    d = json.load(f)
print(d.get('recall', 0))
")
echo "  Recall: $RECALL"
if python -c "import sys; sys.exit(0 if float('$RECALL') >= 1.0 else 1)"; then
    echo "[STAGE 1.5] Recall == 1.0 — labeller validated"
else
    echo "ERROR: Recall < 1.0. Debug delta labeller before proceeding."
    echo "       See outputs/validation/delta_label_validation.json for false negatives."
    exit 1
fi

# --- Stage 2: Filter comments ---
echo ""
echo "[STAGE 2] starting — filter_comments.py"
python scripts/filter_comments.py \
    --input outputs/comments_raw.jsonl \
    --output outputs/comments_filtered.jsonl \
    --counts-json consort/stage2_counts.json
echo "[STAGE 2] complete"

POST2_COUNT=$(python -c "
import json
with open('consort/stage2_counts.json') as f:
    d = json.load(f)
print(d.get('after_quote_only', '?'))
")
echo ""
echo "Post-Stage-2 comment count: $POST2_COUNT"
echo ""

echo "========================================"
echo " Pre-classifier stages done."
echo ""
echo " Stage 3 runs on Hábrók GPU:"
echo "   1. scp outputs/comments_filtered.jsonl to Hábrók"
echo "   2. sbatch slurm/stage3_segmentation.sbatch"
echo "   3. scp outputs/comments_segmented.jsonl back"
echo "   4. run run_pipeline_post_segmentation.sh"
echo "========================================"
