#!/usr/bin/env bash
# Stages 5 and 5.5 — runs after Stage 3 segmentation is complete.
# Stage 5: compute linguistic controls (two-pass: raw → JB-test → final)
# Stage 5.5: explode to sentence-level JSONL for classifier input
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEVICE=${DEVICE:-cpu}

if [[ ! -f outputs/comments_segmented.jsonl ]]; then
    echo "ERROR: outputs/comments_segmented.jsonl not found."
    echo "Run run_stage3_segmentation.sh first."
    exit 1
fi

echo "========================================"
echo " Post-Segmentation Pipeline (5, 5.5)"
echo "========================================"
echo ""

echo "[STAGE 5] starting — compute_controls.py (two-pass)"
python scripts/compute_controls.py \
    --input outputs/comments_segmented.jsonl \
    --raw-output outputs/controls_raw.parquet \
    --final-output outputs/controls_final.parquet \
    --hedge-file data/hedges_hyland_2005.txt \
    --counts-json consort/stage5_counts.json \
    --device "$DEVICE"
echo "[STAGE 5] complete"

echo ""
echo "[STAGE 5.5] starting — prepare_for_classifier.py"
python scripts/prepare_for_classifier.py \
    --input outputs/comments_segmented.jsonl \
    --output outputs/sentences_for_classifier.jsonl \
    --counts-json consort/stage55_counts.json
echo "[STAGE 5.5] complete"

SENT_COUNT=$(python -c "
import json
with open('consort/stage55_counts.json') as f:
    d = json.load(f)
print(d.get('sentences_for_classifier', '?'))
")

echo ""
echo "========================================"
echo " READY FOR STAGE 6 (classifier shootout)."
echo " Total sentences: $SENT_COUNT"
echo " Outputs in outputs/sentences_for_classifier.jsonl"
echo "========================================"
