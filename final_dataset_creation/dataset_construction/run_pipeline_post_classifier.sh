#!/usr/bin/env bash
# Stages 7, 8, 9 — BLOCKED until Stage 6 (classifier) is complete.
# Run once with --ratio 3 (primary) and once with --ratio 5 (robustness).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f outputs/sentence_predictions.parquet ]]; then
    echo "BLOCKED — Stage 6 must complete first."
    echo "Run run_clt_classifier.py to produce outputs/sentence_predictions.parquet"
    exit 1
fi

echo "========================================"
echo " Post-Classifier Pipeline (7, 8, 9)"
echo "========================================"
echo ""

echo "[STAGE 7] starting — aggregate_indices.py"
python scripts/aggregate_indices.py \
    --input outputs/sentence_predictions.parquet \
    --output outputs/comment_indices.parquet \
    --counts-json consort/stage7_counts.json
echo "[STAGE 7] complete"

echo ""
echo "[STAGE 8] starting — merge_final_dataset.py"
python scripts/merge_final_dataset.py \
    --comments-input outputs/comments_filtered.jsonl \
    --indices-input outputs/comment_indices.parquet \
    --controls-input outputs/controls_final.parquet \
    --author-features-input outputs/author_features.parquet \
    --output outputs/analysis_dataset_pre_matching.parquet \
    --counts-json consort/stage8_counts.json
echo "[STAGE 8] complete"

echo ""
echo "[STAGE 9a] starting — match_case_control.py --ratio 3 (primary)"
python scripts/match_case_control.py \
    --input outputs/analysis_dataset_pre_matching.parquet \
    --output outputs/matched_dataset_ratio3.parquet \
    --ratio 3 \
    --seed 42 \
    --counts-json consort/matching_diagnostics_ratio3.json
echo "[STAGE 9a] complete"

echo ""
echo "[STAGE 9b] starting — match_case_control.py --ratio 5 (robustness)"
python scripts/match_case_control.py \
    --input outputs/analysis_dataset_pre_matching.parquet \
    --output outputs/matched_dataset_ratio5.parquet \
    --ratio 5 \
    --seed 42 \
    --counts-json consort/matching_diagnostics_ratio5.json
echo "[STAGE 9b] complete"

echo ""
echo "========================================"
echo " Post-classifier pipeline complete."
echo " Primary dataset:    outputs/matched_dataset_ratio3.parquet"
echo " Robustness dataset: outputs/matched_dataset_ratio5.parquet"
echo "========================================"
