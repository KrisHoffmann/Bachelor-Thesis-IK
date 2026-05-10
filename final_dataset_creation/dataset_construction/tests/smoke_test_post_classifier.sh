#!/usr/bin/env bash
# Smoke test: Stages 7, 8, 9 using mock predictions fixture.
# Depends on smoke_test_pre_classifier.sh having run first (uses its outputs).
# Run from dataset_construction/ directory:
#   bash tests/smoke_test_post_classifier.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Use venv python if available, else fall back to system python
VENV_PYTHON="$(cd "$SCRIPT_DIR/../.." && pwd)/venv/Scripts/python"
if [[ -f "$VENV_PYTHON" ]]; then
    PYTHON="$VENV_PYTHON"
else
    PYTHON="${PYTHON:-python}"
fi
echo "Using python: $PYTHON"

FIXTURES="tests/fixtures"
SMOKE_OUT="tests/smoke_outputs"

PASS=0
FAIL=0

_check_file() {
    local label="$1" file="$2"
    if [[ ! -f "$file" ]]; then
        echo "FAIL [$label]: $file not found"
        FAIL=$((FAIL+1))
    else
        echo "PASS [$label]: $file exists"
        PASS=$((PASS+1))
    fi
}

_check_value() {
    local label="$1" actual="$2" expected="$3"
    if [[ "$actual" == "$expected" ]]; then
        echo "PASS [$label]: $actual (expected $expected)"
        PASS=$((PASS+1))
    else
        echo "FAIL [$label]: got $actual (expected $expected)"
        FAIL=$((FAIL+1))
    fi
}

echo "========================================"
echo " Smoke Test: Post-Classifier Stages 7-9"
echo "========================================"
echo ""

if [[ ! -f "$SMOKE_OUT/comments_filtered.jsonl" ]]; then
    echo "Pre-classifier smoke outputs missing. Run smoke_test_pre_classifier.sh first."
    exit 1
fi

# Generate mock_sentence_predictions.parquet if not already present
if [[ ! -f "$FIXTURES/mock_sentence_predictions.parquet" ]]; then
    echo "Generating mock_sentence_predictions.parquet via setup_fixtures.py..."
    "$PYTHON" tests/setup_fixtures.py
fi

# --- Stage 7: aggregate_indices ---
echo "[STAGE 7] aggregate_indices"
"$PYTHON" scripts/aggregate_indices.py \
    --input "$FIXTURES/mock_sentence_predictions.parquet" \
    --output "$SMOKE_OUT/comment_indices.parquet" \
    --counts-json "$SMOKE_OUT/stage7_counts.json"
_check_file "Stage7:comment_indices" "$SMOKE_OUT/comment_indices.parquet"

# c6 is all-NaN → 4 survivors from 5
AFTER_ZS=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/stage7_counts.json') as f:
    d = json.load(f)
print(d.get('after_remove_zero_salience', -1))
")
_check_value "Stage7:zero_salience_drop" "$AFTER_ZS" "4"

COLS_OK=$("$PYTHON" -c "
import pandas as pd
df = pd.read_parquet('$SMOKE_OUT/comment_indices.parquet')
expected = ['comment_id','A_T','A_S','A_Soc','A_H','num_salient_T','num_salient_S','num_salient_Soc','num_salient_H']
missing = [c for c in expected if c not in df.columns]
print(len(missing), missing)
")
_check_value "Stage7:columns_present" "$COLS_OK" "0 []"

# --- Stage 8: merge_final_dataset ---
echo ""
echo "[STAGE 8] merge_final_dataset (no author features — Stage 4 removed)"
"$PYTHON" scripts/merge_final_dataset.py \
    --comments-input "$SMOKE_OUT/comments_filtered.jsonl" \
    --indices-input "$SMOKE_OUT/comment_indices.parquet" \
    --controls-input "$SMOKE_OUT/controls_final.parquet" \
    --output "$SMOKE_OUT/analysis_dataset_pre_matching.parquet" \
    --counts-json "$SMOKE_OUT/stage8_counts.json"
_check_file "Stage8:pre_matching_dataset" "$SMOKE_OUT/analysis_dataset_pre_matching.parquet"

MERGE_ROWS=$("$PYTHON" -c "
import pandas as pd
df = pd.read_parquet('$SMOKE_OUT/analysis_dataset_pre_matching.parquet')
print(len(df))
")
if [[ "$MERGE_ROWS" -gt 0 ]]; then
    echo "PASS [Stage8:row_count]: pre-matching dataset has $MERGE_ROWS rows"
    PASS=$((PASS+1))
else
    echo "FAIL [Stage8:row_count]: pre-matching dataset is empty"
    FAIL=$((FAIL+1))
fi

# --- Stage 9: match_case_control (ratio 3) ---
echo ""
echo "[STAGE 9a] match_case_control --ratio 3"
"$PYTHON" scripts/match_case_control.py \
    --input "$SMOKE_OUT/analysis_dataset_pre_matching.parquet" \
    --output "$SMOKE_OUT/matched_dataset_ratio3.parquet" \
    --ratio 3 \
    --seed 42 \
    --counts-json "$SMOKE_OUT/matching_diagnostics_ratio3.json"
_check_file "Stage9a:matched_ratio3" "$SMOKE_OUT/matched_dataset_ratio3.parquet"

STRATA=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/matching_diagnostics_ratio3.json') as f:
    d = json.load(f)
print(d.get('total_strata', 0))
")
if [[ "$STRATA" -gt 0 ]]; then
    echo "PASS [Stage9a:strata_count]: $STRATA strata created"
    PASS=$((PASS+1))
else
    echo "FAIL [Stage9a:strata_count]: 0 strata"
    FAIL=$((FAIL+1))
fi

STRAT_COL=$("$PYTHON" -c "
import pandas as pd
df = pd.read_parquet('$SMOKE_OUT/matched_dataset_ratio3.parquet')
print('stratum_id' in df.columns)
")
_check_value "Stage9a:stratum_id_col" "$STRAT_COL" "True"

# --- Stage 9: match_case_control (ratio 5) ---
echo ""
echo "[STAGE 9b] match_case_control --ratio 5"
"$PYTHON" scripts/match_case_control.py \
    --input "$SMOKE_OUT/analysis_dataset_pre_matching.parquet" \
    --output "$SMOKE_OUT/matched_dataset_ratio5.parquet" \
    --ratio 5 \
    --seed 42 \
    --counts-json "$SMOKE_OUT/matching_diagnostics_ratio5.json"
_check_file "Stage9b:matched_ratio5" "$SMOKE_OUT/matched_dataset_ratio5.parquet"

echo ""
echo "========================================"
echo " SMOKE TEST RESULTS: $PASS passed, $FAIL failed"
echo "========================================"
if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
