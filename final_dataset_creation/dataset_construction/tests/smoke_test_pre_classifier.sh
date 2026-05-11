#!/usr/bin/env bash
# Smoke test: Stages 1, 1.5 (validate_delta_labels), 2, 3, 5, 5.5 end-to-end.
# (Stage 4 has been removed from the pipeline.)
# Run from dataset_construction/ directory:
#   bash tests/smoke_test_pre_classifier.sh
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
rm -rf "$SMOKE_OUT"
mkdir -p "$SMOKE_OUT"

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

_check_jsonl_count() {
    local label="$1" file="$2" expected="$3"
    if [[ ! -f "$file" ]]; then
        echo "FAIL [$label]: $file not found"
        FAIL=$((FAIL+1))
        return
    fi
    local actual
    actual=$(wc -l < "$file" | tr -d ' ')
    if [[ "$actual" -eq "$expected" ]]; then
        echo "PASS [$label]: $file has $actual rows (expected $expected)"
        PASS=$((PASS+1))
    else
        echo "FAIL [$label]: $file has $actual rows (expected $expected)"
        FAIL=$((FAIL+1))
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
echo " Smoke Test: Pre-Classifier Stages 1-5.5"
echo "========================================"
echo ""

# --- Stage 1: pull_comments (threads input, corrected labeller) ---
echo "[STAGE 1] pull_comments --smoke (threads schema, corrected R→A→B labeller)"
"$PYTHON" scripts/pull_comments.py \
    --threads-input "$FIXTURES/mock_threads.jsonl" \
    --thread-ids-input "$FIXTURES/mock_econ_thread_ids.txt" \
    --output "$SMOKE_OUT/comments_raw.jsonl" \
    --counts-json "$SMOKE_OUT/stage1_counts.json" \
    --emit-bots \
    --smoke

# Expect 21 rows: all comments from 5 econ threads (bots included)
# t3_econA1: c1,c1a,c1b,c2 (4) | t3_econB2: c3,c3a,c3b,c4,c4a,c4b (6)
# t3_econC3: c5,c5a,c5b (3)    | t3_econD4: c6,c6a,c6b (3)
# t3_econF6: c7,c7a,c7b,c7c,c7d (5)
_check_jsonl_count "Stage1:row_count" "$SMOKE_OUT/comments_raw.jsonl" 21

# Verify non-economics thread E was excluded
NON_ECON=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_raw.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
bad = [l for l in lines if l.get('thread_id') == 't3_NONeconE5']
print(len(bad))
")
_check_value "Stage1:noneconomics_excluded" "$NON_ECON" "0"

# Verify delta count: c1(econA1), c3(econB2), c4(econB2), c5(econC3), c7(econF6) → 5
# (c6/econD4 is NOT y=1 because its DeltaBot child is a rejection, not a confirmation)
DELTA_COUNT=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/stage1_counts.json') as f:
    d = json.load(f)
print(d.get('raw_comments_delta_awarded', -1))
")
_check_value "Stage1:delta_count" "$DELTA_COUNT" "5"

# Verify bot count: c1b,c3b,c4b,c5b,c6b,c7d = 6
BOT_COUNT=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/stage1_counts.json') as f:
    d = json.load(f)
print(d.get('raw_comments_bot', -1))
")
_check_value "Stage1:bot_count" "$BOT_COUNT" "6"

# Verify is_award_gesture flag on award gestures: c1a,c3a,c4a,c5a,c7c = 5 in non-bot rows
# (c6a is NOT an award gesture — its child is a DeltaBot rejection, not confirmation)
AG_COUNT=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_raw.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
print(sum(1 for l in lines if l.get('is_award_gesture', False)))
")
_check_value "Stage1:award_gesture_count" "$AG_COUNT" "5"

# Verify c1 is y=1 (persuasive argument, the RECIPIENT, not the award gesture)
C1_Y=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_raw.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
row = next((l for l in lines if l.get('comment_id') == 'c1'), None)
print(row['y'] if row else 'MISSING')
")
_check_value "Stage1:c1_is_recipient_y1" "$C1_Y" "1"

# Verify c1a is y=0 (award gesture, NOT the recipient)
C1A_Y=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_raw.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
row = next((l for l in lines if l.get('comment_id') == 'c1a'), None)
print(row['y'] if row else 'MISSING')
")
_check_value "Stage1:c1a_award_gesture_y0" "$C1A_Y" "0"

# Verify c6 is y=0 (rejection scenario — DeltaBot child does not match CONFIRMED pattern)
C6_Y=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_raw.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
row = next((l for l in lines if l.get('comment_id') == 'c6'), None)
print(row['y'] if row else 'MISSING')
")
_check_value "Stage1:c6_rejection_y0" "$C6_Y" "0"

# Verify c7 is y=1 (back-and-forth: earliest ancestor of DeltaBot whose author==iris)
C7_Y=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_raw.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
row = next((l for l in lines if l.get('comment_id') == 'c7'), None)
print(row['y'] if row else 'MISSING')
")
_check_value "Stage1:c7_backandforth_y1" "$C7_Y" "1"

# Verify c7b is y=0 (later iris comment — NOT the root-closest, so NOT the recipient)
C7B_Y=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_raw.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
row = next((l for l in lines if l.get('comment_id') == 'c7b'), None)
print(row['y'] if row else 'MISSING')
")
_check_value "Stage1:c7b_later_iris_y0" "$C7B_Y" "0"

# --- Stage 1.5: validate_delta_labels ---
echo ""
echo "[STAGE 1.5] validate_delta_labels (mock_pairs vs mock_threads)"
"$PYTHON" scripts/validate_delta_labels.py \
    --pairs-input "$FIXTURES/mock_pairs.jsonl" \
    --threads-input "$FIXTURES/mock_threads.jsonl" \
    --output "$SMOKE_OUT/delta_label_validation.json" \
    --log-level INFO

RECALL=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/delta_label_validation.json') as f:
    d = json.load(f)
print(d.get('recall', 0))
")
_check_value "Stage1.5:recall_1.0" "$RECALL" "1.0"

PRECISION=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/delta_label_validation.json') as f:
    d = json.load(f)
print(d.get('precision', 0))
")
# Precision may be < 1 for t3_econB2: pairs records c3 only, but we also correctly
# detect c4 as a recipient (multi-delta thread). This is expected and not a bug.
# We only gate on recall == 1.0. Precision is informational.
if [[ "$RECALL" == "1.0" ]]; then
    echo "PASS [Stage1.5:recall_gate]: recall=$RECALL (gate passed)"
    PASS=$((PASS+1))
else
    echo "FAIL [Stage1.5:recall_gate]: recall=$RECALL (must be 1.0)"
    FAIL=$((FAIL+1))
fi
echo "INFO [Stage1.5:precision]: $PRECISION (may be <1.0 for multi-delta threads — not a bug)"

# --- Stage 2: filter_comments ---
echo ""
echo "[STAGE 2] filter_comments"
"$PYTHON" scripts/filter_comments.py \
    --input "$SMOKE_OUT/comments_raw.jsonl" \
    --output "$SMOKE_OUT/comments_filtered.jsonl" \
    --counts-json "$SMOKE_OUT/stage2_counts.json"

# Expected survivors: c1, c2, c3, c4, c6, c7, c7a, c7b = 8
# Removed: bots(6) + self-delta c5(1) + award gestures c1a,c3a,c4a,c5a,c7c(5) + short c6a(1) = 13
# c7a and c7b each have exactly 20 whitespace tokens (not < 20) so they survive
_check_jsonl_count "Stage2:row_count" "$SMOKE_OUT/comments_filtered.jsonl" 8

# Verify bots were removed (no is_bot=True rows)
BOT_ROWS=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_filtered.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
print(sum(1 for l in lines if l.get('is_bot', False)))
")
_check_value "Stage2:no_bot_rows" "$BOT_ROWS" "0"

# Verify award gestures removed (no is_award_gesture=True rows)
AG_ROWS=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_filtered.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
print(sum(1 for l in lines if l.get('is_award_gesture', False)))
")
_check_value "Stage2:no_award_gesture_rows" "$AG_ROWS" "0"

# Verify self-delta c5 is gone (grace awarded delta to herself)
SELF_DELTA_PRESENT=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_filtered.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
print(sum(1 for l in lines if l.get('comment_id') == 'c5'))
")
_check_value "Stage2:self_delta_c5_removed" "$SELF_DELTA_PRESENT" "0"

# Verify c1 (the recipient, y=1) is still present
C1_PRESENT=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_filtered.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
print(sum(1 for l in lines if l.get('comment_id') == 'c1'))
")
_check_value "Stage2:c1_recipient_present" "$C1_PRESENT" "1"

# Verify c1a (award gesture) is gone
C1A_PRESENT=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_filtered.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
print(sum(1 for l in lines if l.get('comment_id') == 'c1a'))
")
_check_value "Stage2:c1a_award_gesture_removed" "$C1A_PRESENT" "0"

# Verify c7 (back-and-forth recipient) survived Stage 2 and is y=1
C7_STAGE2=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_filtered.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
row = next((l for l in lines if l.get('comment_id') == 'c7'), None)
print(row['y'] if row else 'MISSING')
")
_check_value "Stage2:c7_recipient_y1" "$C7_STAGE2" "1"

# Verify c7b survived Stage 2 but is y=0 (not the earliest ancestor)
C7B_STAGE2=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_filtered.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
row = next((l for l in lines if l.get('comment_id') == 'c7b'), None)
print(row['y'] if row else 'MISSING')
")
_check_value "Stage2:c7b_later_iris_y0" "$C7B_STAGE2" "0"

# --- Stage 3: segment_sentences ---
echo ""
echo "[STAGE 3] segment_sentences (CPU)"
"$PYTHON" scripts/segment_sentences.py \
    --input "$SMOKE_OUT/comments_filtered.jsonl" \
    --output "$SMOKE_OUT/comments_segmented.jsonl" \
    --counts-json "$SMOKE_OUT/stage3_counts.json" \
    --device cpu

_check_jsonl_count "Stage3:row_count" "$SMOKE_OUT/comments_segmented.jsonl" 8

HAS_SENTS=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/comments_segmented.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
ok = sum(1 for l in lines if isinstance(l.get('sentences'), list) and len(l['sentences']) > 0)
print(ok)
")
_check_value "Stage3:all_have_sentences" "$HAS_SENTS" "8"

# --- Stage 5: compute_controls ---
echo ""
echo "[STAGE 5] compute_controls (two-pass)"
"$PYTHON" scripts/compute_controls.py \
    --input "$SMOKE_OUT/comments_segmented.jsonl" \
    --raw-output "$SMOKE_OUT/controls_raw.parquet" \
    --final-output "$SMOKE_OUT/controls_final.parquet" \
    --hedge-file data/hedges_hyland_2005.txt \
    --counts-json "$SMOKE_OUT/stage5_counts.json" \
    --device cpu
_check_file "Stage5:controls_raw" "$SMOKE_OUT/controls_raw.parquet"
_check_file "Stage5:controls_final" "$SMOKE_OUT/controls_final.parquet"

CTRL_COLS=$("$PYTHON" -c "
import pandas as pd
df = pd.read_parquet('$SMOKE_OUT/controls_final.parquet')
expected = ['noun_to_verb_ratio','type_token_ratio','flesch_kincaid','hedge_density',
            'mean_sentence_length','mean_word_length','punctuation_density','paragraph_count',
            'num_sentences','num_tokens']
missing = [c for c in expected if c not in df.columns]
print(len(missing), missing)
")
_check_value "Stage5:all_control_columns" "$CTRL_COLS" "0 []"

# --- Stage 5.5: prepare_for_classifier ---
echo ""
echo "[STAGE 5.5] prepare_for_classifier"
"$PYTHON" scripts/prepare_for_classifier.py \
    --input "$SMOKE_OUT/comments_segmented.jsonl" \
    --output "$SMOKE_OUT/sentences_for_classifier.jsonl" \
    --counts-json "$SMOKE_OUT/stage55_counts.json"
_check_file "Stage5.5:output_exists" "$SMOKE_OUT/sentences_for_classifier.jsonl"

SCHEMA_OK=$("$PYTHON" -c "
import json
with open('$SMOKE_OUT/sentences_for_classifier.jsonl') as f:
    first = json.loads(f.readline())
expected = {'comment_id','sentence_index','sentence_text','thread_id'}
missing = expected - set(first.keys())
print(len(missing), missing)
")
_check_value "Stage5.5:schema" "$SCHEMA_OK" "0 set()"

echo ""
echo "========================================"
echo " SMOKE TEST RESULTS: $PASS passed, $FAIL failed"
echo "========================================"
if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
