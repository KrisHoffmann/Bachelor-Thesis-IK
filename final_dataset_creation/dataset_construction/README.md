# Phase B Dataset Construction

## 1. Purpose

This directory implements the full dataset construction pipeline for the BA Thesis Phase B analysis.
It takes the CMV (ChangeMyView) thread dataset and produces a matched case-control dataset ready
for regression analysis of Construal Level Theory (CLT) indices.

---

## 2. Status Table

| Stage | Script | Status | Blocked by |
|-------|--------|--------|------------|
| 0a | `profile_pairs.py` | READY | — |
| 0b | `profile_authors.py` | DEPRECATED | — |
| 1 | `pull_comments.py` | **READY** | — |
| 1.5 | `validate_delta_labels.py` | **READY** | Stage 1 output |
| 2 | `filter_comments.py` | **READY** | Stage 1 output |
| 3 | `segment_sentences.py` | READY (Hábrók GPU) | Stage 2 output |
| 5 | `compute_controls.py` | READY | Stage 3 output |
| 5.5 | `prepare_for_classifier.py` | READY | Stage 3 output |
| 6 | `run_clt_classifier.py` | **BLOCKED** | classifier shootout |
| 7 | `aggregate_indices.py` | READY-but-blocked | Stage 6 output |
| 8 | `merge_final_dataset.py` | READY-but-blocked | Stage 6 output |
| 9 | `match_case_control.py` | READY-but-blocked | Stage 6 output |

Stage 4 (author features) has been **removed**. See Section 9.

---

## 3. Pipeline Diagram

```
threads.jsonl.bz2 ────────────────────────────┐
thread_ids ───────────────────────────────────┤
                                               ▼
                                       [Stage 1] pull_comments
                                               │ comments_raw.jsonl
                                               │
pairs.jsonl.bz2 ──────────────────────────────┤
                                               ▼
                                       [Stage 1.5] validate_delta_labels (QA)
                                               │ recall must ≥ 0.90 (default threshold)
                                               ▼
                                       [Stage 2] filter_comments
                                               │ comments_filtered.jsonl
                                               ▼
                                       [Stage 3] segment_sentences
                                       (Hábrók GPU, sbatch)
                                               │ comments_segmented.jsonl
                                               ├────────────────┐
                                               ▼                ▼
                                  [Stage 5] compute_controls  [Stage 5.5] prepare_for_classifier
                                   controls_final.parquet      sentences_for_classifier.jsonl
                                               │                         │
                                               │               [Stage 6] BLOCKED (shootout)
                                               │                         │ sentence_predictions.parquet
                                               │                         ▼
                                               │               [Stage 7] aggregate_indices
                                               │                         │ comment_indices.parquet
                                               └──────────────┬──────────┘
                                                              ▼
                                                      [Stage 8] merge_final_dataset
                                                              │ analysis_dataset_pre_matching.parquet
                                                              ▼
                                                      [Stage 9] match_case_control
                                                              │ matched_dataset_ratio3.parquet (primary)
                                                              │ matched_dataset_ratio5.parquet (robustness)
```

---

## 4. Confirmed Schema

### threads.jsonl.bz2

| Field | Description |
|-------|-------------|
| `id` | Thread ID, e.g. `t3_3i1y4z` |
| `name` | Same as `id` |
| `selftext` | OP body text |
| `title` | Submission title |
| `author` | OP username |
| `created_utc` | Unix timestamp string |
| `num_comments` | Comment count |
| `delta` | Submission-level: did OP award any delta |
| `comments` | List of top-level comment objects |

Each comment object:

| Field | Description |
|-------|-------------|
| `id` | Comment ID (no `t1_` prefix on this field) |
| `parent_id` | `t1_xxx` (reply to comment) or `t3_xxx` (reply to OP) |
| `link_id` | `t3_xxx` — the thread ID |
| `level` | Nesting depth |
| `body` | Comment text |
| `author` | Username |
| `created_utc` | Unix timestamp string |
| `score` | Vote score |
| `controversiality` | 0 or 1 |
| `children` | List of nested reply objects (same schema) |

---

## 5. Δ-Label Derivation

Delta labels are NOT present as a field on comments. They are derived from the
comment tree structure using DeltaBot's confirmation body.

**The CMV delta-award flow:**

```
R  (delta-recipient — the persuasive argument that changed someone's mind)
└─ A  (award gesture — the persuaded user's reply containing "∆")
   └─ B  (DeltaBot confirmation: "Confirmed: N delta(s) awarded to /u/<recipient>")
```

In simple threads this is a clean depth-2 structure. In back-and-forth threads, R
may be several levels above A because the persuader and persuaded user exchange
multiple replies before the delta is awarded.

**The labelling rule (username-parse + earliest-ancestor):**

A comment R is delta-awarded (y=1) iff there exists a DeltaBot confirmation B
anywhere in R's subtree such that:
1. `B.author == "DeltaBot"` and `B.body` matches `^Confirmed:\s+\d+\s+delta(s)?\s+awarded` (case-insensitive),
2. `B.body` names a recipient username U via `…awarded to /u/<U>…`, and
3. R is the **earliest ancestor** of B (closest to the thread root) whose `author == U`.

Rule (3) ensures the credit goes to R — the first (root-closest) comment by the
named user in DeltaBot's ancestor chain — regardless of how many back-and-forth
exchanges occurred before the award.

**is_award_gesture flag:** A comment A is an award gesture iff one of its direct
children is a DeltaBot confirmation. Award gestures are emitted with
`is_award_gesture=True` and removed by Stage 2 — they are not persuasive arguments
and must not appear as cases or controls.

**Rejected/revoked/removed** DeltaBot replies do NOT match the confirmed pattern
and do NOT trigger y=1.

**Implementation in `scripts/_common.py`:**
```python
def find_delta_recipients(thread_record: dict) -> set[str]:
    # Ancestor-chain walk; parses /u/<username> from DeltaBot body
    # Returns set of comment ids (R) that are delta-recipients
    ...

def is_award_gesture(comment: dict) -> bool:
    # Returns True for award-gesture comments (A)
    ...
```

**QA step (Stage 1.5):** Run `validate_delta_labels.py` against `pairs.jsonl.bz2` ground
truth. The script gates on `--recall-threshold` (default 0.90). Observed real-data recall
is ~0.93; the ~7% gap is structural:
- ~5.3% FNs: recipient's comment was deleted from Reddit before this corpus snapshot.
- ~1.7% FNs: curation differences (pairs chose a later reply; our rule picks the root-closest).
Neither category is a labeller bug. Precision may be < 1.0 for multi-delta threads (correct —
we label all recipients; pairs records only one per submission). If recall < 0.90, debug
`find_delta_recipients` before proceeding.

---

## 6. Run Order

### a. ONE-TIME: install dependencies

```bash
pip install -r requirements_phase_b.txt
python -m spacy download en_core_web_sm
# On Hábrók: also pip install spacy-transformers && python -m spacy download en_core_web_trf
```

### b. Run smoke tests (MUST PASS before real run)

```bash
cd final_dataset_creation/dataset_construction
bash tests/smoke_test_pre_classifier.sh
bash tests/smoke_test_post_classifier.sh
```

### c. Run pre-classifier pipeline on real data

```bash
bash run_pipeline_pre_classifier.sh
```

Runs Stages 1, 1.5, 2. Stage 1.5 exits non-zero if recall < 0.90 (the default threshold).

### d. QA delta labeller

Stage 1.5 runs automatically inside `run_pipeline_pre_classifier.sh`. If it fails,
see `outputs/validation/delta_label_validation.json` for false negatives and debug
`find_delta_recipients` in `_common.py`.

### e. Stage 3: run on Hábrók GPU

Default routing is Hábrók GPU (`slurm/stage3_segmentation.sbatch`).
Local CPU is fallback only (suitable for < ~50k comments).

See `slurm/README.md` for transfer instructions and sbatch submission.

### f. Run post-segmentation pipeline

```bash
bash run_pipeline_post_segmentation.sh
```

Produces `outputs/sentences_for_classifier.jsonl`.

### g. WAIT for classifier shootout

Implement Stage 6 (`scripts/run_clt_classifier.py`) once the shootout winner is decided.
Update `slurm/stage6_classifier.sbatch` for the chosen model.

### h. Run Stage 6 on Hábrók

```bash
sbatch slurm/stage6_classifier.sbatch
```

### i. Run post-classifier pipeline

```bash
bash run_pipeline_post_classifier.sh
```

Runs Stages 7, 8, 9 (both ratio 3 and ratio 5).

---

## 7. Stage 3 Routing Decision

Default routing is **Hábrók GPU** (`slurm/stage3_segmentation.sbatch`).

Local CPU is a fallback if Hábrók is unavailable and comment count is small:

| Post-Stage-2 count | Recommendation |
|--------------------|----------------|
| < 50,000 | Local CPU (en_core_web_sm) may be acceptable |
| > 50,000 | Hábrók GPU (en_core_web_trf) strongly preferred |

Use `run_pipeline_post_segmentation.sh` with `--device cpu` for local runs by setting
`DEVICE=cpu` before calling `segment_sentences.py` directly.

---

## 8. CONSORT Chain

| Key | Written by | Meaning |
|-----|-----------|---------|
| `threads_in_economics_filter` | Stage 1 | Thread IDs in allowlist |
| `threads_matched_in_corpus` | Stage 1 | Threads found in threads.jsonl.bz2 |
| `threads_missing_from_corpus` | Stage 1 | Allowlist IDs not found (deleted/removed threads) |
| `raw_comments_total` | Stage 1 | All comments before any filtering |
| `raw_comments_bot` | Stage 1 | Bot/AutoModerator comments |
| `raw_comments_nonbot` | Stage 1 | Non-bot comments |
| `raw_comments_delta_awarded` | Stage 1 | Delta-recipient comments (y=1, before filtering) |
| `removed_bots` | Stage 2 | Removed in step 1 |
| `after_bots` | Stage 2 | After bot removal |
| `removed_self_deltas` | Stage 2 | Removed in step 2 |
| `after_self_deltas` | Stage 2 | After self-delta removal |
| `removed_award_gestures` | Stage 2 | Removed in step 3 (∆ reply comments, not persuasive arguments) |
| `after_award_gestures` | Stage 2 | After award gesture removal |
| `removed_deleted_or_removed` | Stage 2 | Removed in step 4 |
| `after_deleted_removed` | Stage 2 | After deleted removal |
| `removed_short` | Stage 2 | Removed in step 5 |
| `after_short` | Stage 2 | After short removal |
| `removed_quote_only` | Stage 2 | Removed in step 6 |
| `after_quote_only` | Stage 2 | Final filtered count |
| `comments_segmented` | Stage 3 | Comments with sentences field |
| `comments_with_controls` | Stage 5 | Comments with all control variables |
| `sentences_for_classifier` | Stage 5.5 | Total sentences exported |
| `comments_with_predictions` | Stage 7 | Comments in classifier output |
| `after_remove_zero_salience` | Stage 7 | After dropping all-NaN comments |
| `pre_merge_comments` | Stage 8 | Input rows to merge |
| `post_merge_rows` | Stage 8 | Final pre-matching rows |
| `total_strata` | Stage 9 | Matched strata created |
| `total_comments_retained` | Stage 9 | Comments in matched dataset |

---

## 9. Stage 4 Removed

Stage 4 (`extract_author_features.py`) has been **removed** from the pipeline.

**Reason:** None of the three regression models use author features as fixed-effect
predictors:
- Model 1 (conditional logistic regression): conditions on thread via matched strata;
  no author features.
- Model 2 (mediation decomposition): CLT indices only.
- Model 3 (mixed-effects logistic regression): author identity enters only through a
  random intercept, which requires only the `author` column already present from Stage 1.

`scripts/profile_authors.py` is retained as a deprecated reference for one-off author
analyses outside the registered methodology.

---

## 10. Decisions Deferred to Supervisor

1. **Classifier shootout winner** (Stage 6): The CLT sentence classifier is unimplemented.
   Implement `run_clt_classifier.py` and `slurm/stage6_classifier.sbatch` once decided.
2. **Hedge lexicon completeness** (Stage 5): `data/hedges_hyland_2005.txt` is a partial
   placeholder. Replace with the full Hyland (2005) lexicon before the real run.
3. **Matching ratio** (Stage 9): Primary analysis uses 1:3; robustness check uses 1:5.
   Confirm with supervisor whether both should be reported.

---

## 11. Smoke Test Instructions

```bash
cd final_dataset_creation/dataset_construction

# Pre-classifier stages (1–5.5):
bash tests/smoke_test_pre_classifier.sh

# Post-classifier stages (7–9) — run after pre-classifier:
bash tests/smoke_test_post_classifier.sh
```

Both tests must exit 0 before running on real data.
