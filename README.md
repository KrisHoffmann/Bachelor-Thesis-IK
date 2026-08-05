Kris Hoffmann

S4546407

Title : Construal Level Theory and Persuasion on
ChangeMyView
A Sentence-Level Classifier Cascade and
Conditional Logistic Analysis in Economics
Threads on ChangeMyView

Grade 8.6 
Available at : https://arts.studenttheses.ub.rug.nl/id/eprint/39163.

First Supervisor : Khalid Al-Khatib

Second Reader: Andreas van Cranenburgh

# CLT Cascade Classifier — Bachelor Thesis

Two-stage Construal Level Theory classifier fine-tuned on annotated ChangeMyView sentences.

- **Stage 1** — binary salience classifier (BERT-base-uncased, class-weighted cross-entropy,
  ~83/17 salient/non-salient imbalance).
- **Stage 2** — four-head CLT dimension classifier operating on salient sentences only.
  Each dimension (Temporal, Spatial, Social, Hypothetical) is a 3-class head: N/A / Near / Far.
  Temporal and Spatial are severely imbalanced (~90% N/A); classifier collapse to N/A on
  those dimensions is expected and reported, not hidden.

---

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Data contract

`data/processed/` is **gitignored**. Before running any script, copy the three split files there:

```
data/processed/train.json
data/processed/dev.json
data/processed/test.json
```

Each file is a JSON array of objects with this schema:

| Field           | Type    | Description                                      |
|-----------------|---------|--------------------------------------------------|
| `task_id`       | int     | Label Studio task ID                             |
| `post_id`       | str     | Reddit post ID                                   |
| `sentence_idx`  | int     | Sentence position within the post                |
| `sentence_text` | str     | Raw sentence text (model input)                  |
| `salient`       | bool    | Gold label: `true` = salient (class 1), `false` = not salient (class 0) |
| `temporal`      | int\|null | CLT temporal dimension: -1=N/A, 0=Near, 1=Far (Stage 2) |
| `spatial`       | int\|null | CLT spatial dimension: -1=N/A, 0=Near, 1=Far (Stage 2)  |
| `social`        | int\|null | CLT social dimension: -1=N/A, 0=Near, 1=Far (Stage 2)   |
| `hypothetical`  | int\|null | CLT hypothetical dimension: -1=N/A, 0=Near, 1=Far (Stage 2) |

Expected counts: train=723 (621 salient), dev=176 (147 salient), test=155 (127 salient).
`null` dimension values (4 records total, annotation gaps) are treated as -1 (N/A).

---

## Quickstart

```bash
# Smoke test — untrained forward pass, Stage 1 + Stage 2, CPU, <1 min
python scripts/smoke_test.py

# Mini-train Stage 1 — 50 train + 10 dev sentences, 1 epoch, CPU, <10 min
python scripts/mini_train.py

# Mini-train Stage 2 — 50 salient train + 15 salient dev, 1 epoch, CPU, <15 min
python scripts/mini_train_stage2.py

# Full train Stage 1 — uses configs/bert_stage1.yaml, GPU if available
python src/train.py --config configs/bert_stage1.yaml

# Full train Stage 2 — uses configs/bert_stage2.yaml, GPU if available
python src/train.py --config configs/bert_stage2.yaml
```

---

## Repository layout

```
configs/                     YAML training configs (bert_stage1.yaml, bert_stage2.yaml)
data/processed/              gitignored; place train/dev/test.json here
src/                         core library (data, model, train, evaluate)
  data.py                      load_split, load_split_stage2, verify_splits, verify_stage2_splits
  model.py                     build_model (Stage 1), CLTStage2Model (Stage 2)
  train.py                     unified training script, --stage 1|2
  evaluate.py                  compute_metrics_dict (Stage 1), evaluate_stage2 (Stage 2)
scripts/
  smoke_test.py                Stage 1 + Stage 2 untrained forward passes
  mini_train.py                Stage 1 CPU mini-train
  mini_train_stage2.py         Stage 2 CPU mini-train
  stage5_5_sentence_export.py  Explode comments_segmented.jsonl → sentences_for_classifier.jsonl
  stage6_inference.py          BERT cascade inference driver (Stage 1 + Stage 2)
slurm/
  stage6_smoke.sbatch          Smoke-test SLURM job (1000 sentences, gpushort)
outputs/                     gitignored; checkpoints, predictions, metrics
```

---

## Phase C — Inference runbook

### Stage 5.5 — sentence export (local, CPU)

Run from the **repo root** on your laptop (or on Hábrók without SLURM):

```bash
python scripts/stage5_5_sentence_export.py
```

Input: `final_dataset_creation/dataset_construction/outputs/comments_segmented.jsonl`  
Output: `final_dataset_creation/dataset_construction/outputs/sentences_for_classifier.jsonl`  
Counts: `final_dataset_creation/dataset_construction/consort/stage5_5_counts.json`

Expected: ~3 million sentence rows, ~5 sentences per comment. Runs in a few minutes on a laptop.

---

### Stage 6 smoke test — BERT cascade on 1000 sentences (Hábrók, GPU)

**Prerequisites on Hábrók:**

1. Model weights extracted into `peer_handoff/models/`:
   - `peer_handoff/models/stage1_model/` (HuggingFace checkpoint dir)
   - `peer_handoff/models/stage2_best_model.pt`
2. Stage 5.5 output present at `final_dataset_creation/dataset_construction/outputs/sentences_for_classifier.jsonl`
3. Phase B venv active at `/scratch/s4546407/venvs/phase_b/`

**Submit:**

```bash
cd /scratch/s4546407/clt_thesis/Bachelor-Thesis-IK
sbatch slurm/stage6_smoke.sbatch
```

**Monitor:**

```bash
squeue -u s4546407
tail -f slurm/logs/stage6_smoke_<JOBID>.out
```

**Expected output path:** `final_dataset_creation/dataset_construction/outputs/stage6_smoke_predictions.jsonl`  
**Expected row count:** 1000 (one row per input sentence, including non-salient)

---

### What to check before moving to the profile run

- [ ] **Row count is 1000.** `wc -l stage6_smoke_predictions.jsonl` must equal 1000.
- [ ] **Output schema correct.** Every row has: `sentence_id`, `comment_id`, `thread_id`, `salience_pred`, `salience_logits` (len 2), `temporal_pred`, `temporal_logits` (len 3 or null), `spatial_pred`, `spatial_logits`, `social_pred`, `social_logits`, `hypothetical_pred`, `hypothetical_logits`.
- [ ] **No NaN / all-null failure.** `salience_pred` is 0 or 1 for every row. Salient rows have non-null Stage 2 fields; non-salient rows have null Stage 2 fields.
- [ ] **All 4 CLT dimensions populated for salient rows.** Check that `social_pred` and `hypothetical_pred` are not uniformly -1 across all salient sentences (that would indicate Stage 2 collapsed or was never run).
- [ ] **Wall-clock per sentence is sane.** Divide total Stage 1 + Stage 2 GPU wall-clock by 1000; target <10 ms/sentence on a V100. If >50 ms, investigate before committing to the full-corpus job.
- [ ] **GPU utilisation visible in the log.** Lines like `VRAM alloc=X.XXgb` should appear at each `--log-every` interval, confirming `python -u` flushing worked and the model is actually on GPU.
