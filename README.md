# CLT Salience Classifier — Bachelor Thesis

Binary salience classifier (Stage 1 of a Construal Level Theory cascade) fine-tuned on
annotated ChangeMyView sentences. Uses BERT-base-uncased with class-weighted cross-entropy
to handle the ~83/17 salient/non-salient imbalance.

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
| `temporal`      | int     | CLT temporal dimension (Stage 2, not used here)  |
| `spatial`       | int     | CLT spatial dimension (Stage 2, not used here)   |
| `social`        | int     | CLT social dimension (Stage 2, not used here)    |
| `hypothetical`  | int     | CLT hypothetical dimension (Stage 2, not used here) |

Expected counts: train=723, dev=176, test=155.

---

## Quickstart

```bash
# Smoke test — untrained forward pass on 5 sentences, CPU, <1 min
python scripts/smoke_test.py

# Mini-train — 50 train + 10 dev sentences, 1 epoch, CPU, <10 min
python scripts/mini_train.py

# Full train — uses configs/bert_stage1.yaml, GPU if available
python src/train.py --config configs/bert_stage1.yaml
```

---

## Repository layout

```
configs/           YAML training configs
data/processed/    gitignored; place train/dev/test.json here
src/               core library (data, model, train, evaluate)
scripts/           smoke_test.py, mini_train.py
outputs/           gitignored; checkpoints, predictions, metrics
```
