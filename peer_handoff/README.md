# CLT Cascade Classifier — Inference Package

This package contains everything needed to run inference with a BERT-base cascade
classifier trained on the CLT (Construal Level Theory) annotation set. **Stage 1**
classifies sentence-level salience (binary: salient / not salient). **Stage 2** then
classifies four CLT dimensions — Temporal, Spatial, Social, and Hypothetical — on the
salient sentences only, predicting each as Near, Far, or N/A. The model was trained on
899 sentences (train + dev combined) from 9 CMV Reddit threads and evaluated once on
155 sentences from 2 held-out threads. Tokenizer and encoder: `bert-base-uncased`.

---

## Test Set Results

| Component | macro-F1 |
|---|---|
| Stage 1 (salience) | 0.869 |
| Stage 2: Temporal | 0.308 ⚠ |
| Stage 2: Spatial | 0.397 ⚠ |
| Stage 2: Social | 0.448 |
| Stage 2: Hypothetical | 0.590 |

---

> **⚠ Important — Temporal and Spatial predictions are not reliable.**
>
> On the held-out test set, the Temporal and Spatial Stage 2 heads effectively predict
> N/A for nearly all sentences. Per-class F1 scores:
>
> - **Temporal:** [N/A: 0.92, Near: 0.00, Far: 0.00], Krippendorff α = 0.03
> - **Spatial:** [N/A: 0.91, Near: 0.00, Far: 0.29], Krippendorff α = 0.20
>
> This is consistent with the known class sparsity issue (Near appears in fewer than 5%
> of salient training sentences, Far in fewer than 9%). **Do not use Temporal or Spatial
> predictions for downstream regression analysis.** Social (α = 0.14) and Hypothetical
> (α = 0.33) are usable.

---

## Setup

### 1. Drop model weights into `peer_handoff/models/`

You will receive the model weights as a separate zip file. Extract and place them as follows:

```
peer_handoff/
└── models/
    ├── stage1_model/           # extracted from the zip's stage1_model/ folder
    │   ├── config.json
    │   ├── model.safetensors   (or pytorch_model.bin)
    │   ├── tokenizer.json
    │   └── tokenizer_config.json
    └── stage2_best_model.pt    # from the root of the zip
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For Hábrók, load a Python 3.10+ module and a CUDA-enabled PyTorch before running pip:

```bash
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
pip install -r requirements.txt
```

### 3. Verify

```bash
python inference.py \
    --input example_input.json \
    --stage1_model models/stage1_model \
    --stage2_model models/stage2_best_model.pt \
    --output example_output.json
```

---

## Usage

### Basic command

```bash
python inference.py \
    --input  your_sentences.json \
    --stage1_model models/stage1_model \
    --stage2_model models/stage2_best_model.pt \
    --output predictions.json \
    --batch_size 32
```

| Flag | Required | Description |
|---|---|---|
| `--input` | ✓ | Path to input JSON file (see format below) |
| `--stage1_model` | ✓ | Path to Stage 1 HuggingFace checkpoint directory |
| `--stage2_model` | ✓ | Path to Stage 2 `.pt` state-dict file |
| `--output` | ✓ | Where to write the output JSON |
| `--batch_size` | | Inference batch size; default 32 |

### Running on Hábrók (SLURM)

Submit a SLURM job requesting a GPU. Example job script:

```bash
#!/bin/bash
#SBATCH --job-name=clt_inference
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

cd /path/to/peer_handoff

python inference.py \
    --input your_sentences.json \
    --stage1_model models/stage1_model \
    --stage2_model models/stage2_best_model.pt \
    --output predictions.json \
    --batch_size 32
```

**GPU is strongly recommended** (V100 or A100). The script automatically falls back to
CPU but will be significantly slower. Expected GPU memory footprint: roughly **2 GB** at
batch size 32 (Stage 1 and Stage 2 share a bert-base-uncased encoder but are loaded
sequentially, so only one model is in GPU memory at a time).

---

## Input Format

A JSON array of objects. Each object **must** contain:

| Field | Type | Required | Description |
|---|---|---|---|
| `sentence_text` | string | **Yes** | The sentence to classify |
| `post_id` | string | No | Reddit post or document ID — preserved in output |
| `sentence_idx` | int | No | Position of the sentence in its document — preserved in output |

Any additional fields you include are passed through unchanged to the output.

See `example_input.json` for a concrete example:

```json
[
  {
    "sentence_text": "If we implement stricter environmental regulations now, future generations will benefit.",
    "post_id": "post_1",
    "sentence_idx": 0
  },
  {
    "sentence_text": "I bought a sandwich for lunch.",
    "post_id": "post_1",
    "sentence_idx": 1
  }
]
```

---

## Output Format

A JSON array of objects in the same order as the input. All input fields are preserved
and the following fields are added:

| Field | Type | Values | Description |
|---|---|---|---|
| `salient` | bool | `true` / `false` | Stage 1 prediction — whether the sentence is salient |
| `temporal` | int | -1, 0, 1 | Stage 2 Temporal prediction |
| `spatial` | int | -1, 0, 1 | Stage 2 Spatial prediction |
| `social` | int | -1, 0, 1 | Stage 2 Social prediction |
| `hypothetical` | int | -1, 0, 1 | Stage 2 Hypothetical prediction |

**Stage 2 integer encoding:** `-1` = N/A (not applicable), `0` = Near, `1` = Far.

**Non-salient sentences** receive `-1` for all four Stage 2 dimensions. Stage 2 was not
run on them — this matches the cascade design used during training.

Example output record:

```json
{
  "sentence_text": "If we implement stricter environmental regulations now, future generations will benefit.",
  "post_id": "post_1",
  "sentence_idx": 0,
  "salient": true,
  "temporal": 1,
  "spatial": -1,
  "social": -1,
  "hypothetical": 1
}
```

---

## Source Code & Repository

Full training code, notebooks, and annotation pipeline:
[https://github.com/KrisHoffmann/Bachelor-Thesis-IK](https://github.com/KrisHoffmann/Bachelor-Thesis-IK)

This package includes `src/model.py` (the `CLTStage2Model` class) and `src/data.py`
(data loading utilities). If you want to inspect or modify the training pipeline, clone
the full repository. Do not modify `src/model.py` in this package without ensuring
it matches the file used during training — the model weights are tied to the exact
architecture defined there.

---

## Contact

Questions: [your contact info]
