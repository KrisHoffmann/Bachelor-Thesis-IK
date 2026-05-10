# Hábrók SLURM Scripts

Stage 3 (spaCy sentence segmentation) and Stage 6 (CLT classifier) run on the
Hábrók GPU cluster. This directory contains the sbatch job scripts.

## Stage 3 — Sentence Segmentation

### 1. Transfer input to Hábrók

```bash
# From local machine, after run_pipeline_pre_classifier.sh completes:
scp outputs/comments_filtered.jsonl \
    s4546407@login1.hb.hpc.rug.nl:/scratch/s4546407/clt_thesis/dataset_construction/outputs/

# Also transfer the scripts if not already synced:
rsync -av --exclude='outputs/' --exclude='tests/smoke_outputs/' \
    . s4546407@login1.hb.hpc.rug.nl:/scratch/s4546407/clt_thesis/dataset_construction/
```

### 2. Verify modules on Hábrók

```bash
ssh s4546407@login1.hb.hpc.rug.nl
module avail Python
module avail CUDA
```

Update `#SBATCH` module lines in `stage3_segmentation.sbatch` if the version
strings differ.

### 3. Set up Python environment (one-time)

Option A — reuse the existing llama venv if it has spaCy + spacy-transformers:
```bash
source /scratch/s4546407/venvs/llama/bin/activate
python -c "import spacy; spacy.load('en_core_web_trf')"
```

Option B — create a dedicated venv:
```bash
python -m venv /scratch/s4546407/venvs/phase_b
source /scratch/s4546407/venvs/phase_b/bin/activate
pip install spacy>=3.7 spacy-transformers textstat pandas pyarrow scipy numpy
python -m spacy download en_core_web_trf
```

Edit `stage3_segmentation.sbatch` to activate whichever venv you choose.

### 4. Submit the job

```bash
cd /scratch/s4546407/clt_thesis/dataset_construction
mkdir -p logs
sbatch slurm/stage3_segmentation.sbatch
```

Monitor:
```bash
squeue -u s4546407
# When complete, check:
tail -50 logs/stage3_<jobid>.out
cat consort/stage3_counts.json
```

### 5. Transfer output back

```bash
# From local machine:
scp s4546407@login1.hb.hpc.rug.nl:/scratch/s4546407/clt_thesis/dataset_construction/outputs/comments_segmented.jsonl \
    outputs/comments_segmented.jsonl

scp s4546407@login1.hb.hpc.rug.nl:/scratch/s4546407/clt_thesis/dataset_construction/consort/stage3_counts.json \
    consort/stage3_counts.json
```

Then run `run_pipeline_post_segmentation.sh` locally.

---

## Stage 6 — CLT Classifier

`stage6_classifier.sbatch` is a placeholder. Fill it in after the classifier
shootout winner is decided. See `scripts/run_clt_classifier.py` for the
input/output contract.
