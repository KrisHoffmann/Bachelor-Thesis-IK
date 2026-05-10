# final_dataset_creation

## What this folder does

Filters the raw CMV (ChangeMyView) Reddit corpus down to threads about economics or economic policy using zero-shot classification with `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`. The pipeline has three stages: (1) a CPU-only profiling step that detects field names in the bz2 JSONL corpus, (2) a pre-filter that drops deleted/short threads, and (3) a DeBERTa inference pass that assigns an economics relevance score and writes the surviving thread IDs to disk.

---

## Local setup

The raw dataset (`threads.jsonl.bz2`) lives **outside the repository** at `~/cmv_data/3778298/` (or wherever you placed it) and is covered by `.gitignore` — it must never be committed.

Before the first Hábrók run, create the target directory on the cluster and copy the file:

```bash
# 1. Create the directory on Hábrók (run once)
ssh s4546407@interactive1.hb.hpc.rug.nl 'mkdir -p /scratch/s4546407/clt_thesis/data/3778298'

# 2. Copy the bz2 file from your local machine
scp ~/cmv_data/3778298/threads.jsonl.bz2 \
    s4546407@interactive1.hb.hpc.rug.nl:/scratch/s4546407/clt_thesis/data/3778298/
```

---

## Hábrók runbook

1. **Confirm the dataset arrived:**
   ```bash
   find /scratch/s4546407 -name '*.bz2'
   ```

2. **Run the smoke job** (profiling + pre-filter check, ~10 min, CPU):
   ```bash
   sbatch final_dataset_creation/scripts/habrok_topic_smoke.slurm
   ```

3. **Inspect `schema_probe.json`** to confirm field detection looks right:
   ```bash
   cat final_dataset_creation/outputs/topic_filter/schema_probe.json
   ```

4. **Run the validation job** (50-thread sample with DeBERTa, ~30 min, GPU):
   ```bash
   sbatch final_dataset_creation/scripts/habrok_topic_validation.slurm
   ```

5. **Hand-label** the `gold_label` column in `topic_validation_sample.csv`. Use `1` for economics, `0` for not. Leave blank rows you are unsure about.

6. **Compute accuracy** (see snippet below). Target: ≥ 80 % before running full.

7. **If accuracy ≥ 80 %, run the full job** (~4 hours, GPU):
   ```bash
   sbatch final_dataset_creation/scripts/habrok_topic_full.slurm
   ```

---

## Compute accuracy snippet

```python
import pandas as pd

df = pd.read_csv("final_dataset_creation/outputs/topic_filter/topic_validation_sample.csv")
df = df[df["gold_label"].notna() & (df["gold_label"].astype(str).str.strip() != "")]
accuracy = (df["model_label"] == df["gold_label"].astype(str)).mean()
print(f"Accuracy: {accuracy:.2%}  ({len(df)} labelled rows)")
```

---

## Outputs produced

| File | Description |
|---|---|
| `outputs/topic_filter/schema_probe.json` | Detected field names (body, title, id, comment-count) from the first 100 records. Required by `filter_economics.py`. |
| `outputs/topic_filter/topic_validation_sample.csv` | 50 randomly sampled threads with model scores. Add `gold_label` column manually to measure accuracy. |
| `outputs/topic_filter/cmv_economics_thread_ids.txt` | One thread ID per line — the final economics-filtered set. |
| `outputs/topic_filter/cmv_economics_profile.json` | Counts at each pipeline stage plus model metadata and wall-clock time. |
