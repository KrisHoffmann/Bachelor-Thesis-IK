#!/bin/bash
#SBATCH --job-name=topic_smoke
#SBATCH --partition=regular
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home5/s4546407/clt_thesis_logs/topic_smoke_%j.out

# --- UPDATE THIS PATH before submitting ---
DATASET_PATH=/scratch/s4546407/clt_thesis/data/3778298/threads.jsonl.bz2

source /scratch/s4546407/venvs/llama/bin/activate

export HF_HOME=/scratch/s4546407/hf_cache

echo "[$(date)] Starting profile_corpus.py"
python final_dataset_creation/scripts/profile_corpus.py --input "$DATASET_PATH"

echo "[$(date)] Starting filter_economics.py (smoke mode)"
python final_dataset_creation/scripts/filter_economics.py \
    --input "$DATASET_PATH" \
    --mode smoke

echo "[$(date)] Done."
