#!/bin/bash
#SBATCH --job-name=topic_validation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home5/s4546407/clt_thesis_logs/topic_validation_%j.out

# --- UPDATE THIS PATH before submitting ---
DATASET_PATH=/scratch/s4546407/clt_thesis/data/3778298/threads.jsonl.bz2

source /scratch/s4546407/venvs/llama/bin/activate

export HF_HOME=/scratch/s4546407/hf_cache

echo "[$(date)] Starting filter_economics.py (validation mode)"
python final_dataset_creation/scripts/filter_economics.py \
    --input "$DATASET_PATH" \
    --mode validation

echo "[$(date)] Done. Inspect: final_dataset_creation/outputs/topic_filter/topic_validation_sample.csv"
