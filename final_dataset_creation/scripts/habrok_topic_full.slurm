#!/bin/bash
#SBATCH --job-name=topic_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home5/s4546407/clt_thesis_logs/topic_full_%j.out

# --- UPDATE THIS PATH before submitting ---
DATASET_PATH=/scratch/s4546407/clt_thesis/data/3778298/threads.jsonl.bz2

source /scratch/s4546407/venvs/llama/bin/activate

export HF_HOME=/scratch/s4546407/hf_cache
export TRANSFORMERS_OFFLINE=1

echo "[$(date)] Starting filter_economics.py (full mode)"
python final_dataset_creation/scripts/filter_economics.py \
    --input "$DATASET_PATH" \
    --mode full \
    --batch_size 8

echo "[$(date)] Done. Outputs in: final_dataset_creation/outputs/topic_filter/"
