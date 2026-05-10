#!/bin/bash
#SBATCH --job-name=phase_b_stage3_segment
#SBATCH --output=logs/stage3_%j.out
#SBATCH --error=logs/stage3_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module purge
module load Python/3.11
module load CUDA/12.1
# Verify available modules on Hábrók with: module avail Python; module avail CUDA

# Option A: use the llama venv (if it has spacy + spacy-transformers installed)
source /scratch/s4546407/venvs/llama/bin/activate
# Option B: dedicated phase_b venv — run once to create:
#   python -m venv /scratch/s4546407/venvs/phase_b
#   source /scratch/s4546407/venvs/phase_b/bin/activate
#   pip install spacy>=3.7 spacy-transformers
#   python -m spacy download en_core_web_trf

cd /scratch/s4546407/clt_thesis/dataset_construction

python scripts/segment_sentences.py \
    --input outputs/comments_filtered.jsonl \
    --output outputs/comments_segmented.jsonl \
    --counts-json consort/stage3_counts.json \
    --device cuda \
    --batch-size 128 \
    --log-level INFO
