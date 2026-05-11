#!/bin/bash
#SBATCH --job-name=phase_b_stage3_segment
#SBATCH --output=logs/stage3_%j.out
#SBATCH --error=logs/stage3_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=regular
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

module purge
module load Python/3.11.5-GCCcore-13.2.0

source /scratch/s4546407/venvs/phase_b/bin/activate

cd /scratch/s4546407/clt_thesis/Bachelor-Thesis-IK/final_dataset_creation/dataset_construction

mkdir -p logs outputs consort

python scripts/segment_sentences.py \
    --input outputs/comments_filtered.jsonl \
    --output outputs/comments_segmented.jsonl \
    --counts-json consort/stage3_counts.json \
    --device cpu \
    --batch-size 64 \
    --smoke \
    --log-level INFO