#!/bin/bash
#SBATCH --job-name=phase_b_stage6_classifier
#SBATCH --output=logs/stage6_%j.out
#SBATCH --error=logs/stage6_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# BLOCKED — fill in after classifier shootout winner is decided.
# See run_clt_classifier.py for input/output contract.
echo "Stage 6 sbatch is a placeholder. Implement after shootout winner is selected."
exit 1
