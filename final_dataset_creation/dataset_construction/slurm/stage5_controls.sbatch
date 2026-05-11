#!/bin/bash
#SBATCH --job-name=phase_b_stage5_controls
#SBATCH --output=logs/stage5_%j.out
#SBATCH --error=logs/stage5_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=regular
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

module purge
module load Python/3.11.5-GCCcore-13.2.0
source /scratch/s4546407/venvs/phase_b/bin/activate
cd /scratch/s4546407/clt_thesis/Bachelor-Thesis-IK/final_dataset_creation/dataset_construction
mkdir -p logs outputs consort

echo "=== Stage 5 controls — start $(date) ==="
echo "Host: $(hostname)"
echo "Python: $(which python)"
python -c "import spacy; print(f'spaCy {spacy.__version__}')"
python -c "import textstat; print('textstat OK')"

# Verify hedge file is present
if [ ! -f data/hedges_hyland_2005.txt ]; then
    echo "ERROR: data/hedges_hyland_2005.txt not found"
    exit 1
fi
echo "Hedge file: $(wc -l < data/hedges_hyland_2005.txt) lines"

# Verify input is present and has expected row count
if [ ! -f outputs/comments_segmented.jsonl ]; then
    echo "ERROR: outputs/comments_segmented.jsonl not found"
    exit 1
fi
echo "Input rows: $(wc -l < outputs/comments_segmented.jsonl)"

python scripts/compute_controls.py \
    --input outputs/comments_segmented.jsonl \
    --output outputs/comments_controls.parquet \
    --hedge-file data/hedges_hyland_2005.txt \
    --counts-json consort/stage5_counts.json \
    --device cpu \
    --log-level INFO

echo "=== Stage 5 controls — end $(date) ==="
echo "Output size: $(ls -lh outputs/comments_controls.parquet)"
