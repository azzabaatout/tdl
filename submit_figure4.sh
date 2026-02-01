#!/bin/bash
#SBATCH --job-name=figure4
#SBATCH --partition=serc
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=fig.out
#SBATCH --error=fig.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=baatout@stanford.edu

module load python/3.12

cd "$SLURM_SUBMIT_DIR"

python generate_figure4.py