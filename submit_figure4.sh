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

# Load required modules with specific versions for compatibility
module load python/3.11 || module load python/3.10 || module load python/3.9
module load gcc/11.2.0 || module load gcc/10.1.0 || module load gcc/9.3.0 || echo "GCC module not found"

cd "$SLURM_SUBMIT_DIR"

python generate_figure4.py