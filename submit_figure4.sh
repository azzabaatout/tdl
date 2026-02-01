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
module load cuda/11.8 || module load cuda/11.7 || echo "CUDA module not loaded"

PROJECT_DIR="/scratch/users/baatout/tdl"
cd "$PROJECT_DIR"

if [ -d "venv_sherlock" ]; then
    source venv_sherlock/bin/activate
else
    echo "Virtual environment not found!"
    exit 1
fi

export PYTHONPATH=$PYTHONPATH:$PWD
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

python generate_figure4.py