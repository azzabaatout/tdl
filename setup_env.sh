#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=setup_env_%j.out
#SBATCH --error=setup_env_%j.err

echo "🔧 Setting up PV Forecasting Environment"
echo "=" * 50

# Load required modules with specific versions for compatibility
echo "Loading Python module..."
module load python/3.11 || module load python/3.10 || module load python/3.9

echo "Loading GCC compiler..."
module load gcc/11.2.0 || module load gcc/10.1.0 || module load gcc/9.3.0 || echo "GCC module not found"

echo "Loading CUDA..."
module load cuda/11.8 || module load cuda/11.7 || module load cuda/11.5 || echo "CUDA module not loaded"

echo "Loaded modules:"
module list

# Force recreate virtual environment to ensure fresh GPU PyTorch installation
echo "Removing existing virtual environment..."
rm -rf venv_sherlock

echo "Creating fresh virtual environment..."
python3 -m venv venv_sherlock

source venv_sherlock/bin/activate

echo "Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip and install requirements
pip install --upgrade pip wheel setuptools

# Set compiler environment variables
export CC=gcc
export CXX=g++
export MPICC=mpicc
export MPICXX=mpic++

# Install packages in order with fallbacks
echo "Installing core scientific packages..."
pip install numpy scipy matplotlib scikit-learn

echo "Installing pandas (may take time)..."
pip install pandas || pip install pandas==2.0.3

echo "Installing TensorFlow..."
pip install tensorflow-cpu  # Use CPU version to avoid CUDA issues

echo "Installing Prophet..."
pip install prophet || pip install --no-binary prophet prophet

echo "Installing statsmodels..."
pip install statsmodels

echo "Installing PyTorch GPU version with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing Hugging Face Transformers..."
pip install transformers

echo "Verifying PyTorch GPU installation..."
python -c "
import torch
print('='*50)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
    print('✅ GPU installation successful!')
else:
    print('❌ GPU installation failed - will use CPU (very slow)')
print('='*50)
"

echo "Installing pytorch-forecasting..."
pip install "importlib-metadata>=4.0" || echo "importlib-metadata upgrade failed"
pip install pytorch-forecasting || echo "pytorch-forecasting installation failed, continuing..."

echo "Installing Google Cloud Storage..."
pip install google-cloud-storage

echo "Installing system monitoring tools..."
pip install psutil

pip install tqdm

# Set environment variables for optimal performance
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Google Cloud Storage optimization for network stability
export GOOGLE_CLOUD_DISABLE_GRPC=true
export CLOUDSDK_CORE_REQUEST_TIMEOUT=600
export CLOUDSDK_CORE_CHECK_GCE_METADATA=false

# Set Python path
export PYTHONPATH=$PYTHONPATH:$PWD


echo ""
echo "🎉 Environment setup completed successfully!"
echo "Virtual environment location: $(pwd)/venv_sherlock"
echo "To use this environment in training jobs:"
echo "  source $(pwd)/venv_sherlock/bin/activate"
echo ""
echo "Setup completed at: $(date)"

# Deactivate virtual environment
deactivate