#!/bin/bash
#SBATCH --job-name=mlops_gpu_test
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_course
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --output=gpu_test_output.txt

echo "=== JOB STARTED ==="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"

# Load modules (same as before)
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load matplotlib/3.10.3-gfbf-2025a

# Activate virtual environment
source ~/mlops_env/bin/activate

echo "=== ENVIRONMENT INFO ==="
python --version

echo "=== PYTORCH/CUDA TEST ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')
"

echo "=== SIMPLE CALCULATION ==="
python -c "
import torch
import time

# Simple tensor calculation
x = torch.randn(1000, 1000)
if torch.cuda.is_available():
    x = x.cuda()
    
start = time.time()
y = x @ x.t()
elapsed = time.time() - start
print(f'Matrix multiplication (1000x1000): {elapsed:.4f} seconds')
"

echo "=== JOB COMPLETED ==="
date
