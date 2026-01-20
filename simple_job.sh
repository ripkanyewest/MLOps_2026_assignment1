#!/bin/bash
#SBATCH --job-name=my_test
#SBATCH --time=00:03:00
#SBATCH --partition=gpu_course
#SBATCH --gres=gpu:1
#SBATCH --output=my_output.txt

echo "=== JOB START ==="
date
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module load 2025
module load Python/3.13.1-GCCcore-14.2.0
source ~/mlops_env/bin/activate

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo "=== JOB END ==="
date
