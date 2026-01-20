## Question 4: Your First Batch Job (Slurm)

### 1. Complete job script and output files

**Job script (`my_first_slurm_job.sh`):**
```bash
#!/bin/bash
#SBATCH --job-name=mlops_assignment1_test
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_course
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_output_%j.txt

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load the same modules I used in Question 2
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load matplotlib/3.10.3-gfbf-2025a

# Activate my virtual environment
source ~/mlops_env/bin/activate

# Test PyTorch and CUDA on a GPU node
python -c "
import torch
print(f'PyTorch version installed: {torch.__version__}')
print(f'CUDA available on this node: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'GPU device name: {torch.cuda.get_device_name(0)}')
    print(f'Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')
else:
    print('No GPU detected - unexpected on gpu_course partition')
"

echo "Job completed at: $(date)"
