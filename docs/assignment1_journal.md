# Assignment 1: Setup & Debugging Journal

**Name:** Farrell O'Keefe
**Student ID:** 15875202
**GitHub Repository:** https://github.com/ripkanyewest/MLOps_2026_assignment1

---

## Question 1: First Contact with Snellius

### 1. Exact command and login node
- Command: `ssh scur2337@snellius.surf.nl`
- Login node: int6

### 2. Problem encountered
Ik moest eerst SSH keys instellen via het SURF portaal. Daarna kon ik zonder wachtwoord inloggen.

### 3. SSH client details
- SSH client: macOS Terminal
- Prior experience: Geen ervaring met HPC clusters
- Preemptive steps: Ik las eerst de Snellius documentatie

---

## Question 2: Environment Setup

### 1. Sequence of commands and path to virtual environment
```bash
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load matplotlib/3.10.3-gfbf-2025a
python3 -m venv mlops_env
source mlops_env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Full path to virtual environment: /home/scur2337/mlops_env

### 2. Installation of torch

Installation time: Ongeveer 5 minuten
Warnings encountered: Geen belangrijke waarschuwingen
Size of venv folder: 2.8 GB

### 3. Mistake or unexpected behavior

De Python module die in de opdracht stond (Python/3.11.6-GCCcore-13.2.0) bestond niet. Ik gebruikte module spider Python om beschikbare versies te vinden en koos Python/3.13.1-GCCcore-14.2.0.

### 4. Output of PyTorch/CUDA check

PyTorch: 2.5.1+cu121
CUDA available: False

Uitleg: Dit is normaal op login nodes (geen GPU). CUDA zou alleen True zijn op GPU compute nodes.

## Question 3: Version Control Setup

### 1. GitHub repository URL

https://github.com/ripkanyewest/MLOps_2026_assignment1

### 2. Authentication method and errors

Gebruikte methode: HTTPS met Personal Access Token

Fouten die ik kreeg:

Verkeerde remote URL (nog de placeholder JOUW-USERNAME)
Per ongeluk token als commando ingetypt in terminal

Hoe ik het oploste:

URL gecorrigeerd: git remote set-url origin https://github.com/ripkanyewest/MLOps_2026_assignment1.git
Token bij wachtwoordprompt gebruikt tijdens git push

### 3. .gitignore contents

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
mlops_env/

# Data
*.h5
*.hdf5
*.csv
*.json
*.pkl
*.pickle
data/
*.zip
*.tar.gz

# Logs
*.log
logs/

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Snellius specifiek
.slurm/
*.out
*.err

### 4. git log output
880d065 (HEAD -> main, origin/main) Initial commit with skeleton repository

## Question 4: Your First Batch Job (Slurm)

### 1. Job script and output

Job script (correct_job.sh):
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

module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load matplotlib/3.10.3-gfbf-2025a
source ~/mlops_env/bin/activate

python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo "=== JOB COMPLETED ==="
date

Output excerpt:
=== JOB STARTED ===
mon jan 19 23:25:53 CET 2026
Running on node: gcn14
Job ID: 18479908
PyTorch version: 2.5.1+cu121
CUDA available: True
=== JOB COMPLETED ===
mon jan	19 23:25:53 CET	2026

### 2. Job ID and wait time

Job ID: 18479908
Wait time: Ongeveer 2-3 minuten
### 3. Problem encountered

Eerste job submission faalde omdat ik niet de juiste resource specificaties gebruikte voor gpu_course partition. Foutmelding: "Requested partition configuration not available now".

Oplossing: Juiste specificaties gebruikt: --ntasks=1 --cpus-per-task=2 --gpus=1 --mem=16G

### 4. Verification

Job succesvol afgerond met:

Slurm status: COMPLETED
Output file aangemaakt met verwachte inhoud
CUDA: True (GPU toegang bevestigd)

### 5. Login node vs batch job

Login node (int6): Voor setup, testen, file management. Geen GPU, interactief gebruik.
Batch job (gcnXX): Voor berekeningen. Heeft GPU, draait via scheduler, toegewezen resources.

### 6. Why use a cluster?

Toegang tot krachtige GPUs voor deep learning
Meerdere experimenten parallel draaien
Grote datasets verwerken
Eerlijke resource verdeling tussen gebruikers
Jobs blijven draaien zelfs als je verbinding verbreekt

## Question 5: Reflection & Conceptual Understanding

### 5.1 Filesystem Question

Kleine bestanden (100.000 × 10KB) veroorzaken performance problemen op GPFS vanwege hoge metadata operaties (IOPs). Het systeem is geoptimaliseerd voor grote sequentiële reads.

Twee oplossingen:

Bestanden samenvoegen in grotere containers (HDF5/TFRecord)
Lokale opslag gebruiken (/dev/shm RAM disk) tijdens training
Version control voor grote datasets:
Gebruik DVC (Data Version Control) met Git - sla alleen metadata op in Git, echte data apart.

### 5.2 Reproducibility Issues

Drie redenen voor verschillende resultaten:

Random seeds (data shuffling, weight initialization)
Software versie verschillen (PyTorch, CUDA, Python)
Hardware verschillen (GPU vs CPU, floating-point precision)
MLOps preventie:

Stel alle random seeds in configuratie
Gebruik exacte versies in pyproject.toml
Documenteer hardware requirements

### 5.3 Environment Tools Comparison

astral uv:

Goed: Snelle dependency resolutie
Fout: Moet geïnstalleerd worden, alleen Python
python venv:

Goed: Standaard in Python, simpel
Fout: Langzamere dependency resolutie
conda:

Goed: Beheert alle dependencies
Fout: Veel kleine bestanden (IOPs probleem), conflicten met systeem modules
Keuze voor Snellius: python venv - werkt goed met module systeem.

## Question 6: Package Integrity

### 1. ModuleNotFoundError experience

Moest meerdere missing dependencies installeren:

pip install pillow (voor PIL)
pip install six (vereist door torchvision)
pip install h5py (voor PCAM dataset)
pip install tqdm (voor progress bars)
pip install pyyaml (voor configuratie files)
### 2. Why import from ml_core.data?

Importeren vanaf package level (ml_core.data) biedt abstractie. Gebruikers hoeven interne file structuur niet te kennen, en interne veranderingen breken imports niet.

### 3. pytest test suite results

============================= test session starts ==============================
platform linux -- Python 3.13.1, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/scur2337/MLOps_2026
collected 1 item

tests/test_imports.py::test_imports PASSED                               [100%]

============================== 1 passed in 0.18s ===============================

