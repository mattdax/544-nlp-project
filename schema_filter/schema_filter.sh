#!/bin/bash

#SBATCH --account=swabhas_1457
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

module purge
module load gcc git nvhpc python/3.12.2


# Setup virtualenv
if [[ -d venv ]]; then
  source ./venv/bin/activate
else
  python -m venv venv
  source ./venv/bin/activate
  python -m pip install -r requirements.txt
fi

python main.py
