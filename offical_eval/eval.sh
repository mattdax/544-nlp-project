#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00

module purge
module load gcc git nvhpc python/3.12.2

python -m venv venv
source ./venv/bin/activate
python -m pip install -r requirements.txt

python eval.py
