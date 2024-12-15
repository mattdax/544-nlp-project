#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=40G
#SBATCH --time=48:00:00 
#SBATCH --nodelist=b05-10    
#SBATCH --exclusive 
module purge
module load gcc git nvhpc python/3.12.2

python -m venv venv
source ./venv/bin/activate
python -m pip install -r requirements.txt
MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation

python finetune.py
