#!/bin/bash

#SBATCH --account=swabhas_1457
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=20:00

module purge
module load gcc git python/3.12.2

# Setup virtualenv
if [[ -d venv ]]; then
  source ./venv/bin/activate
else
  python -m venv venv
  source ./venv/bin/activate
  pip install nltk sqlparse
fi

# Download NLTK tokenizer
python -c "import nltk; nltk.download('punkt_tab')"

# Run benchmark
python ./benchmarks/test-suite-sql-eval/evaluation.py \
  --gold gold_query.txt \
  --pred generated.txt \
  --db ./benchmarks/spider_data/database \
  --table ./benchmarks/spider_data/tables.json \
  --etype all
