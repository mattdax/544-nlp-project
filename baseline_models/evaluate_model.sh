#!/bin/bash

#SBATCH --account=swabhas_1457
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00

module purge
module load gcc git nvhpc python/3.12.2

export HF_HOME="$PWD/cache/huggingface"

# Clone submodules
if [[ ! -f ./benchmarks/test-suite-sql-eval/evaluation.py ]]; then
  pushd $(git rev-parse --show-toplevel)
  git submodule update --init --recursive
  popd
fi

# Setup virtualenv
if [[ -d venv ]]; then
  source ./venv/bin/activate
else
  python -m venv venv
  source ./venv/bin/activate
  python -m pip install -r requirements.txt
  MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation
fi

# Download spider dataset
if [[ ! -d ./benchmarks/spider_data ]]; then
  python -m pip install gdown
  gdown -O spider_data.zip 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J
  unzip spider_data.zip -d ./benchmarks
  rm -rf ./benchmarks/__MACOSX
fi

# Download NLTK tokenizer
python -c "import nltk; nltk.download('punkt_tab')"

# Predict queries using model
if [[ -z "$1" ]]; then
  python ./src/evaluate_model.py --batch_size 4
else
  python ./src/evaluate_model.py --model_name "$1" --batch_size 4
fi

# Run benchmark
python ./benchmarks/test-suite-sql-eval/evaluation.py \
  --gold gold_query.txt \
  --pred generated.txt \
  --db ./benchmarks/spider_data/database \
  --table ./benchmarks/spider_data/tables.json \
  --etype all 
