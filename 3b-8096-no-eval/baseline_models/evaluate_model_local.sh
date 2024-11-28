#!/bin/bash

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
if [[ ! -d benchmarks/spider_data ]]; then
  python -m pip install gdown
  gdown -O spider_data.zip 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J
  unzip spider_data.zip -d ./benchmarks
  rm -rf ./benchmarks/__MACOSX
fi

# Download NLTK tokenizer
python -c "import nltk; nltk.download('punkt_tab')"

# Predict queries using model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -z "$1" ]]; then
  python ./src/evaluate_model.py --batch_size 1 --quantize
else
  python ./src/evaluate_model.py --model_name $1 --batch_size 1 --quantize
fi

# Run benchmark
python ./benchmarks/test-suite-sql-eval/evaluation.py \
  --gold gold_query.txt \
  --pred predicted.txt \
  --db ./benchmarks/spider_data/database \
  --table ./benchmarks/spider_data/tables.json \
  --etype all 
