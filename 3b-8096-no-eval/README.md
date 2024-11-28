# 544-nlp-project

## Evaluate models

Some baseline models to try:
- defog/sqlcoder-7b-2
- seeklhy/codes-1b
- seeklhy/codes-7b
- seeklhy/codes-7b-merged

Running on CARC:

```
cd baseline_models
sbatch evaluate_model.sh "model_name"
```

Running locally (GPU recommended):

```
cd baseline_models
./evaluate_model_local.sh "model_name"
```
