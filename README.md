# Predicting TempNorm Experiment

This repository runs a two-phase experiment.

## Phase 1: Data generation
For each input context:
1. Generate `initial_wander` tokens by pure sampling to form `starting_point`.
2. Record the final-layer hidden state at the last token of `starting_point` as `starting_hidden_vector` (`float16`).
3. From `starting_point`, generate `gen_number` independent completions of length `gen_length` by pure sampling.
4. For each completion, store the sampled token probabilities (`model_probs`) at each step.
5. For each `eval_length`, compute:
   - `individual_tempnorms_{eval_length}`: length-`gen_number` vector where each value is
     `prod_{t=1..eval_length} p_t^(alpha-1)`
   - `tempnorm_mean_{eval_length}`
   - `tempnorm_variance_{eval_length}`

## Phase 2: Learn hidden-vector -> individual tempnorm
Train one regression model per `eval_length` to predict an individual completion tempnorm from `starting_hidden_vector`.

Current default learner:
- Standardized ridge regression trained in log-tempnorm space (stable for very small products).

## Default parameters
- `model`: `facebook/opt-125m`
- `gen_length`: `100`
- `eval_lengths`: `[1,2,5,10,20,30,40,50,60,80,100]`
- `gen_number`: `8`
- `alpha`: `4`
- `initial_wander`: `30`

## Setup
```bash
pip install -r requirements.txt
```

## Prepare contexts
Put one context per line in `data/contexts.txt`.

## Run phase 1
```bash
python -m predicting_tempnorm.run_experiment \
  --contexts-file data/contexts.txt \
  --output-file data/experiment_data.jsonl
```

## Run phase 2
```bash
python -m predicting_tempnorm.train_tempnorm_predictor \
  --data-file data/experiment_data.jsonl \
  --output-dir data/models
```

## Colab usage
```python
!git clone <your-repo-url>
%cd predicting_tempnorm
!pip install -r requirements.txt
!python -m predicting_tempnorm.run_experiment --contexts-file data/contexts.txt --output-file data/experiment_data.jsonl
!python -m predicting_tempnorm.train_tempnorm_predictor --data-file data/experiment_data.jsonl --output-dir data/models
```

## Output schema (phase 1)
Each JSONL line is one starting-point data record and includes:
- `starting_point`
- `starting_hidden_vector`
- `generations` with `token_ids`, `generated_text`, and `model_probs`
- `individual_tempnorms_{eval_length}` for each eval length
- `tempnorm_mean_{eval_length}` for each eval length
- `tempnorm_variance_{eval_length}` for each eval length

## Output schema (phase 2)
In `--output-dir`:
- `tempnorm_model_eval_{eval_length}.pkl` (trained model)
- `split_eval_{eval_length}.json` (train/test sample indices)
- `metrics.json` (aggregate train/test metrics)