# Sanity Check (Independent Codebase)

This folder is a fully separate experiment codebase for a cluster-based sanity check.

## Overview

Phase A (`build_reference_space`):
1. Sample 2000 hidden activations at token index 50 (51st token) from distinct texts.
2. L2-normalize activations.
3. Fit PCA and keep top 30 directions.
4. Define `PC_activation` as the 30D PCA projection.
5. Fit KMeans with 50 clusters in `PC_activation` space.

Phase B (`generate_clustered_tempnorm_data`):
1. Sample 5000 new texts.
2. Use first 30 tokens as context.
3. Generate 20 tokens by pure sampling with `facebook/opt-125m`.
4. Compute final hidden activation at this starting point, project to `PC_activation`, assign cluster.
5. Generate 8 independent continuations of length 50.
6. Compute tempnorms at lengths `[1,2,5,10,20,50]`.

Phase C (`analyze_clustered_tempnorm`):
1. Compare global mean/variance of tempnorm@50 to within-cluster statistics.
2. Compare cluster-level variance to within-starting-point variance across 8 continuations.
3. Evaluate how predictive tempnorms@1,2,5,10 are for tempnorm@50.

## Install
```bash
pip install -r sanity_check/requirements.txt
```

## Run
```bash
python -m sanity_check.build_reference_space \
  --output-dir sanity_check/artifacts \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1

python -m sanity_check.generate_clustered_tempnorm_data \
  --artifacts-dir sanity_check/artifacts \
  --output-file sanity_check/data/clustered_tempnorm.jsonl \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1

python -m sanity_check.analyze_clustered_tempnorm \
  --data-file sanity_check/data/clustered_tempnorm.jsonl \
  --output-dir sanity_check/artifacts/analysis
```

## Notes
- This codebase is intentionally isolated inside `sanity_check/`.
- Pure sampling is used (no top-k/top-p/temperature scaling).
- `alpha` defaults to 4 for tempnorm computation: `prod_t p_t^(alpha-1)`.