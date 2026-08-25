# Training

## Install

From the repository root:

```bash
pip install -e .
pip install -e "training[dev]"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Training never overwrites an existing checkpoint.  Use a new artifact root for
every clean reproduction.

## HUST

Prepare the immutable one-second uniform-subband cache:

```bash
python training/scripts/prepare_hust.py \
  --data-root /path/to/HUST \
  --config training/configs/hust/dynamics_full_nyquist.json \
  --output training/artifacts/hust/subband_cache
```

The released folds came from the following staged algorithm. Run every held
bearing (`6205`, `6206`, `6207`, `6208`). The examples show one fold.

```bash
# Initial source-only dynamics and query-adaptive operation metric.
python training/scripts/train_hust.py \
  --data-root /path/to/HUST \
  --config training/configs/hust/model_full_nyquist_query_adaptive.json \
  --artifact-root training/artifacts/hust_full_nyquist_query_adaptive_v1 \
  --only-held 6205 --device cuda

# Whole-source pseudo-held-bearing classification refinement. The dynamical
# forecast task has already been learned in the preceding source-only stage.
python training/scripts/refine_hust_dynamics.py \
  --data-root /path/to/HUST \
  --config training/configs/hust/model_full_nyquist_meta_classification_only.json \
  --artifact-root training/artifacts/hust_full_nyquist_meta_classification_only_v1 \
  --held 6205 --device cuda

# Frozen-trunk health attention.
python training/scripts/run_hust_attentive_health.py \
  --data-root /path/to/HUST \
  --config training/configs/hust/model_full_nyquist_attentive_health.json \
  --artifact-root training/artifacts/hust_attentive_health_v1 \
  --held 6205 --device cuda
```

The released `6208` refinement config intentionally points to the recorded
fallback initial dynamics checkpoint.  A byte-identical regeneration therefore
requires the corresponding initial stage under
`model_full_nyquist.json`. A new clean study should instead declare one uniform
initialization rule before fitting. Released hashes identify the packaged
checkpoints. A fresh training run receives its own hashes.

After all folds exist, `training/scripts/export_hust_release.py` assembles the
deployable checkpoint. It writes source centroids and support fields. No held
bearing value is fitted.

The retained dynamics pretraining is not cosmetic. A frozen random-dynamics
control reached mean final BAcc `0.888889`, prefix BAcc `0.902778`, and NLL
`0.525701`. Dynamics pretraining improved these to `0.972222`, `0.938889`, and
`0.322599`. The final classification-only refinement reached `1.000000`,
`0.991667`, and `0.120696` on the conditional fault-operation task. Keeping the
forecast loss active during that final refinement was weaker (`1.000000`,
`0.983333`, `0.136095`), which is why it is not part of the released final
refinement objective.

## Final project model

```bash
python training/scripts/train_final.py \
  --config training/configs/final/v1.json \
  --source /path/to/Data_final/2024 \
  --target /path/to/Data_final/2026 \
  --output training/artifacts/final_v1 \
  --device cuda:0
```

The program fits only on 2024, writes and hashes `model.pt`, then reopens that
sealed checkpoint before target replay.  Labels `-1` are excluded by the source
loader.  The 2026 mixed stream is evaluated in original order with one
file-start reset and no target gradient or statistic update.

## Reproducibility

Released weights and their manifest hashes are the authoritative packaged
artifacts. A fresh training run is recorded as a new stochastic reproduction
unless its checkpoint hashes match the release.
