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

The released folds came from the following staged algorithm.  Run every held
bearing (`6205`, `6206`, `6207`, `6208`); the examples show one fold.

```bash
# Initial source-only dynamics and query-adaptive operation metric.
python training/scripts/train_hust.py \
  --data-root /path/to/HUST \
  --config training/configs/hust/model_full_nyquist_query_adaptive.json \
  --artifact-root training/artifacts/hust_full_nyquist_query_adaptive_v1 \
  --only-held 6205 --device cuda

# Whole-source pseudo-held-bearing representation refinement.
python training/scripts/refine_hust_dynamics.py \
  --data-root /path/to/HUST \
  --config training/configs/hust/model_full_nyquist_meta_refine.json \
  --artifact-root training/artifacts/hust_full_nyquist_meta_refine_v1 \
  --held 6205 --device cuda

# Frozen-trunk health attention.
python training/scripts/run_hust_attentive_health.py \
  --data-root /path/to/HUST \
  --config training/configs/hust/model_full_nyquist_attentive_health.json \
  --artifact-root training/artifacts/hust_attentive_health_v1 \
  --held 6205 --device cuda
```

The historical `6208` refinement config intentionally points to the recorded
fallback initial dynamics checkpoint.  A byte-identical regeneration therefore
requires the corresponding initial stage under
`model_full_nyquist.json`; a new clean study should instead declare one uniform
initialization rule before fitting and report that it is a new reproduction,
not claim the old hashes.

After all folds exist, `training/scripts/export_hust_release.py` assembles the
deployable checkpoint.  It writes source centroids and support fields; no held
bearing value is fitted.

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

## Reproducibility boundary

These programs expose the actual training lineage; they do not make the HUST
table prospectively independent.  The four HUST targets were inspected during
method development.  Released weights remain the authoritative exact
reproduction artifacts, while a newly trained run is a new stochastic
reproduction unless its hashes match.
