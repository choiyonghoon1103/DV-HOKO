# DV-HOKO

**Dual-View Health–Operation Koopman Observer**.
The primary research setting is **source-only domain generalization (DG)**:
the class ontology is known, but the target bearing is excluded from fitting.

The repository contains the complete released neural architecture, the staged
source-only training programs, deployable inference code, immutable
configurations, released weights, reproduced results, and contract tests.  The
training and inference packages instantiate the same
`DualViewKoopmanMoriField`. The deployment model is not a reduced surrogate.

```text
configs/       immutable HUST and Data_final inference contracts
docs/          method definition and reproduction guide
scripts/       evaluation entry points
src/dvhoko/    model, adapters, readouts, data readers, inference
training/      source-only training modules, configs, scripts, and tests
tests/         interface, weight, and target-exclusion checks
weights/       four HUST LODO folds and one Data_final checkpoint
results/       reproduced aggregate metrics
```

## Install

```bash
pip install -e .
python scripts/verify_release.py
```

## HUST domain generalization

Each checkpoint uses three source bearings and holds out the named target
bearing. The raw MAT file must contain both `data` and the measured shaft-speed
field `fs`.

```bash
python scripts/evaluate_hust.py \
  --data /path/to/HUST \
  --bearing 6205 \
  --device cuda:0
```

## Data_final project inference

```bash
python scripts/evaluate_final.py \
  --data /path/to/Data_final/2026 \
  --device cuda:0
```

## Reproduced results

HUST leave-one-bearing-domain-out:

| held bearing | final BAcc | prefix BAcc | class-balanced prequential NLL |
|---|---:|---:|---:|
| 6205 | 1.000000 | 1.000000 | 0.063119 |
| 6206 | 1.000000 | 0.966667 | 0.132158 |
| 6207 | 1.000000 | 0.983333 | 0.130174 |
| 6208 | 1.000000 | 0.991667 | 0.156905 |
| mean | **1.000000** | **0.985417** | **0.120589** |

`prefix BAcc` averages correctness over every causal prefix, not only the last
prediction of each record. It therefore exposes startup/recovery errors hidden
by final-record BAcc.

Data_final 2026 mixed stream: second BAcc `1.000000`, packet-weighted BAcc
`1.000000`, and second class-balanced NLL `0.004736`.


## Method

The complete computation is summarized in [docs/METHOD.md](docs/METHOD.md).
The short version is:

1. convert each completed second into generic subband-envelope trajectories
2. map time to measured shaft angle and retain the full angular Nyquist grid
3. pretrain a causal Koopman–Mori neural field by source forecasting
4. refine that field with whole-source pseudo-held-bearing classification
5. read health with learned full-mode attention
6. read fault operation with a source-support-conditioned query-adaptive metric
7. accumulate local evidence causally.

This is a learned neural dynamical model with explicit physical inductive bias,
not a fault-frequency rule table. It remains a known-class DG method, not an
unseen-class/open-set method.
