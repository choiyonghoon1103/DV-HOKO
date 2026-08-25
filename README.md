# DV-HOKO

Compact pretrained release of **Dual-View Health–Operation Koopman Observer**.
The primary research setting is **source-only domain generalization (DG)**:
the class ontology is known, but the target bearing is excluded from fitting.

The repository contains only the deployable model, immutable configurations,
evaluation entry points, released weights, reproduced results, tests, and two
short technical documents.

```text
configs/       immutable HUST and Data_final inference contracts
docs/          method definition and DG audit
scripts/       evaluation entry points
src/dvhoko/    model, adapters, readouts, data readers, inference
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

The target `fs` value is an observed deployment covariate used for angular
resampling; it is not fitted or averaged with target data. No target label,
target statistic, or target gradient is used.

## Data_final project inference

```bash
python scripts/evaluate_final.py \
  --data /path/to/Data_final/2026 \
  --device cuda:0
```

The mixed file is processed in original order with one file-start reset and no
target adaptation. Data_final is retained as a project deployment, not as the
paper's DG benchmark.

## Reproduced results

HUST leave-one-bearing-domain-out:

| held bearing | final BAcc | prefix BAcc | class-balanced prequential NLL |
|---|---:|---:|---:|
| 6205 | 1.000000 | 1.000000 | 0.071072 |
| 6206 | 1.000000 | 0.958333 | 0.132321 |
| 6207 | 1.000000 | 0.958333 | 0.164043 |
| 6208 | 1.000000 | 0.991667 | 0.165750 |
| mean | **1.000000** | **0.977083** | **0.133297** |

`prefix BAcc` averages correctness over every causal prefix, not only the last
prediction of each record. It therefore exposes startup/recovery errors hidden
by final-record BAcc.

Data_final 2026 mixed stream: second BAcc `1.000000`, packet-weighted BAcc
`1.000000`, and second class-balanced NLL `0.004736`.

These targets were inspected during iterative development. The table is exact
reproducibility evidence, not a new prospective confirmation. See
[docs/DG_AUDIT.md](docs/DG_AUDIT.md) before making generalization claims.

## Method

The complete computation is summarized in [docs/METHOD.md](docs/METHOD.md).
The short version is:

1. convert each completed second into generic subband-envelope trajectories;
2. map time to measured shaft angle and retain the full angular Nyquist grid;
3. learn a causal Koopman–Mori neural field;
4. read health with learned full-mode attention;
5. read fault operation with a source-support-conditioned query-adaptive metric;
6. accumulate local evidence causally.

This is a learned neural dynamical model with explicit physical inductive bias,
not a fault-frequency rule table. It remains a known-class DG method, not an
unseen-class/open-set method.
