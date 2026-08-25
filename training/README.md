# DV-HOKO training source

This directory contains the source-only programs that generated the model
families packaged by the repository.  It is deliberately separate from the
small inference package under `src/dvhoko`.

Install both packages from the repository root:

```bash
pip install -e .
pip install -e "training[dev]"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The HUST release is staged rather than a single end-to-end checkpoint fit:

1. deterministic raw-waveform envelope cache;
2. source-only Koopman--Mori forecast training and operation metric;
3. whole-source pseudo-held-bearing refinement of the dynamical trunk;
4. frozen-trunk full-mode health attention;
5. export of the source-induced memories and learned weights.

The Final project model has a separate native-spectrum adapter but calls the
same public `dvhoko.model.DualViewKoopmanMoriField`.  Its training command seals
the 2024 source checkpoint before opening the 2026 target path.

See `../docs/TRAINING.md` for exact commands and limitations.
