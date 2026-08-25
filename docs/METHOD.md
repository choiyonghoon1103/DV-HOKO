# Method

## Problem

For HUST, a domain is a bearing location (`6205`–`6208`) and the known labels
are `N`, `I`, `O`, and `B`. In each fold, three bearings are labeled sources and
the fourth is an unseen target domain. The target contributes neither labels
nor fitted statistics. This is domain generalization, not unseen-class
zero-shot learning.

## Observation and angular clock

Every completed 1-second waveform block is split into 32 uniform Fourier
subbands. Their analytic envelopes are resampled against the record's observed
shaft speed `fs` at 64 samples per revolution. The adapter uses no hand-picked
fault-frequency or bearing-geometry order. The model then forms 4-revolution
right-aligned windows with a 0.5-revolution hop.

The resolved mode grid is the complete angular Nyquist support
`0.25, 0.50, ..., 32.00` order. This removes the earlier manually cropped mode
range, but angular resampling itself remains an explicit physical inductive
bias and requires speed measurement at deployment.

## Learned dynamical field

A balanced attention mixer maps 32 generic subbands to four learned carrier
paths. For each angular mode, the network receives:

- its complex resolved trajectory;
- the same trajectory under exact rotational Koopman transport;
- the one-step Mori innovation between observation and transported state.

A causal Transformer summarizes the trajectory. A shared order embedding and
neural decoder produce an order-resolved 12-dimensional field. Training first
uses source-only multi-horizon closure prediction to establish a dynamical
representation. It then refines that representation with whole-source
pseudo-held-bearing classification only. The final refinement deliberately
does not retain an auxiliary forecast loss: the released ablation found that
forecast retention worsened the held-domain online risk, while dynamics
pretraining itself remained necessary. Thus the final trunk is a learned
dynamical architecture, although it is not purely self-supervised or a
discovered physical law.

## Two learned readouts

The model decomposes a four-class label into:

1. **health state:** `N` versus fault;
2. **fault operation:** conditional `I/O/B` identity.

Both readouts use the same frozen Koopman–Mori trunk.

The health decoder applies multi-head attention across all 128 state modes and
compares the decoded state to binary source centroids. The operation decoder
uses source fault fields to learn a support-conditioned reliability prior over
modes, then adjusts those weights from the current query field without changing
any parameter. A learned shared Mahalanobis transform compares the query with
the three source-induced operation memories.

This factorization is a methodological assumption, not a discovered universal
class ontology. A new task must supply labels for which a health/operation
factorization is meaningful, or replace the readout while retaining the trunk.

## Causal evidence

Each second yields one binary health logit and three conditional operation
logits. Deterministic prefix summation accumulates evidence from the record
start. The four-class probability is

`[P(normal), P(fault) P(I|fault), P(fault) P(O|fault), P(fault) P(B|fault)]`.

No future sample, target fit, or prediction-conditioned reset is used. `final
BAcc` evaluates only the last prefix; `prefix BAcc` averages all causal
prefixes and is the stricter online measure.

## Data_final

Data_final uses the same neural Koopman–Mori core and a native spectrum adapter,
with a readout trained for its binary background/leak deployment task. HUST and
Data_final therefore share the dynamical model class while retaining
observation- and task-appropriate adapters and readouts.
