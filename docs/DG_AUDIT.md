# Domain-generalization audit

## Verdict

The released HUST method is a coherent **known-class, source-only DG** system.
The held bearing is absent from weights, centroids, support fields, and fitting;
the only target-side auxiliary input is the record-observed shaft speed used by
the fixed angular adapter. No direct implementation leakage was found.

It nevertheless has material scientific limitations. The 100% final BAcc is a
strong result on this benchmark, but it does not establish general DG in a new
machine population.

## What the protocol establishes

- Whole-bearing exclusion is aligned with the claimed sensor-location domain.
- Every released fold stores exactly 36 source records from three bearings and
  reports zero held-bearing resources used for fit.
- Source support induces the operation memory; query values only receive a
  forward pass. There is no target optimization or target-statistic update.
- The full angular Nyquist grid avoids manual fault-frequency/order selection.
- All four target folds reproduce with the standalone code.
- A shaft-speed-only nearest-centroid audit obtains `0.25`, `0.25`, `0.333`,
  and `0.25` accuracy on the four held bearings, so `fs` alone does not explain
  the four-class result.

## Material weaknesses

### 1. Retrospective target reuse — critical

All four HUST bearings were repeatedly inspected while the architecture was
developed, including the health-attention change that fixed a known O600
failure. The final folds have computational target exclusion, but they are not
scientifically pristine targets. A genuinely new bearing/machine or sealed
external dataset is required for prospective confirmation.

### 2. Very few independent domains — critical

Each fold learns from only three source domains and evaluates one held domain.
Seconds, loads, and records improve within-bearing precision but do not increase
the number of independent domains. Four successful folds cannot support a
population-level invariance theorem or narrow DG confidence interval.

### 3. Restricted shift family — major

The benchmark changes bearing/sensor location within one dataset family. It
does not demonstrate transfer across machines, sensor technologies, sampling
systems, or arbitrary class-conditional inversions. If a target reverses the
source relation between class and learned dynamics, no source-only method can
identify that reversal.

### 4. Measured-speed dependency — major

Inference consumes target `fs` as an observed covariate. This is legitimate
covariate-conditioned DG, but deployment requires an accurate shaft clock. The
claim is therefore not “target-data-free”; it is “no target fitting or labels.”
Robustness to missing, biased, or out-of-range speed has not been established.

### 5. Known ontology and manual factorization — major

The method assumes the labels decompose into normal/fault health and conditional
I/O/B operation. It cannot discover a new target class and is not open-set or
unseen-class zero-shot learning. The hierarchy is a human modeling assumption,
even though both mappings inside it are learned.

### 6. Source-conditioned memories can still overfit — major

Health centroids and operation support fields are induced from the three source
bearings. Attention and the query-adaptive metric reduce manual mode selection,
but they do not guarantee invariant semantics. With only three sources, the
readouts may fit source-shared acquisition artifacts that happen to persist in
the fourth bearing.

### 7. Fixed physical and temporal choices remain — moderate

One-second blocks, 32 subbands, four carriers, 64 angular samples/revolution,
four-revolution windows, and half-revolution hops are fixed. They are broad
inductive biases rather than fault-specific lookup rules, but they remain human
choices. Robustness to these choices needs sensitivity tests or a fixed
pre-registration in a future study.

### 8. Cumulative evidence can hide startup errors — moderate

Final BAcc is 100%, while mean prefix BAcc is 97.708%. The latter is the more
informative streaming result because it scores every causal prefix. Variable
record lengths also change how much evidence is available, although the
reported aggregation is record/class balanced.

### 9. Probability calibration is unverified — moderate

Class-balanced NLL is favorable, but the output is a conditional decision score
under an equal-class evaluation distribution. It has not been calibrated to
natural deployment prevalence and should not be called a Bayesian posterior.

### 10. Fair contemporary baseline evidence is incomplete — moderate

Perfect accuracy alone does not isolate which component caused DG. A publishable
claim still needs exact-split comparisons against a strong matched ERM/modern
DG baseline and fixed ablations for angular clock, dynamics, learned health
attention, and source-conditioned operation metric. Those comparisons must not
be used to redesign the already inspected target result.

## Safe claim

> DV-HOKO achieved reproducible source-only leave-one-bearing-domain-out
> classification on all four HUST bearings using a shared learned dynamical
> field, source-induced memories, and no held-bearing fitting.

## Claims not supported

- general zero-shot recognition of unseen classes;
- arbitrary unseen-machine or cross-dataset generalization;
- discovery of the true governing physical law;
- independence from target-side speed measurement;
- prospective confirmation or a population-level DG guarantee.

## Next decisive validation

Freeze this repository and evaluate it once on a newly sealed compatible
bearing/machine dataset. Report every target domain separately and preserve the
same observation contract. That experiment is more valuable than further
target-informed architecture iteration on HUST.
