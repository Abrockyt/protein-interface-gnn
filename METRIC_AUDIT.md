# Metric audit and evaluation rationale

## Audit outcome

Both saved checkpoints were evaluated again on the exact released labels, with no
retraining or threshold optimization on the test set. Original ROC AUC and AP
values reproduce: maximum per-complex absolute errors are
0 and
0. AUC and AP were also recomputed from
independent mathematical formulas, including tied scores; maximum discrepancy
from scikit-learn is 3.33e-16.

The headline ranking numbers were numerically correct. The old evaluation was
incomplete for judging whether the highest-ranked contacts are useful. It also
needed explicit tie handling, score-calibration warnings, and a clearer
distinction between AP and trapezoidal PR area.

**The practical conclusion must include retrieval:** GCN top-10 precision is
0.04909, versus 0.05091
for the MLP: 27.0 versus 28.0
expected true contacts among 550 recommendations
(ten per complex). The GCN finds no contact in the top ten on
39 complexes, compared with 36
for the MLP. GCN top-10 precision is lower; its paired interval is
[-0.02727272727272726, 0.021818181818181827].
The GCN's partial ROC AUC at low FPR is lower. These measures must
be considered alongside the overall ROC AUC, rather than declaring a uniform
improvement from the headline score alone.

## What was checked

- Both compressed datasets match the authors' published MD5 hashes.
- Raw labels contain exactly -1 and +1; +1 is the contact class. Pair indices
  are within the corresponding ligand/receptor feature arrays. No labeled pair
  repeats inside a complex.
- The fixed 140/35 development split and the 55 official test IDs match the
  saved experiment. Development and test IDs are disjoint.
- The test includes 4,958,317 released pairs and
  4,871 contacts. Unlisted Cartesian-product pairs are excluded,
  rather than silently converted to negatives.
- Exact identical feature-and-neighborhood representations across development
  and test: 0. This is an exact-array
  check, not a sequence-homology audit.
- The biological 6 Å contact definition comes from the data authors. This
  audit verifies the released labels and their use, not a fresh reconstruction
  of bound/unbound residue correspondences from experimental structures.
- Saved model/data SHA-256 hashes, package versions, and pair-level outputs
  make the recomputation traceable in `results/metric_audit.json` and
  `results/audit_predictions`.

## Why these metrics are included

| Metric | Question answered | Limitation |
| --- | --- | --- |
| Median complex ROC AUC | Does a contact usually rank above a noncontact within a typical complex? | Paper-comparable ranking; says little about precision at the top. |
| AP | How well are contacts concentrated near the top across the full ranking? | Depends on contact prevalence; compare models on identical labels. |
| Trapezoidal PR area | What is the area under the linearly joined PR points? | A different numerical convention from AP; never call them interchangeable. |
| Precision/recall at 10, 50, L/5, L | If we inspect a limited number of pairs per complex, how many contacts do we recover? | K is a stated budget; L=min(partner residue counts), L/5=floor(L/5), at least 1. |
| Top-K enrichment | How much better is top-K precision than random selection in that complex? | Relative lift can look large while absolute precision is low. |
| Partial ROC AUC at FPR <=1% | Does ranking remain useful when false-positive rates must be small? | Standardized scikit-learn convention; chance=0.5. |
| Confusion counts, precision, recall, F1, MCC | What happens at a stated score threshold? | Here 0.5 is diagnostic only; no deployment threshold has been validated. |
| Brier score and log loss | Can the weighted sigmoid outputs safely be interpreted as probabilities? | Diagnostic only; outputs were not calibrated on a naturally imbalanced holdout. |

## Recomputed ranking results

Ranking uses raw logits to avoid extra rounding/saturation ties from sigmoid.
The old sigmoid-based metrics are retained separately for reproduction.
The added retrieval budgets and diagnostic metrics were specified for this
post-hoc audit; they are exploratory, not preregistered confirmatory endpoints.

| Measure | MLP | GCN |
| --- | ---: | ---: |
| Median per-complex ROC AUC | 0.843190 | 0.891311 |
| Median per-complex AP | 0.012737 | 0.018246 |
| Median trapezoidal PR area | 0.011289 | 0.017445 |
| Mean precision at 10 | 0.050909 | 0.049091 |
| Mean recall at 10 | 0.006787 | 0.006579 |
| Mean precision at L/5 | 0.044822 | 0.050522 |
| Mean precision at L | 0.033723 | 0.036760 |
| Pooled ROC AUC | 0.844794 | 0.873831 |
| Pooled AP | 0.008346 | 0.009276 |
| Pooled trapezoidal PR area | 0.008321 | 0.009239 |
| Standardized pooled partial ROC AUC (FPR <= 1%) | 0.542150 | 0.541595 |

Top-K precision uses the expected number of contacts under uniform selection
inside a tie at the cutoff. This prevents positive-first input ordering from
deciding which tied predictions receive credit. Each complex contributes
equally to macro means/medians. Pooled metrics rank all test pairs together;
large complexes and score offsets between complexes can change them.

## Fixed-score classification diagnostic

The following uses sigmoid score >=0.5. It is not an independently validated
decision rule and must not be presented as optimal operating performance.

| Model | TP | FP | FN | TN | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 3,923 | 1,437,314 | 948 | 3,516,132 | 0.002722 | 0.805379 | 0.005426 | 0.035546 |
| GCN | 2,702 | 475,316 | 2,169 | 4,478,130 | 0.005653 | 0.554712 | 0.011191 | 0.048693 |

An always-negative classifier has test accuracy
99.901761%, recall 0, F1 0 and MCC 0.
This is why plain accuracy is not the headline metric. Sensitivity, specificity,
balanced accuracy and the confusion counts are retained in the JSON/CSV.

## Probability interpretation

Development pairs were downsampled to about 10 negatives per contact, and the
loss gives positives weight 10. That combination changes the effective training
prior. A sigmoid score of 0.8 is therefore not evidence of an 80% contact
probability under the test prevalence. The audit reports Brier/log-loss diagnostics
without fitting a calibrator on test labels. A useful threshold and calibration
require independent validation data with the natural pair distribution; the
released development files do not provide that distribution directly.

| Score source | Mean score | Brier score (lower is better) | Log loss (lower is better) |
| --- | ---: | ---: | ---: |
| MLP raw sigmoid | 0.319767 | 0.176810 | 0.511639 |
| GCN raw sigmoid | 0.127628 | 0.066811 | 0.214964 |
| Empirical-prevalence constant (diagnostic) | 0.000982 | 0.000981 | 0.007785 |

The constant reference uses the observed test prevalence after the fact. It
illustrates the scale of probability error; it is not a separately trained
deployable baseline. The raw neural scores should be used for ranking until
calibration is evaluated on suitable independent validation data.

## Uncertainty and remaining limits

Paired bootstrap intervals resample complexes, not millions of dependent residue
pairs. Full-set and the fixed 31-case historical subset are in the JSON. These
intervals describe variation across the observed complexes conditional on the
two trained models. They do not measure training-seed variation or establish
sequence-family independence. All 55 cases have now been examined; the
31-case historical subset is not a new blind test. One training seed and prior
test exposure remain study limitations even after the metric code is correct.

The MLP has 21,121 trainable parameters;
the GCN has 29,697. They use matched
feature/head widths, not matched parameter counts. Thus this comparison supports
an association with graph context but does not isolate it from added capacity.

## Reproduce this audit

Run `python paper_benchmark/test_evaluation.py` and
`python paper_benchmark/audit_metrics.py`. The second command checks cached
prediction provenance before reuse; delete that cache to force fresh inference.
The original experiment JSON and checkpoints are preserved.

Sources: [paper metric definition](https://papers.neurips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks.pdf),
[AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
[partial ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html),
[probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
