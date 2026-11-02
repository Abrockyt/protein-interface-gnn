# Understanding and defending the evaluation metrics

## What the experiment is measuring

There are 55 test complexes. Each has a list of candidate residue pairs and
known contact/noncontact labels. The two saved models give each pair a score.
The audit checks whether these scores put real contacts ahead of noncontacts.
It also checks how useful a short list of predicted contacts would be.

## Why several measures are needed

The main paper reports **median per-complex ROC AUC**, so this is our main
paper-aligned ranking measure. **AP** describes the contact ranking under
severe imbalance. **Top-K precision and recall** answer the practical question
of inspecting a limited list of pairs. **F1 and MCC** describe a classifier
only after a threshold has been specified. No single metric answers all four
questions, and a larger number of metrics does not itself make a study sound.

### ROC AUC: overall ordering

ROC AUC is the probability that a randomly selected positive pair receives a
higher score than a randomly selected negative pair, with half credit for a
tie. A random ranking scores 0.5. We compute it for each complex and take the
median: MLP **0.8432**, GCN **0.8913**. This does not mean that 89.13% of
GCN predictions are correct. It describes ordering, not contact precision.

### AP: precision through the ranking

Without score ties, average precision is the average of the precision values
at the positions where true contacts appear. With ties, we use distinct score
thresholds. Higher AP means positives are generally concentrated earlier in
the list. The median complex AP is **0.01274** for the MLP and **0.01825**
for the GCN. Pooled AP is **0.00835** and **0.00928**, respectively.

AP is not the same as the trapezoidal area under a precision-recall curve:
the latter connects successive points linearly. We report both conventions
with different names. AP is also not the precision of one selected list, so
we should not describe AP 0.00928 as “0.928% of predictions are correct.”

### Top-K: the pairs you would actually inspect

Precision at K = contacts found among the K highest scores / K.
Recall at K = those contacts / all known contacts in that complex.
We use K=10 and 50, plus budgets based on partner length: `L=min(n_ligand,
n_receptor)` and `L/5=max(1,floor(L/5))`. These are explicitly defined project
budgets; they do not claim to reproduce every published top-L convention.
When a score tie crosses the cutoff, credit is the expected value under
uniform tie breaking, so the original row order cannot give positives an advantage.

The practical outcome is weaker than the AUC headline. The GCN finds **27
contacts among 550 top-10 recommendations**; the MLP finds **28**. This is
4.91% versus 5.09% mean precision. The GCN has no true contact in its top ten
on **39 of 55 complexes**. We cannot claim it produces a better top-10 list.
Mean precision at L/5 is 5.05% versus 4.48%, but the paired uncertainty
interval for that difference includes zero.

### Threshold metrics: a diagnostic, not a validated operating rule

At a threshold of 0.5, the GCN produces 2,702 true positives, 475,316 false
positives, 2,169 false negatives and 4,478,130 true negatives. Therefore:

- Precision = TP/(TP+FP) = **0.565%**.
- Recall/sensitivity = TP/(TP+FN) = **55.47%**.
- Specificity = TN/(TN+FP) = **90.40%**.
- F1 = 2TP/(2TP+FP+FN) = **0.01119**.
- MCC uses all four confusion counts = **0.04869**.

These are numerically well-defined but the threshold itself has not been
validated for deployment. The training data sampled negatives and the loss
upweighted positives. Consequently the sigmoid score is not a calibrated
contact probability. We did not choose a threshold to maximize test F1.

### Why accuracy is misleading

Only 4,871 of 4,958,317 test pairs are contacts. Predicting “noncontact” for
every pair gives **99.9018% accuracy** but misses every contact. It therefore
has recall 0, F1 0 and MCC 0. This is why accuracy is a diagnostic rather
than the headline result.

For a small example, suppose 10 out of 10,000 pairs are contacts. A classifier
predicts 100 contacts, of which six are correct: TP=6, FP=94, FN=4,
TN=9,896. Precision is 6%, recall is 60%, and F1 is about 0.109. Its accuracy
is 99.02%, below the always-negative classifier's 99.9%, even though it finds
six contacts. AUC cannot be recovered from this one confusion matrix; it
requires the scores over their full ranking.

## Why median, mean and pooled scores differ

Per-complex measures calculate a score for every protein complex. A macro
mean or median gives each complex equal influence. A pooled score concatenates
all pairs; a complex with many candidate pairs then contributes much more.
The median also hides difficult failures: the GCN's mean complex AUC is
0.8605 while its median is 0.8913. Both are recorded, so the result does not
depend on selecting the most flattering aggregation.

## How correctness was verified

The saved model files were loaded again and all 55 test cases rescored.
Every original per-complex AUC, AP and legacy top-L value reproduced exactly.
Independent rank-sum AUC and threshold-group AP formulas agreed with
scikit-learn within **3.33e-16**. Hand-calculated perfect, reverse and tied
rankings were tested. Input labels, pair indices, duplicate pairs, dataset
checksums and split IDs were checked. The audit stores pair-level logits,
scores and labels, file hashes, a per-complex CSV and all metric results.

This verifies numerical evaluation and data provenance. It does not establish
that every published biological annotation is error free, that proteins from
the same family never cross the split, or that a single training seed is robust.

## Questions to answer honestly in the viva

**Why does AUC improve while top-10 precision does not?** AUC rewards correct
ordering throughout the entire contact/noncontact comparison. The top ten
tests only the extreme head of the ranking; improvement elsewhere need not
improve those ten positions.

**Did you pick metrics to make the GCN win?** The audit reports measures on
the same frozen checkpoints, including metrics where the GCN loses. The new
diagnostics are post-hoc and are not presented as preregistered hypotheses.

**Are the scores probabilities?** No probability calibration has been validated.
The sigmoid values are treated as ranking scores; the raw 0.5 decision rule
is only a diagnostic.

**Is the difference statistically certain?** The complex-bootstrap AUC
interval is positive on all 55 cases, conditional on the saved models. On
the fixed 31-case historical subset it includes zero. Neither interval
accounts for retraining seeds or family dependence, and neither creates a
new blind test set.

**What is a defensible conclusion?** The GCN improves overall contact ranking
on the reported test set, but does not improve top-10 retrieval and is not
a reliable contact classifier at its raw 0.5 threshold. More seeds, an
independent natural-prevalence validation set and a fresh external holdout
are needed before making stronger claims.
