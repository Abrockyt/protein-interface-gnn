# Presentation and viva notes

Aim for 8–10 minutes. Show the editable slides in
`submission/Protein_Interface_GNN_Presentation.pptx` and use these notes.

## Slide 1 — Question and paper

“My solo project is protein interface prediction. I ask whether graph
message passing helps identify contact pairs across two proteins. My main
reference is Fout et al., NeurIPS 2017.”

## Slide 2 — What is predicted?

Draw two proteins. Circle one amino-acid residue on each. We score whether
those two residues contact when the proteins bind. This is not protein
folding or whole-protein interaction classification.

## Slide 3 — Data and labels

The authors' DB5 release supplies unbound partner features and labels from
bound complexes. A positive label means a 6 Å heavy-atom contact. There are
175 development and 55 official test complexes. The model cannot see the
cross-protein distance used to define the label.

## Slide 4 — Graph representation and convolution

One residue is one node; its 20 spatial nearest residues are neighbors. Each
node has 70 precomputed features. Equation: `ReLU(Wc h_i + Wn mean(h_j))`.
The center and neighbor effects have separate weights. Two layers allow
information from a broader local patch.

## Slide 5 — Pair classifier and baseline

The same graph encoder processes both proteins. Combine two residue vectors
using their sum, absolute difference, and elementwise product. A dense layer
scores the pair. The baseline uses the same input and pair classifier but
removes neighbor aggregation.

## Slide 6 — Honest experimental design

Use a fixed 140/35 split of the 175 development complexes. Select epoch by
median validation ROC AUC. Retrain on all 175 and evaluate on the released
55-complex test. Test has 4,958,317 labeled pairs but only 4,871 contacts.
We report median complex ROC AUC and AP rather than accuracy.

## Slide 7 — Results

The no-convolution MLP gets **0.843** median complex ROC AUC; the GCN gets
**0.891**. Median AP is **0.0127** versus **0.0182**. The GCN wins on 42/55
complexes. The full-set bootstrap interval for the difference of medians is
[0.0148, 0.0655]. But top-10 precision is **4.91% for the GCN** and
**5.09% for the MLP**. Higher AUC does not establish better early retrieval.

## Slide 8 — Why several metrics are needed

ROC AUC measures contact/noncontact ordering across the whole ranking. AP
emphasizes the precision-recall ranking; it differs from trapezoidal PR area.
Precision at K describes the finite list a researcher might inspect. The
GCN has 27 true contacts among 550 top-10 recommendations; the MLP has 28.
The metrics answer different questions, so show the GCN's weaker results too.

## Slide 9 — Fixed-threshold diagnostic

At raw score 0.5, GCN TP=2,702, FP=475,316, FN=2,169, TN=4,478,130.
Precision is 0.565%, recall 55.47%, F1 0.01119, and MCC 0.04869. Training
negative sampling and weighting mean sigmoid scores are uncalibrated.
The threshold has not been validated. An always-negative classifier gets
99.9018% accuracy but detects no contact, so accuracy is misleading here.

## Slide 10 — Audit evidence and limitations

The paper reports 0.882 ± 0.007 for its two-layer node-average method across
ten runs. My model differs, and my result is one run. The earlier exploratory
project used 24 of the test complexes, so the 55-set is not wholly blind.
On the 31 not previously explored, GCN AUC is 0.853 versus 0.819, with a
bootstrap interval including zero. The metric audit independently reproduces
the saved AUC and AP values. MLP and GCN have different parameter counts.
Next steps: more seeds, a fresh external holdout, family-aware splits and
independent validation/calibration. A correct metric calculation alone does
not establish practical contact-prediction quality.

## Questions your professor may ask

**Why a graph?** Spatially neighboring residues can be far apart in the
amino-acid sequence. The graph exposes that local 3D context.

**What is shared between the two proteins?** Both pass through the same
encoder weights. This reduces parameters and reflects the arbitrary choice
of ligand and receptor labels.

**What does the convolution equation mean?** A residue learns from its own
features and the average of nearby residues. The two weight matrices let
the model treat those sources differently.

**How do you prevent data leakage?** Development and official test PDB IDs
are disjoint. Bound cross-protein distances define labels only; the inputs
are unbound partner features. Earlier project exposure to 24 test complexes
is disclosed and analyzed separately.

**Why use median AUC?** It matches the paper's main metric and gives each
complex one vote rather than letting very large complexes dominate.

**Is 0.891 the same as the paper's result?** No. The paper uses different
model sizes, hyperparameter search, validation stratification, and ten runs.

**Why is AP small?** Contacts are extremely rare among all residue pairs.
ROC AUC can be strong while precision is modest. The project reports both.

**Can higher AUC coexist with worse top-10 precision?** Yes. AUC compares
all positive-negative orderings. Improving many of those orderings can leave
the very first ten positions unchanged or worse. That occurs in this run.

**Why not optimize F1 on the test labels?** It would select an operating rule
using the answers on the evaluation set. We report the raw 0.5 threshold only
as a diagnostic and require independent validation to select a useful rule.
