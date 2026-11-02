# Presentation outline and speaking notes

Use 7–9 minutes. The format follows a typical CS224W project explanation:
motivation → graph representation → GNN design → objective → evaluation → findings.

## Slide 1 — Title and one-sentence goal

**Protein Interface Prediction with Graph Neural Networks.**
“I predict which amino-acid residue pairs across two proteins touch when the
proteins form a complex.” This is a solo project based on the Protein Interface
Prediction option in the course handout.

## Slide 2 — Why this is a graph problem

Draw two separate protein graphs. A node is one residue. Within each graph,
an edge means two residues are spatially close. A dotted line between the
proteins is the pair we want to classify. Explain that far-apart sequence
positions can still be neighbors in 3D.

## Slide 3 — Dataset and label

Show one complex ID and its two chains. Explain the **5 Å heavy-atom contact
rule**. State clearly: “The model does not receive the cross-chain distance.
I calculate it only to create the answer labels from the known complex.”

## Slide 4 — GCN message passing

Show `H_next = ReLU(D^-1/2 (A+I) D^-1/2 H W)`. In plain English: each residue
mixes its own features with nearby residues' features, then learns a new
representation. Two layers mean two rounds of neighboring information.

## Slide 5 — Pair scoring and baseline

Show `protein A graph → GCN → residue embeddings` and the same for B. A pair
MLP turns two embeddings into a score. Baseline: same features and score head,
but no graph aggregation. This tests whether message passing helps.

## Slide 6 — Evaluation design

16 train / 4 validation / 4 test complexes. Explain why whole complexes,
not random residue pairs, must be separated. All pairs are scored at test
time. Only 210 out of 797,291 test pairs are contacts, so accuracy is misleading.
Average precision is the main metric.

## Slide 7 — Results

Show `results/test_average_precision.png`. GCN AP = 0.00254; MLP AP =
0.00126; random-ranking reference = 0.000263. Explain that the GCN improves
ranking but that practical precision is still poor.

## Slide 8 — What I learned and next experiment

Graph structure adds useful local context, but small data and extreme class
imbalance make the task hard. Next steps: more complexes, sequence-family
split, unbound inputs, and richer partner-only residue features.

## Likely viva questions

**Why use a GCN?** Protein residues have both sequence identity and spatial
neighbors. A GCN learns from both through message passing.

**What is a node, edge, and label?** Node: amino-acid residue. Edge: nearby
residue in the *same* protein. Label: whether a pair across proteins touches
within 5 Å in the known complex.

**Why not use cross-protein distance as input?** It would reveal the label
immediately; that is data leakage.

**Why is AP small?** Only 0.0263% of test pairs are contacts. AP must be
interpreted against that prevalence, and low absolute precision remains a
real limitation.

**What does this model not solve?** It does not fold proteins or predict the
binding orientation of two unbound proteins.
