# Project plan: Protein Interface Prediction with a GCN

## 1. Translate the assigned topic into a precise task

Given the separate three-dimensional structures of two proteins, predict which
residue pairs will be in contact when the proteins bind. A residue is one amino
acid; a contact label comes from a known experimental complex. This is pairwise
binary classification, not protein folding or binding affinity prediction.

## 2. Build an educational, reproducible dataset

Use a fixed subset of protein complexes from the public Docking Benchmark 5
collection mirrored by LightDock. Parse only the two protein chains in each
reference complex. Create one graph per protein: amino-acid residues are nodes,
and residues with nearby C-alpha atoms within the same protein are edges.
Generate positive pair labels from inter-protein heavy-atom distance below 5 Å.
The cross-protein distance is used only to form labels and is never an input.

## 3. Implement the model and comparison

Implement a two-layer graph convolutional network (GCN) in PyTorch, following
the normalized neighbor aggregation taught in Stanford CS224W. Encode each
protein independently, then score every candidate residue pair with an MLP.
Compare against an MLP that sees exactly the same node features but performs no
graph message passing.

## 4. Evaluate honestly

Split by whole protein complex before training. Select hyperparameters and a
decision threshold on validation data; report final average precision, ROC AUC,
precision, recall, F1, and precision at the number of true contacts on held-out
test complexes. Show class prevalence and per-complex results. Record random
seed, structure IDs, model settings, and limitations (small dataset and possible
protein-family similarity across complexes).

## 5. Deliver a complete solo-project package

Provide executable code, dependency list, data-download script, training and
evaluation commands, a plain-English explanation, a presentation outline, and
references to the assigned PDF, Stanford CS224W material, and dataset sources.
Run the pipeline and include measured results rather than invented scores.
