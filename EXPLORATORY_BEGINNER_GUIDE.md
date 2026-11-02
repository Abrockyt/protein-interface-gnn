# Beginner guide: what this project actually does

## Five words you need

**Protein:** a molecule built from a chain of amino acids.

**Residue:** one amino-acid unit in a protein. We treat a residue as one graph
node, even though it has several atoms.

**Interface:** the part where two proteins touch.

**Graph:** a set of nodes and edges. An edge in this project joins two nearby
residues *inside the same protein*.

**GNN:** a neural network that repeatedly passes information along graph
edges. A GCN is one particular kind of GNN.

## The task in one tiny example

Imagine protein A has residues A1, A2, A3 and protein B has B1, B2. There
are six possible pairs: A1–B1, A1–B2, A2–B1, A2–B2, A3–B1, A3–B2. The model
gives every pair a score. If a pair's atoms touch in the known 3D complex,
its correct label is 1; otherwise it is 0.

The known complex is used as an **answer key** while training and testing.
The network sees the separate protein graphs, not the answer-key distance
between the two proteins.

## Why do we use Stanford CS224W ideas?

The course teaches a standard progression: define a graph, encode node
features, pass messages through neighbors, and make predictions from the
resulting node embeddings. This project implements that progression directly:

1. **Graph:** one graph for each protein.
2. **Features:** which amino acid each residue is, plus how many close
   neighbors it has.
3. **Message passing:** two GCN layers mix nearby residues' information.
4. **Prediction:** combine one embedding from A with one from B and predict
   a contact score.

The GCN is trained by comparing its scores with known contact labels, then
adjusting its weights to reduce binary cross-entropy loss.

## What happens when you run the code?

From this folder in PowerShell:

```powershell
python -m pip install -r requirements.txt
python protein_gnn.py prepare
python protein_gnn.py train --epochs 30
python plot_results.py
```

`prepare` downloads real protein complex structures. `train` builds graphs,
creates labels, trains the MLP and GCN, selects the best epoch on validation,
and scores all pairs in the held-out test complexes. The results appear in
`results/metrics.json`; the chart appears in `results/test_average_precision.png`.

## How to explain the score to your professor

There are 210 true contacts among 797,291 test pairs. A random ranking has
average precision near 0.000263. The GCN gets 0.002544 and the MLP gets
0.001264. So graph message passing improves the ranking on this particular
test set, but the GCN's actual precision is only 0.00453 at the validation
chosen threshold. That means the model is a learning demonstration, not a
dependable tool for selecting biological contacts.

If asked for the biggest methodological weakness, say: **the separate protein
graphs come from bound complex structures, and the dataset is small**. A
harder and more realistic next experiment would use unbound structures and a
larger sequence-disjoint dataset.
