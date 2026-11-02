# Beginner guide: understanding the protein interface GNN

## The biology in plain words

A protein is a chain of amino acids. One amino acid in the chain is called a
**residue**. When two proteins bind, a small surface region on each partner
touches the other. That touching region is the **interface**. This project's
prediction is about **pairs**: one residue from protein A and one from B.

## What is a graph neural network here?

Treat each residue as a **node**. Link it to its nearby residues inside the
same protein with **edges**. The GNN starts with 70 precomputed properties
for each residue. In a graph convolution, each residue receives information
from its spatial neighbors. After two rounds, its learned vector describes
some of the surrounding protein patch. A small classifier combines a vector
from A and one from B to score whether the pair contacts.

This follows the [Stanford CS224W](https://web.stanford.edu/class/cs224w/)
pattern: graph representation → message passing → node embeddings →
pairwise prediction → held-out evaluation.

## Where does the answer key come from?

The [Fout et al. research dataset](https://zenodo.org/records/1127774)
provides separate **unbound** structures as inputs and labels calculated
from the experimentally observed **bound** complex. A pair is positive if
non-hydrogen atoms are within **6 Å** in that bound structure. The bound
cross-protein distance is an **answer key**, never an input feature.

## Why compare an MLP with the GCN?

The MLP has the same residue features and pair classifier but does not read
neighboring residues. If the GCN performs better under the same evaluation,
that supports the value of graph message passing. It does not prove that
every GNN design is better in every setting.

## What did the experiment find?

On 55 official test complexes, the MLP median ROC AUC is **0.843** and the
GCN's is **0.891**. The GCN ranks contacts better on **42 of 55** complexes.
Its median average precision rises from **0.0127** to **0.0182**. That AP
still looks numerically low because only **0.0982%** of the evaluated pairs
are actual contacts. ROC AUC 0.891 does **not** mean 89.1% of predicted
contacts are correct.

The paper reports a similar range of AUCs, but this project is a smaller,
single-run reimplementation with a different validation split and classifier.
The full explanation and limitations are in `REPORT.md`.

The audit adds a crucial qualification: the GCN finds **27** true contacts
among its top ten per complex (550 recommendations total), while the MLP
finds **28**. So the GCN's improved overall ranking does not make its top-ten
list better. At a raw score threshold of 0.5 it also creates many false
positives. Read `METRICS_GUIDE.md` to understand this difference and why the
sigmoid scores are not validated contact probabilities.

## Run it yourself

Open PowerShell in this project folder and run:

```powershell
python -m pip install -r requirements.txt
python paper_benchmark/download.py
python paper_benchmark/benchmark.py --epochs 35 --patience 7
python paper_benchmark/analyze.py
python paper_benchmark/verify.py
python paper_benchmark/test_evaluation.py
python paper_benchmark/audit_metrics.py
```

The download is about 98 MB. The benchmark ran on CPU. The complete run may
take several minutes. Results appear in `paper_benchmark/results`.

## One-minute explanation for your professor

“I followed Fout et al.'s protein-interface GCN paper. I used their official
Docking Benchmark 5 features from unbound proteins and contact labels from
bound complexes. My PyTorch model has two shared node-average graph layers
and a symmetric pair classifier. I compared it with a no-convolution MLP
using the same features. I selected epochs on development complexes and
evaluated on the authors' 55 test complexes. The GCN improved median
per-complex ROC AUC from 0.843 to 0.891, but the rare-contact precision
is still low. Its top-10 precision does not improve. This is one run rather
than an exact replication, and the audit reports the weak results too.”
