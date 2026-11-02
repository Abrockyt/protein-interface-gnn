# Official DB5 benchmark experiment

This is the project's **primary experiment**. It is a modern PyTorch
reimplementation of the node-average idea in Fout et al. (NeurIPS 2017),
using the [authors' released data](https://zenodo.org/records/1127774).
It is not the original TensorFlow implementation.

## Run

From the project root:

```powershell
python -m pip install -r requirements.txt
python paper_benchmark/download.py
python paper_benchmark/benchmark.py --epochs 35 --patience 7
python paper_benchmark/analyze.py
python paper_benchmark/test_evaluation.py
python paper_benchmark/audit_metrics.py
```

`download.py` verifies the exact MD5 checksums listed by the data authors.
The compressed files are about 98 MB total. Their Python 2 pickle format is
loaded by `dataset.py` with a restricted list of NumPy constructors. Only load
files obtained from the documented authors' record.

## What the code does

1. Loads 175 development and 55 official test complexes. These feature
   vectors come from **unbound** protein structures. The published pair
   labels derive from bound complexes using the paper's 6 Å heavy-atom rule.
2. Makes a fixed random 140/35 development train/validation split. The
   authors' original difficulty-stratified split is not encoded in the
   released file, so this is an adaptation.
3. Trains a center-only neural network and a two-layer node-average GCN.
   Both use the same 70 node features and symmetric residue-pair head.
4. Chooses the epoch by median validation ROC AUC. Development labels have
   sampled negatives at a 10:1 ratio, so validation AP does not have the
   same prevalence as the full official test set.
5. Retrains each model on all 175 development cases for its chosen number
   of epochs and evaluates every **provided** pair label in all 55 test
   complexes. The paper's metric is median ROC AUC across complexes.

## Outputs

`results/benchmark_metrics.json` contains split IDs, selection history,
per-complex metrics, pooled metrics, and test summary. `results/gcn.pt` and
`results/mlp.pt` are the trained models. The main report explains what the
numbers mean and how they differ from the paper's published experiment.

`audit_metrics.py` loads the saved checkpoints, caches pair-level logits and
scores, and independently verifies AUC/AP formulas. `metric_audit.json` and
`audited_per_complex.csv` add top-K retrieval, partial AUC, confusion counts,
F1, MCC and calibration diagnostics. The original results are preserved.
See `../METRIC_AUDIT.md` and `../METRICS_GUIDE.md` for interpretation.

The earlier `protein_gnn.py` / `results/metrics.json` experiment used 24
bound structures and a different 5 Å label definition. Its scores should
not be presented as a replication of Fout et al.
