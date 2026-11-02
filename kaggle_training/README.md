# Kaggle GPU experiment

[Open the private Kaggle notebook](https://www.kaggle.com/code/malleabhinay/protein-interface-gnn-validated-gpu-training?scriptVersionId=360528046).

Version 1: **DB5Plus 3 seeds - validated thresholds**. Completed with **Save & Run All**, using Kaggle **GPU T4 x2**, in **385.4 seconds**. The viewer confirmed CUDA and two Tesla T4 devices. The output was downloaded to `cloud_results/` and independently audited.

See [the results explanation](../KAGGLE_RESULTS.md) for the method, tables, limitations and next research step. The ensemble achieved **2.16% precision, 11.94% recall and 3.66% F1** versus **2.43% baseline F1**. These remain low contact-prediction scores. All 4,210 independent audit checks passed.

## What changed

The earlier project uses the Fout 2017 release with 70 features and sampled development negatives. This new experiment uses the publisher's **DB5-Plus** release, richer features, exhaustive validation pairs, three GNN seeds, and a separate threshold-calibration set. Both address residue-pair interface contacts. Because the releases differ in candidate pairs and a few labels, their scores are **not a controlled architecture-only comparison**.

The primary scientific comparison within this notebook is its MLP versus its GNN, using identical DB5-Plus inputs and evaluation labels.

## Plan and fixed protocol

1. Download `final_raw_db5.tar.gz` from [Zenodo](https://zenodo.org/records/5134732) inside Kaggle and verify MD5 `9cbc07672e705f9ba8549168b06e1e06`.
2. Read each protein's unbound CA residue features. Build 20-neighbor intramolecular graphs. Distance features use only neighbors within one chain. Cross-partner pose coordinates are excluded.
3. Use the official 140 training, 35 validation, and 55 test complexes. Fixed seed 224 partitions the validation complexes into 18 for early stopping and 17 for threshold fitting. No random residue-pair split is used.
4. Fit median imputation and z scaling on training proteins only. Append missing-value indicators: 218 node features and 17 edge features.
5. Train a width-128, three-layer MLP baseline and three residual GNN seeds (224, 225, 226). Share encoders across the two partners and use a symmetric residue-pair head. A 0.2-weight auxiliary interface loss supports the encoder; final metrics remain pair-contact metrics.
6. Use AdamW, learning rate 0.0003, weight decay 0.001, dropout 0.25, gradient clipping 5, a validation-driven learning-rate schedule, up to 150 epochs, and patience 20. Each epoch samples up to 512 positives per complex and ten negatives per positive, with balanced pair loss. The held-out sets retain all candidate pairs.
7. Choose checkpoints by **mean per-complex validation average precision**, then choose the F1 logit threshold on the **17 separate calibration complexes**. Retain the fitted checkpoints; do not retrain using the threshold-calibration labels.
8. Evaluate all seeds, the predeclared equal-logit GNN ensemble, and the baseline. Report pooled and per-complex metrics, class balance, confusion counts, and a sequence-pair-group bootstrap interval. Never choose seeds using test scores.

Independent runs use separate T4 GPUs in two waves. They save best checkpoints and histories, then evaluate and export `Protein_Interface_GNN_Kaggle_Results.zip` automatically.

## Data checks completed locally before submission

| Split | Complexes | All candidate pairs | Positive contacts | Contact prevalence |
|---|---:|---:|---:|---:|
| Train | 140 | 9,723,407 | 13,333 | 0.1371% |
| Validation | 35 | 1,347,119 | 2,867 | 0.2128% |
| Test | 55 | 5,224,053 | 4,891 | 0.09362% |

The full candidate matrix is constructed using the publisher's `build_complex_labels` convention: every CA residue pair is negative unless present in `pos_idx`. The archive's sampled `neg_idx` is **not** treated as exhaustive. Checks confirm positive indices map to CA atoms, there are no duplicate positive pairs, stored sampled negatives do not contradict positives, and input paths refer to unbound structures.

Exact full sequence-pair matches do not cross the official splits. Some sequences repeat within train and test. The 55 test complexes have **52 exact sequence-pair groups**; the bootstrap groups these repeated pairs together. This does not establish homology-disjointness or prevent a shared individual partner across sets.

## How to interpret the results

- Precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = 2 TP / (2 TP + FP + FN).
- A threshold is required for these measures. Threshold fitting on a balanced or sampled validation set would misrepresent the natural contact prevalence; this run uses the full calibration matrices.
- ROC-AUC describes ranking. It does not mean the reported percentage of predictions is correct.
- Average precision emphasizes precision and recall under rare positives. It is separately named from trapezoidal PR area.
- Raw accuracy is compared with an always-negative reference. At the test prevalence here, that reference has about 99.906% accuracy but zero contact recall and F1.
- MCC, balanced accuracy, specificity, P@10, per-complex summaries and confusion counts expose different failure modes.
- Sampling and loss weighting mean raw model logits are not calibrated contact probabilities. Threshold calibration here chooses an operating point; it does not perform probability calibration.
- The ensemble and all three individual seeds are reported. No high score is guaranteed by training longer.

**Prior exposure:** the underlying DB5 test structures were examined in earlier project experiments. This is a transparent fixed benchmark rerun, not a new blind holdout. A future external, homology-controlled evaluation is needed for stronger generalization claims.

## Files and verification

- `train.py`: complete data preparation, models, training, threshold fitting and evaluation.
- `Protein_Interface_GNN_Kaggle.ipynb`: self-contained uploaded notebook, including the plan, method, metric explanations, and automated two-GPU job launcher.
- `build_notebook.py`: generates that notebook directly from the source.
- `cloud_results/`: verified Kaggle metrics, predictions, checkpoints, histories and source.
- `audit_downloaded_results.py`: independent arithmetic and ranking checks over all saved test predictions.
- `manifest.json`: local data checks, frozen split IDs, label policy and preprocessing details.
- `gnn_999/`: one-epoch **local preflight only**, not a final result. It verified that data loading, GPU backpropagation, validation, and checkpoint writing execute successfully. This seed is excluded from the Kaggle ensemble.

Preflight checked notebook code compilation and threshold/F1 consistency with hand examples including tied scores. Cloud evaluation checked the symmetric head's swap identity. The completed predictions passed 4,210 independent metric checks, with maximum absolute difference 2.22e-16. The original Fout-release results remain separately documented because their evaluation pair coverage differs.

Version 1 installs `dill==0.4.0` for the publisher's data format. Kaggle printed compatibility warnings for its unrelated preinstalled `pathos` and `multiprocess` packages; this experiment uses standard-library `subprocess` and does not import either. Check execution logs for actual failures rather than treating a package warning as a training result.

## Research sources

- [Fout et al. 2017 paper](https://papers.neurips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks.pdf)
- [Original Fout implementation](https://github.com/fouticus/pipgcn)
- [DB5-Plus / DIPS-Plus data](https://zenodo.org/records/5134732)
- [DB5-Plus data-processing implementation](https://github.com/BioinfoMachineLearning/DIPS-Plus/blob/main/project/utils/training_utils.py)
