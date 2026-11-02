# What to submit and how to explain it

## Finished files

- **Latest Kaggle experiment:** `submission/Kaggle_Experiment_Report.pdf`
  explains the completed two-T4 run, all three GNN seeds, calibrated
  thresholds, independently verified metrics and the low-F1 limitation.
- **Kaggle notebook and outputs:** `kaggle_training/` contains the
  self-contained notebook and downloaded checkpoints, histories and
  predictions in `cloud_results/`.
- **Research report:** `submission/Protein_Interface_GNN_Report.pdf`
- **Editable slides:** `submission/Protein_Interface_GNN_Presentation.pptx`
- **Code and reproducibility:** `Protein_Interface_GNN_Submission.zip` in the
  parent Documents folder
- **Primary paper:** `references/Fout_et_al_2017_Protein_Interface_GCN.pdf`
- **Evaluation audit:** `submission/Metric_Audit.pdf`
- **Metric learning guide:** `submission/Metrics_Explanation_Guide.pdf`

If only a written submission is requested, the report PDF is the main file.
Include the latest Kaggle experiment report as the follow-up experiment;
its DB5-Plus candidate pairs differ from the earlier Fout-release benchmark.
If code is required, include the ZIP. The ZIP contains the research links,
all source code, saved results and models, source assignment, and materials
needed to rerun the official benchmark. Add your own name, student ID,
course, and professor details if your institution requires them; those
personal details have not been invented here.

## Sixty-second explanation of the latest Kaggle experiment

"I trained a baseline and three graph neural networks directly on Kaggle's
two T4 GPUs. I used DB5-Plus unbound protein features and full residue-pair
contact labels. I kept training, checkpoint selection, threshold calibration
and test complexes separate. Validation selected epochs 7 to 10; early
stopping ended training after 27 to 30 epochs. The fixed GNN ensemble
achieved 2.16% precision, 11.94% recall and 3.66% F1, versus 2.43% baseline
F1. Its pooled ROC-AUC is 0.8746. The large AUC does not imply strong contact
precision: 584 of its 26,993 proposed contacts were correct. I independently
checked the saved predictions with 4,210 metric checks. These results
demonstrate the pipeline, but remain too noisy for reliable contact
prediction. The underlying test structures were examined earlier, so I do
not claim a new blind external evaluation."

## Sixty-second explanation of the earlier Fout-release benchmark

“I followed Fout et al.'s 2017 protein-interface graph convolution paper.
I used the authors' official Docking Benchmark 5 data: unbound protein
features as inputs and bound-complex contact labels as targets. My PyTorch
model shares a two-layer node-average graph encoder between the proteins,
then predicts whether a residue pair contacts. Against a matched model
without neighbor aggregation, median ROC AUC on 55 official test complexes
improved from 0.843 to 0.891, and median average precision improved from
0.0127 to 0.0182. Top-10 precision did not improve: it is 4.91% versus 5.09%
for the MLP. Contacts are extremely rare, so I report overall ranking,
early retrieval, diagnostic threshold metrics and the limitations.
This is a paper-guided reimplementation, not an exact
replication of the authors' ten-run experiment.”

Include the audit qualification in your explanation: the GCN's overall AUC
is higher, but top-10 precision is **4.91%**, compared with the MLP's **5.09%**.
The report does not claim practical improvement across all metrics. The
uncalibrated raw 0.5 threshold is diagnostic, not a validated operating rule.

Read `PRESENTATION.md` for detailed speaking notes and viva answers.
