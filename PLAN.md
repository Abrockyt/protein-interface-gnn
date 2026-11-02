# Completed plan: paper-guided protein interface GNN

1. **Identify the primary paper.** Match the assigned protein-interface
   topic to Fout et al., NeurIPS 2017, and inspect its method, official code,
   data release, contact definition, and reported evaluation metric. Done.
2. **Use the research dataset.** Download the authors' 175 development and
   55 test complexes; verify both published MD5 checksums. Done.
3. **Implement two matched models.** Build a two-layer node-average GCN
   based on the paper's Equation (1), plus a center-only MLP that uses the
   same input features and pair classifier. Done.
4. **Select on development data.** Fix a seeded 140/35 split, select training
   epoch by median validation ROC AUC, then retrain on all 175 cases. Done.
5. **Evaluate on the official test.** Score all released test pair labels,
   report median per-complex ROC AUC and AP, pooled AP, per-complex wins,
   and a paired bootstrap interval. Done.
6. **Audit prior exposure and validity.** Separate the 31 complexes not used
   in the earlier exploratory project; verify split disjointness and model
   invariances. Done.
7. **Prepare the hand-in.** Update the written report, presentation, beginner
   guide, research links, PDF, slides, and project package. Done.
8. **Independently audit evaluation.** Reload saved checkpoints, rescore every
   released pair, reproduce old values, validate AUC/AP with independent
   formulas, add tie-aware retrieval and diagnostic metrics, and correct the
   conclusions where overall AUC does not reflect top-10 usefulness. Done.
