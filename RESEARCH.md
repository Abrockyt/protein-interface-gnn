# Research basis and related GitHub projects

## Primary paper for this coursework topic

**Fout, Byrd, Shariat and Ben-Hur (NeurIPS 2017), [Protein Interface
Prediction using Graph Convolutional Networks](https://papers.neurips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks.pdf).**
This is the paper that closely matches the professor's topic description.
It treats a residue as a graph node, encodes the two protein partners with
weight-sharing graph convolutions, and classifies residue pairs across them.
Its inputs are **unbound** protein structures; labels come from the **bound**
complex using a 6 Å non-hydrogen atom contact definition. The paper uses
Docking Benchmark 5, trains on 175 development complexes, and evaluates on
55 separate complexes. Its main comparison reports the median ROC AUC across
test complexes, averaged across ten random runs. The best node-and-edge
average variant reports 0.898 ± 0.005; a no-convolution comparison reports
about 0.81. The authors also state that extreme imbalance keeps all methods'
precision-recall area below 0.017.

That precision-recall area statement should not be compared directly with
our median AP: both the integration convention and the aggregation must
match. The numerical audit and expanded diagnostic measures are documented
in `METRIC_AUDIT.md`.

- [Authors' original GitHub code](https://github.com/fouticus/pipgcn)
- [Authors' released DB5 data and file descriptions](https://zenodo.org/records/1127774)

The original code requires Python 2.7 and TensorFlow 1.0.1. This project
implements a **new, smaller PyTorch educational version** of the paper's
node-average idea. It uses the authors' published preprocessed data and exact
test labels; it is **not an exact reproduction** of their architecture,
hyperparameter search, stratified validation split, or ten-run protocol.

## Related work and code

| Study | What it contributes | GitHub |
| --- | --- | --- |
| Morehead et al., [Geometric Transformers for Protein Interface Contact Prediction](https://arxiv.org/abs/2110.02423) (ICLR 2022) | DeepInteract uses richer geometric graphs and a transformer for partner-specific contact maps. Its released training setup needs large feature archives and sequence databases, so it is a research comparison, not a claim that our lightweight code reproduces it. | [DeepInteract](https://github.com/BioinfoMachineLearning/DeepInteract) |
| Morehead et al., [DIPS-Plus](https://www.nature.com/articles/s41597-023-02409-3) (Scientific Data 2023) | A much larger, feature-rich dataset for interface prediction; highlights the value of unbound testing and sequence or structure-aware splits. | [DIPS-Plus](https://github.com/BioinfoMachineLearning/DIPS-Plus) |

## How this project's benchmark aligns with Fout et al.

| Item | Fout et al. paper | This project |
| --- | --- | --- |
| Input proteins | Unbound structures | Their released unbound-chain features |
| Labels | 6 Å heavy-atom contact in bound complex | Their released labels |
| Development / test | 175 / 55 complexes | Same released groups |
| Validation split | 140 / 35, stratified by type and difficulty | 140 / 35, fixed seeded split; original assignment unavailable |
| Graph layer | Several variants; node average is Eq. (1) | Two-layer node-average variant with 64 hidden features |
| Pair predictor | Shared encoders plus dense pair classifier | Shared encoders plus symmetric pair classifier |
| Test metric | Median per-complex ROC AUC over ten runs | Median per-complex ROC AUC for one fixed run, plus AP |

**Why the earlier exploratory report is superseded:** it used only 24 bound
structures and a 5 Å label cutoff. Those scores do not directly compare to
the paper. The newer `paper_benchmark` experiment is the primary result.

**Prior exposure disclosure:** the earlier exploratory work evaluated 24
complexes that overlap the paper's 55-complex test set. The new model design
follows the paper and selects epoch count only on the development group, but
the official test is not completely blind to the project history. A truly
fresh final study would require a new external holdout.
