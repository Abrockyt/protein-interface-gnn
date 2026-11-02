"""Recompute both saved models and audit metric correctness without retraining.

Stores pair-level predictions and cross-checks AUC/AP against independent
rank-sum/threshold-group formulas, then reports ranking and diagnostic metrics.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from benchmark import PairModel, convert_case, set_seed, split_training
from dataset import load_split
from download import FILES, digest
from evaluation import (independent_ap, independent_auc, ranking_metrics,
                        threshold_metrics, topk_expected, uncalibrated_score_diagnostics)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CACHE = OUT / "audit_predictions"
NAMES = {"mlp": "No-convolution MLP", "gcn": "Node-average GCN"}
OLD_IDS = set("3A4S 3V6Z 4JCV 3HMX 3P57 4HX3 3EOA 3L89 4DN4 4M76 3VLB 2VXT 1EXB 2YVJ 3G6D 4H03 4G6M 1M27 3PC8 3F1P 3MXW 3EO1 3AAD 4GXU".split())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def inference(model, case, chunk=65536):
    model.eval()
    left, right = model.embed(case)
    logits, scores = np.empty(len(case.y), np.float32), np.empty(len(case.y), np.float32)
    for start in range(0, len(case.y), chunk):
        pair = torch.from_numpy(case.pair[start:start+chunk].astype(np.int64))
        z = model.score(left, right, pair)
        logits[start:start+len(pair)] = z.numpy()
        scores[start:start+len(pair)] = torch.sigmoid(z).numpy()
    return logits, scores


def chain_fingerprints(cases):
    result = {}
    for c in cases:
        for side in ("l", "r"):
            x, h = getattr(c, f"x_{side}"), getattr(c, f"hood_{side}")
            key = hashlib.sha256(x.numpy().tobytes()+h.numpy().tobytes()).hexdigest()
            result.setdefault(key, []).append(f"{c.code}:{side}")
    return result


def model_metrics(case, logits, scores):
    result = ranking_metrics(case.y, logits)
    length = min(len(case.x_l), len(case.x_r))
    for label, k in (("10", 10), ("50", 50), ("l_over_5", max(1, length//5)), ("l", length)):
        top = topk_expected(case.y, logits, k)
        result.update({f"precision_at_{label}": top["precision"],
                       f"recall_at_{label}": top["recall"],
                       f"enrichment_at_{label}": top["enrichment"],
                       f"k_{label}": top["k"], f"tie_size_at_{label}": top["cutoff_tie_size"]})
    result["ap_lift_over_prevalence"] = result["average_precision"]/result["prevalence"]
    result["fixed_0_5_threshold_diagnostic"] = threshold_metrics(case.y, scores)
    result.update(uncalibrated_score_diagnostics(case.y, logits, scores))
    return result


def bootstraps(records, codes, repetitions=5000):
    rng = np.random.default_rng(224)
    samples = rng.integers(0, len(codes), size=(repetitions, len(codes)))
    result = {}
    for metric, reducer in (("roc_auc", np.median), ("average_precision", np.median),
                            ("precision_at_10", np.mean), ("precision_at_l_over_5", np.mean)):
        a, b = [np.array([records[m][c][metric] for c in codes]) for m in ("mlp", "gcn")]
        av, bv = reducer(a[samples], axis=1), reducer(b[samples], axis=1)
        result[metric] = {"aggregation": "median" if reducer is np.median else "mean",
                          "mlp": float(reducer(a)), "gcn": float(reducer(b)),
                          "difference": float(reducer(b)-reducer(a)),
                          "paired_difference_ci95": np.quantile(bv-av, [.025, .975]).tolist(),
                          "gcn_wins": int(np.sum(b > a)), "mlp_wins": int(np.sum(a > b)),
                          "ties": int(np.sum(a == b))}
    return result


def write_report(result):
    rows = []
    for label, key in (("Median per-complex ROC AUC", "roc_auc"),
                       ("Median per-complex AP", "average_precision"),
                       ("Median trapezoidal PR area", "pr_auc_trapezoid"),
                       ("Mean precision at 10", "precision_at_10"),
                       ("Mean recall at 10", "recall_at_10"),
                       ("Mean precision at L/5", "precision_at_l_over_5"),
                       ("Mean precision at L", "precision_at_l")):
        stat = "median" if label.startswith("Median") else "mean"
        a, b = [result["models"][m]["macro"][key][stat] for m in ("mlp", "gcn")]
        rows.append(f"| {label} | {a:.6f} | {b:.6f} |")
    for label, key in (("Pooled ROC AUC", "roc_auc"), ("Pooled AP", "average_precision"),
                       ("Pooled trapezoidal PR area", "pr_auc_trapezoid"),
                       ("Standardized pooled partial ROC AUC (FPR <= 1%)", "standardized_partial_roc_auc_fpr_0_01")):
        a, b = [result["models"][m]["pooled"][key] for m in ("mlp", "gcn")]
        rows.append(f"| {label} | {a:.6f} | {b:.6f} |")
    confusion = []
    for m in ("mlp", "gcn"):
        d = result["models"][m]["pooled"]["fixed_0_5_threshold_diagnostic"]
        confusion.append(f"| {m.upper()} | {d['tp']:,} | {d['fp']:,} | {d['fn']:,} | {d['tn']:,} | {d['precision']:.6f} | {d['recall']:.6f} | {d['f1']:.6f} | {d['mcc']:.6f} |")
    reproduced = result["reproduction"]
    data = result["data_integrity"]
    calibration = []
    for m in ("mlp", "gcn"):
        p = result["models"][m]["pooled"]
        calibration.append(f"| {m.upper()} raw sigmoid | {p['mean_uncalibrated_score']:.6f} | {p['brier_uncalibrated_score']:.6f} | {p['log_loss_uncalibrated_score']:.6f} |")
    base = result["baselines"]
    calibration.append(f"| Empirical-prevalence constant (diagnostic) | {base['pooled_prevalence']:.6f} | {base['empirical_prevalence_constant_brier']:.6f} | {base['empirical_prevalence_constant_log_loss']:.6f} |")
    retrieval = {}
    for m in ("mlp", "gcn"):
        cases = result["models"][m]["per_complex"]
        retrieval[m] = {"zero": sum(v["precision_at_10"] == 0 for v in cases.values()),
                        "hits": sum(v["precision_at_10"]*v["k_10"] for v in cases.values()),
                        "budget": sum(v["k_10"] for v in cases.values()),
                        "precision": result["models"][m]["macro"]["precision_at_10"]["mean"]}
    top10_word = "higher" if retrieval["gcn"]["precision"] > retrieval["mlp"]["precision"] else "lower"
    low_fpr_word = ("higher" if result["models"]["gcn"]["pooled"]["standardized_partial_roc_auc_fpr_0_01"]
                    > result["models"]["mlp"]["pooled"]["standardized_partial_roc_auc_fpr_0_01"] else "lower")
    body = f"""# Metric audit and evaluation rationale

## Audit outcome

Both saved checkpoints were evaluated again on the exact released labels, with no
retraining or threshold optimization on the test set. Original ROC AUC and AP
values reproduce: maximum per-complex absolute errors are
{reproduced['max_absolute_roc_auc_error']:.3g} and
{reproduced['max_absolute_ap_error']:.3g}. AUC and AP were also recomputed from
independent mathematical formulas, including tied scores; maximum discrepancy
from scikit-learn is {result['independent_formula_max_error']:.3g}.

The headline ranking numbers were numerically correct. The old evaluation was
incomplete for judging whether the highest-ranked contacts are useful. It also
needed explicit tie handling, score-calibration warnings, and a clearer
distinction between AP and trapezoidal PR area.

**The practical conclusion must include retrieval:** GCN top-10 precision is
{retrieval['gcn']['precision']:.5f}, versus {retrieval['mlp']['precision']:.5f}
for the MLP: {retrieval['gcn']['hits']:.1f} versus {retrieval['mlp']['hits']:.1f}
expected true contacts among {retrieval['gcn']['budget']} recommendations
(ten per complex). The GCN finds no contact in the top ten on
{retrieval['gcn']['zero']} complexes, compared with {retrieval['mlp']['zero']}
for the MLP. GCN top-10 precision is {top10_word}; its paired interval is
{result['paired_bootstrap_55']['precision_at_10']['paired_difference_ci95']}.
The GCN's partial ROC AUC at low FPR is {low_fpr_word}. These measures must
be considered alongside the overall ROC AUC, rather than declaring a uniform
improvement from the headline score alone.

## What was checked

- Both compressed datasets match the authors' published MD5 hashes.
- Raw labels contain exactly -1 and +1; +1 is the contact class. Pair indices
  are within the corresponding ligand/receptor feature arrays. No labeled pair
  repeats inside a complex.
- The fixed 140/35 development split and the 55 official test IDs match the
  saved experiment. Development and test IDs are disjoint.
- The test includes {data['test_labeled_pairs']:,} released pairs and
  {data['test_contacts']:,} contacts. Unlisted Cartesian-product pairs are excluded,
  rather than silently converted to negatives.
- Exact identical feature-and-neighborhood representations across development
  and test: {len(data['exact_chain_representation_overlap'])}. This is an exact-array
  check, not a sequence-homology audit.
- The biological 6 Å contact definition comes from the data authors. This
  audit verifies the released labels and their use, not a fresh reconstruction
  of bound/unbound residue correspondences from experimental structures.
- Saved model/data SHA-256 hashes, package versions, and pair-level outputs
  make the recomputation traceable in `results/metric_audit.json` and
  `results/audit_predictions`.

## Why these metrics are included

| Metric | Question answered | Limitation |
| --- | --- | --- |
| Median complex ROC AUC | Does a contact usually rank above a noncontact within a typical complex? | Paper-comparable ranking; says little about precision at the top. |
| AP | How well are contacts concentrated near the top across the full ranking? | Depends on contact prevalence; compare models on identical labels. |
| Trapezoidal PR area | What is the area under the linearly joined PR points? | A different numerical convention from AP; never call them interchangeable. |
| Precision/recall at 10, 50, L/5, L | If we inspect a limited number of pairs per complex, how many contacts do we recover? | K is a stated budget; L=min(partner residue counts), L/5=floor(L/5), at least 1. |
| Top-K enrichment | How much better is top-K precision than random selection in that complex? | Relative lift can look large while absolute precision is low. |
| Partial ROC AUC at FPR <=1% | Does ranking remain useful when false-positive rates must be small? | Standardized scikit-learn convention; chance=0.5. |
| Confusion counts, precision, recall, F1, MCC | What happens at a stated score threshold? | Here 0.5 is diagnostic only; no deployment threshold has been validated. |
| Brier score and log loss | Can the weighted sigmoid outputs safely be interpreted as probabilities? | Diagnostic only; outputs were not calibrated on a naturally imbalanced holdout. |

## Recomputed ranking results

Ranking uses raw logits to avoid extra rounding/saturation ties from sigmoid.
The old sigmoid-based metrics are retained separately for reproduction.
The added retrieval budgets and diagnostic metrics were specified for this
post-hoc audit; they are exploratory, not preregistered confirmatory endpoints.

| Measure | MLP | GCN |
| --- | ---: | ---: |
{chr(10).join(rows)}

Top-K precision uses the expected number of contacts under uniform selection
inside a tie at the cutoff. This prevents positive-first input ordering from
deciding which tied predictions receive credit. Each complex contributes
equally to macro means/medians. Pooled metrics rank all test pairs together;
large complexes and score offsets between complexes can change them.

## Fixed-score classification diagnostic

The following uses sigmoid score >=0.5. It is not an independently validated
decision rule and must not be presented as optimal operating performance.

| Model | TP | FP | FN | TN | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(confusion)}

An always-negative classifier has test accuracy
{result['baselines']['always_negative_accuracy']:.6%}, recall 0, F1 0 and MCC 0.
This is why plain accuracy is not the headline metric. Sensitivity, specificity,
balanced accuracy and the confusion counts are retained in the JSON/CSV.

## Probability interpretation

Development pairs were downsampled to about 10 negatives per contact, and the
loss gives positives weight 10. That combination changes the effective training
prior. A sigmoid score of 0.8 is therefore not evidence of an 80% contact
probability under the test prevalence. The audit reports Brier/log-loss diagnostics
without fitting a calibrator on test labels. A useful threshold and calibration
require independent validation data with the natural pair distribution; the
released development files do not provide that distribution directly.

| Score source | Mean score | Brier score (lower is better) | Log loss (lower is better) |
| --- | ---: | ---: | ---: |
{chr(10).join(calibration)}

The constant reference uses the observed test prevalence after the fact. It
illustrates the scale of probability error; it is not a separately trained
deployable baseline. The raw neural scores should be used for ranking until
calibration is evaluated on suitable independent validation data.

## Uncertainty and remaining limits

Paired bootstrap intervals resample complexes, not millions of dependent residue
pairs. Full-set and the fixed 31-case historical subset are in the JSON. These
intervals describe variation across the observed complexes conditional on the
two trained models. They do not measure training-seed variation or establish
sequence-family independence. All 55 cases have now been examined; the
31-case historical subset is not a new blind test. One training seed and prior
test exposure remain study limitations even after the metric code is correct.

The MLP has {result['models']['mlp']['parameter_count']:,} trainable parameters;
the GCN has {result['models']['gcn']['parameter_count']:,}. They use matched
feature/head widths, not matched parameter counts. Thus this comparison supports
an association with graph context but does not isolate it from added capacity.

## Reproduce this audit

Run `python paper_benchmark/test_evaluation.py` and
`python paper_benchmark/audit_metrics.py`. The second command checks cached
prediction provenance before reuse; delete that cache to force fresh inference.
The original experiment JSON and checkpoints are preserved.

Sources: [paper metric definition](https://papers.neurips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks.pdf),
[AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
[partial ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html),
[probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
"""
    (ROOT.parent / "METRIC_AUDIT.md").write_text(body, encoding="utf-8")


def plot_audit_summary(result):
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.9))
    for ax, key, aggregation, title in (
        (axes[0], "roc_auc", "median", "Median complex ROC AUC"),
        (axes[1], "average_precision", "median", "Median complex AP"),
        (axes[2], "precision_at_10", "mean", "Mean precision at 10"),
    ):
        values = [result["models"][m]["macro"][key][aggregation] for m in ("mlp", "gcn")]
        bars = ax.bar(["MLP", "GCN"], values, color=["#708090", "#8c1515"], width=.55)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1 if key == "roc_auc" else max(values)*1.25)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.4f}", (bar.get_x()+bar.get_width()/2, value),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    fig.suptitle("Audited fixed checkpoints: ranking gains do not imply top-10 gains", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "audited_metric_comparison.png", dpi=180)
    plt.close(fig)


def main():
    set_seed()
    CACHE.mkdir(exist_ok=True)
    original = json.loads((OUT / "benchmark_metrics.json").read_text())
    md5_checks = {}
    for filename, expected in FILES.items():
        found = digest(ROOT / "data" / filename)
        if found != expected:
            raise ValueError(f"Dataset checksum mismatch: {filename}")
        md5_checks[filename] = found
    dev_codes, raw_dev = load_split("train")
    test_codes, raw_test = load_split("test")
    assert len(set(dev_codes)) == 175 and len(set(test_codes)) == 55
    assert not set(dev_codes) & set(test_codes)
    assert test_codes == original["official_test_ids"]
    for raw in raw_dev+raw_test:
        assert set(np.unique(raw["label"][:, 2])) == {-1, 1}
    development = [convert_case(c) for c in raw_dev]
    cases = [convert_case(c) for c in raw_test]
    train, valid = split_training(development)
    assert [c.code for c in train] == original["train_ids"]
    assert [c.code for c in valid] == original["validation_ids"]
    dev_hash, test_hash = chain_fingerprints(development), chain_fingerprints(cases)
    overlap = [{"development": dev_hash[k], "test": test_hash[k]} for k in dev_hash.keys() & test_hash.keys()]
    dev_positive = sum(int(c.y.sum()) for c in development)
    dev_negative = sum(len(c.y)-int(c.y.sum()) for c in development)
    del raw_dev, raw_test, development, train, valid
    gc.collect()
    provenance = {"torch_version": torch.__version__, "data_md5": md5_checks,
                  "weights_sha256": {m: sha256(OUT / f"{m}.pt") for m in NAMES},
                  "inference_version": "logit-and-torch-sigmoid-v1"}
    cache_manifest = CACHE / "manifest.json"
    cache_valid = cache_manifest.exists() and json.loads(cache_manifest.read_text()) == provenance
    models = {}
    for key in NAMES:
        model = PairModel(key == "gcn").eval()
        model.load_state_dict(torch.load(OUT / f"{key}.pt", map_location="cpu", weights_only=True))
        models[key] = model
    records, all_scores, all_logits = {m: {} for m in NAMES}, {m: [] for m in NAMES}, {m: [] for m in NAMES}
    independent_error, reproduction_auc, reproduction_ap, reproduction_l = [], [], [], []
    for number, case in enumerate(cases, 1):
        path = CACHE / f"{case.code}.npz"
        if cache_valid and path.exists():
            with np.load(path, allow_pickle=False) as z:
                assert np.array_equal(z["pair"], case.pair) and np.array_equal(z["y"], case.y)
                output = {name: z[name] for m in NAMES for name in (f"{m}_logit", f"{m}_score")}
        else:
            output = {}
            for m, model in models.items():
                output[f"{m}_logit"], output[f"{m}_score"] = inference(model, case)
            np.savez_compressed(path, pair=case.pair, y=case.y, **output)
        for m in NAMES:
            logit, score = output[f"{m}_logit"], output[f"{m}_score"]
            assert np.isfinite(logit).all() and np.isfinite(score).all()
            metrics = model_metrics(case, logit, score)
            records[m][case.code] = metrics
            for s in (logit, score):
                independent_error.extend([abs(independent_auc(case.y, s)-roc_auc_score(case.y, s)),
                                          abs(independent_ap(case.y, s)-average_precision_score(case.y, s))])
            old = original["models"][NAMES[m]]["test_per_complex"][case.code]
            reproduction_auc.append(abs(roc_auc_score(case.y, score)-old["roc_auc"]))
            reproduction_ap.append(abs(average_precision_score(case.y, score)-old["average_precision"]))
            k = old["l"]
            legacy_precision = float(case.y[np.argpartition(score, -k)[-k:]].mean())
            reproduction_l.append(abs(legacy_precision-old["precision_at_l"]))
            all_scores[m].append(score)
            all_logits[m].append(logit)
        if number % 5 == 0:
            print(f"Audited {number}/55 complexes", flush=True)
    cache_manifest.write_text(json.dumps(provenance, indent=2))
    result = {"provenance": provenance,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
              "data_integrity": {"development_complexes": 175, "test_complexes": 55,
                  "development_contacts": dev_positive, "development_noncontacts": dev_negative,
                  "development_negative_positive_ratio": dev_negative/dev_positive,
                  "test_labeled_pairs": sum(len(c.y) for c in cases),
                  "test_contacts": sum(int(c.y.sum()) for c in cases),
                  "test_unlisted_cartesian_pairs": sum(len(c.x_l)*len(c.x_r)-len(c.y) for c in cases),
                  "exact_chain_representation_overlap": overlap},
              "independent_formula_max_error": max(independent_error),
              "reproduction": {"max_absolute_roc_auc_error": max(reproduction_auc),
                  "max_absolute_ap_error": max(reproduction_ap),
                  "max_absolute_legacy_precision_at_l_error": max(reproduction_l)}, "models": {}}
    assert max(reproduction_auc+reproduction_ap+reproduction_l) < 1e-7
    assert max(independent_error) < 1e-12
    pooled_y = np.concatenate([c.y for c in cases])
    prevalence = float(pooled_y.mean())
    result["baselines"] = {"pooled_prevalence": prevalence, "random_roc_auc": 0.5,
        "random_precision_reference": prevalence, "always_negative_accuracy": 1-prevalence,
        "empirical_prevalence_constant_brier": prevalence*(1-prevalence),
        "empirical_prevalence_constant_log_loss": float(-prevalence*np.log(prevalence)-(1-prevalence)*np.log1p(-prevalence)),
        "constant_baseline_note": "Empirical test-prevalence diagnostic reference, not a fitted deployment model."}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for m in NAMES:
        logits, scores = np.concatenate(all_logits[m]), np.concatenate(all_scores[m])
        pooled = ranking_metrics(pooled_y, logits)
        pooled["fixed_0_5_threshold_diagnostic"] = threshold_metrics(pooled_y, scores)
        pooled.update(uncalibrated_score_diagnostics(pooled_y, logits, scores))
        independent_error.extend([abs(independent_auc(pooled_y, logits)-pooled["roc_auc"]),
                                  abs(independent_ap(pooled_y, logits)-pooled["average_precision"])])
        scalar_keys = [k for k, v in records[m][cases[0].code].items() if isinstance(v, (int, float))]
        macro = {k: {"mean": float(np.mean([records[m][c.code][k] for c in cases])),
                     "median": float(np.median([records[m][c.code][k] for c in cases]))} for k in scalar_keys}
        result["models"][m] = {"parameter_count": sum(p.numel() for p in models[m].parameters()),
                               "macro": macro, "pooled": pooled, "per_complex": records[m]}
        precision, recall, _ = precision_recall_curve(pooled_y, logits)
        fpr, tpr, _ = roc_curve(pooled_y, logits)
        color = "#8c1515" if m == "gcn" else "#708090"
        axes[0].plot(recall, precision, color=color, label=f"{m.upper()} AP={pooled['average_precision']:.5f}")
        axes[1].plot(fpr, tpr, color=color, label=m.upper())
    axes[0].axhline(prevalence, color="black", linestyle="--", label="Prevalence reference")
    axes[0].set(xlabel="Recall", ylabel="Precision (log scale)", yscale="log", ylim=(prevalence/2, 1),
                title="Pooled precision-recall; released pairs")
    axes[1].plot([0, .01], [0, .01], "--", color="black", label="Random ranking")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, .01),
                title="Pooled ROC at low false-positive rates")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "audited_pr_low_fpr_roc.png", dpi=180)
    plt.close(fig)
    codes = [c.code for c in cases]
    result["paired_bootstrap_55"] = bootstraps(records, codes)
    result["paired_bootstrap_historical_31"] = bootstraps(records, [c for c in codes if c not in OLD_IDS])
    result["independent_formula_max_error"] = max(independent_error)
    assert result["independent_formula_max_error"] < 1e-12
    result["provenance"]["audit_code_sha256"] = sha256(__file__)
    result["provenance"]["metric_code_sha256"] = sha256(ROOT / "evaluation.py")
    result["provenance"]["model_code_sha256"] = sha256(ROOT / "benchmark.py")
    result["prediction_file_sha256"] = {c.code: sha256(CACHE / f"{c.code}.npz") for c in cases}
    (OUT / "metric_audit.json").write_text(json.dumps(result, indent=2))
    csv_rows = []
    for m in NAMES:
        for code, metrics in records[m].items():
            row = {"model": m, "complex": code, "initial_exploration_overlap": code in OLD_IDS}
            row.update({k: v for k, v in metrics.items() if not isinstance(v, dict)})
            row.update({f"threshold_0_5_{k}": v for k, v in metrics["fixed_0_5_threshold_diagnostic"].items()})
            csv_rows.append(row)
    with (OUT / "audited_per_complex.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_report(result)
    plot_audit_summary(result)
    print("Original metric reproduction:", result["reproduction"], flush=True)
    print("Independent formula max error:", result["independent_formula_max_error"], flush=True)
    print("Saved metric_audit.json, audited_per_complex.csv, METRIC_AUDIT.md", flush=True)


if __name__ == "__main__":
    main()
