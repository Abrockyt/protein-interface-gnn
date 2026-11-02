"""Summarize the fixed test run, paired uncertainty, and prior-exposure subset."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
report = json.loads((RESULTS / "benchmark_metrics.json").read_text())
mlp = report["models"]["No-convolution MLP"]["test_per_complex"]
gcn = report["models"]["Node-average GCN"]["test_per_complex"]
codes = list(report["official_test_ids"])
old_exploratory_ids = {
    "3A4S", "3V6Z", "4JCV", "3HMX", "3P57", "4HX3", "3EOA", "3L89",
    "4DN4", "4M76", "3VLB", "2VXT", "1EXB", "2YVJ", "3G6D", "4H03",
    "4G6M", "1M27", "3PC8", "3F1P", "3MXW", "3EO1", "3AAD", "4GXU",
}
untouched = [code for code in codes if code not in old_exploratory_ids]


def values(items, key, selected=codes):
    return np.array([items[code][key] for code in selected], dtype=float)


def paired_bootstrap(key, selected=codes, repetitions=5000):
    a, b = values(mlp, key, selected), values(gcn, key, selected)
    rng = np.random.default_rng(224)
    sampled = rng.integers(0, len(a), size=(repetitions, len(a)))
    difference = np.median(b[sampled], axis=1) - np.median(a[sampled], axis=1)
    return [float(x) for x in np.quantile(difference, [0.025, 0.975])]


analysis = {
    "all_test": {
        "n": len(codes),
        "median_auc_mlp": float(np.median(values(mlp, "roc_auc"))),
        "median_auc_gcn": float(np.median(values(gcn, "roc_auc"))),
        "median_auc_difference_ci95_bootstrap": paired_bootstrap("roc_auc"),
        "gcn_better_auc_count": int(np.sum(values(gcn, "roc_auc") > values(mlp, "roc_auc"))),
        "median_ap_mlp": float(np.median(values(mlp, "average_precision"))),
        "median_ap_gcn": float(np.median(values(gcn, "average_precision"))),
    },
    "previously_unexplored_test_subset": {
        "n": len(untouched), "ids": untouched,
        "median_auc_mlp": float(np.median(values(mlp, "roc_auc", untouched))),
        "median_auc_gcn": float(np.median(values(gcn, "roc_auc", untouched))),
        "median_auc_difference_ci95_bootstrap": paired_bootstrap("roc_auc", untouched),
        "gcn_better_auc_count": int(np.sum(values(gcn, "roc_auc", untouched) > values(mlp, "roc_auc", untouched))),
    },
    "warning": "The earlier exploratory experiment used the remaining test IDs; the subset is a disclosure analysis, not a newly designed benchmark.",
}
(RESULTS / "analysis.json").write_text(json.dumps(analysis, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
palette = ["#9aa6b2", "#8c1515"]
model_names = ["No-convolution\nMLP", "Node-average\nGCN"]
auc = [analysis["all_test"]["median_auc_mlp"], analysis["all_test"]["median_auc_gcn"]]
ap = [analysis["all_test"]["median_ap_mlp"], analysis["all_test"]["median_ap_gcn"]]
for ax, heights, title, fmt in (
    (axes[0], auc, "Median ROC AUC across 55 complexes", ".3f"),
    (axes[1], ap, "Median average precision", ".4f"),
):
    bars = ax.bar(model_names, heights, color=palette, width=0.55)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(heights)*1.22)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, heights):
        ax.annotate(format(value, fmt), (bar.get_x()+bar.get_width()/2, value),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
fig.suptitle("Official DB5 test set · one fixed PyTorch run", fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS / "benchmark_comparison.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 5.1))
x = values(mlp, "roc_auc")
y = values(gcn, "roc_auc")
ax.scatter(x, y, s=25, alpha=0.82, color="#8c1515")
low, high = min(x.min(), y.min()) - 0.02, max(x.max(), y.max()) + 0.02
ax.plot([low, high], [low, high], "--", color="#5e6a75", linewidth=1)
ax.set(xlim=(low, high), ylim=(low, high),
       xlabel="No-convolution MLP ROC AUC", ylabel="Node-average GCN ROC AUC",
       title="Each point is one test complex")
ax.text(0.03, 0.97, f"GCN higher: {analysis['all_test']['gcn_better_auc_count']}/55",
        transform=ax.transAxes, va="top", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(RESULTS / "complex_auc_scatter.png", dpi=180)
plt.close(fig)
print(json.dumps(analysis, indent=2))
