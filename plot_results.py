"""Create the result figure from measured metrics.json, without retraining."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
report = json.loads((root / "results" / "metrics.json").read_text())
names = ["MLP", "GCN"]
test = [report["models"][name]["test"] for name in names]
baseline = test[0]["prevalence"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(names, [item["average_precision"] for item in test],
              color=["#9aa6b2", "#8c1515"], width=0.55)
ax.axhline(baseline, color="#355c7d", linestyle="--", linewidth=1.5,
           label=f"Random ranking = prevalence ({baseline:.5f})")
for bar, item in zip(bars, test):
    ax.annotate(f"{item['average_precision']:.5f}",
                (bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points", ha="center")
ax.set(ylabel="Average precision on held-out complexes",
       title="Protein residue-pair contact prediction", ylim=(0, max(item["average_precision"] for item in test)*1.3))
ax.legend(loc="upper left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(root / "results" / "test_average_precision.png", dpi=180)
plt.close(fig)
