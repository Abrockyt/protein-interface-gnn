"""Modern PyTorch baseline on Fout et al.'s official DB5 feature files.

This is an educational reimplementation, not the authors' TensorFlow model.
It uses their unbound-chain features, neighbor indices, and pair labels.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from dataset import load_split
from evaluation import topk_expected

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SEED = 224


@dataclass
class Case:
    code: str
    x_l: torch.Tensor
    x_r: torch.Tensor
    hood_l: torch.Tensor
    hood_r: torch.Tensor
    pair: np.ndarray
    y: np.ndarray


def convert_case(raw: dict) -> Case:
    xl = np.asarray(raw["l_vertex"], dtype=np.float32)
    xr = np.asarray(raw["r_vertex"], dtype=np.float32)
    hl = np.asarray(raw["l_hood_indices"], dtype=np.int64).squeeze(-1)
    hr = np.asarray(raw["r_hood_indices"], dtype=np.int64).squeeze(-1)
    labels = np.asarray(raw["label"], dtype=np.int64)
    if labels.ndim != 2 or labels.shape[1] != 3 or not np.isin(labels[:, 2], [-1, 1]).all():
        raise ValueError(f"{raw['complex_code']}: expected pair indices with -1/+1 labels")
    pair = labels[:, :2].astype(np.int32)
    y = (labels[:, 2] == 1).astype(np.uint8)
    if not (np.isfinite(xl).all() and np.isfinite(xr).all()):
        raise ValueError(f"{raw['complex_code']}: non-finite node features")
    if pair[:, 0].min() < 0 or pair[:, 0].max() >= len(xl):
        raise ValueError(f"{raw['complex_code']}: ligand pair index out of bounds")
    if pair[:, 1].min() < 0 or pair[:, 1].max() >= len(xr):
        raise ValueError(f"{raw['complex_code']}: receptor pair index out of bounds")
    if hl.min() < 0 or hl.max() >= len(xl) or hr.min() < 0 or hr.max() >= len(xr):
        raise ValueError(f"{raw['complex_code']}: neighbor index out of bounds")
    if np.unique(pair, axis=0).shape[0] != len(pair):
        raise ValueError(f"{raw['complex_code']}: duplicate labeled pair")
    if not (0 < y.sum() < len(y)):
        raise ValueError(f"{raw['complex_code']}: only one target class")
    return Case(str(raw["complex_code"]), torch.from_numpy(xl), torch.from_numpy(xr),
                torch.from_numpy(hl), torch.from_numpy(hr), pair, y)


def load_cases(split: str) -> list[Case]:
    _, raw_cases = load_split(split)
    cases = [convert_case(raw) for raw in raw_cases]
    del raw_cases
    gc.collect()
    return cases


class NodeAverageLayer(nn.Module):
    """Fout et al. 2017 Eq. (1): distinct center and neighbor transformations."""

    def __init__(self, d_in: int, d_out: int, use_neighbors: bool):
        super().__init__()
        self.use_neighbors = use_neighbors
        self.center = nn.Linear(d_in, d_out)
        if use_neighbors:
            self.neighbor = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor, hood: torch.Tensor):
        output = self.center(x)
        if self.use_neighbors:
            output = output + self.neighbor(x[hood].mean(dim=1))
        return torch.relu(output)


class PairModel(nn.Module):
    def __init__(self, use_neighbors: bool, hidden=64, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            NodeAverageLayer(70, hidden, use_neighbors),
            NodeAverageLayer(hidden, hidden, use_neighbors),
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(3*hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def encode(self, x, hood):
        for layer in self.layers:
            x = self.dropout(layer(x, hood))
        return x

    def embed(self, case: Case):
        return self.encode(case.x_l, case.hood_l), self.encode(case.x_r, case.hood_r)

    def score(self, left, right, pairs):
        a, b = left[pairs[:, 0]], right[pairs[:, 1]]
        # Symmetric in ligand/receptor order, unlike direct concatenation.
        pair_features = torch.cat((a+b, torch.abs(a-b), a*b), dim=1)
        return self.head(pair_features).squeeze(-1)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))


def split_training(cases: list[Case]):
    # The released archive groups all 175 development complexes together.
    # The paper's exact stratified 140/35 split is not published in this file.
    permutation = np.random.default_rng(SEED).permutation(len(cases))
    return [cases[int(i)] for i in permutation[:140]], [cases[int(i)] for i in permutation[140:]]


def train_epoch(model, cases, optimizer, rng):
    model.train()
    total = 0.0
    for index in rng.permutation(len(cases)):
        case = cases[int(index)]
        pairs = torch.from_numpy(case.pair.astype(np.int64))
        target = torch.from_numpy(case.y.astype(np.float32))
        optimizer.zero_grad(set_to_none=True)
        left, right = model.embed(case)
        score = model.score(left, right, pairs)
        # Released development labels already have a 10:1 negative ratio.
        # Equalizing the two classes in the loss approximates paper weighting.
        loss = nn.functional.binary_cross_entropy_with_logits(
            score, target, pos_weight=torch.tensor(10.0))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach())
    return total / len(cases)


@torch.no_grad()
def predict(model, case: Case, chunk=65536) -> np.ndarray:
    model.eval()
    left, right = model.embed(case)
    output = np.empty(len(case.pair), dtype=np.float32)
    for start in range(0, len(output), chunk):
        pairs = torch.from_numpy(case.pair[start:start+chunk].astype(np.int64))
        logits = model.score(left, right, pairs)
        output[start:start+len(pairs)] = torch.sigmoid(logits).numpy()
    return output


def case_metrics(case, score):
    y = case.y
    k = min(len(case.x_l), len(case.x_r), len(y))
    top = topk_expected(y, score, k)
    return {
        "n_pairs": int(len(y)), "n_contacts": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "precision_at_l": top["precision"],
        "l": int(k),
    }


def summarize(cases, model, include_predictions=False):
    results = {}
    all_y, all_score = [], []
    for index, case in enumerate(cases, 1):
        score = predict(model, case)
        results[case.code] = case_metrics(case, score)
        all_y.append(case.y)
        all_score.append(score)
        if index % 10 == 0:
            print(f"  evaluated {index}/{len(cases)} complexes", flush=True)
    med_auc = float(np.median([x["roc_auc"] for x in results.values()]))
    med_ap = float(np.median([x["average_precision"] for x in results.values()]))
    pooled_y = np.concatenate(all_y)
    pooled_score = np.concatenate(all_score)
    summary = {
        "n_complexes": len(cases), "n_pairs": int(len(pooled_y)),
        "n_contacts": int(pooled_y.sum()), "prevalence": float(pooled_y.mean()),
        "median_complex_roc_auc": med_auc,
        "median_complex_average_precision": med_ap,
        "mean_complex_precision_at_l": float(np.mean([x["precision_at_l"] for x in results.values()])),
        "pooled_roc_auc": float(roc_auc_score(pooled_y, pooled_score)),
        "pooled_average_precision": float(average_precision_score(pooled_y, pooled_score)),
    }
    return summary, results


def fit_selected_model(name, use_neighbors, development, train, valid, epochs, patience, lr):
    set_seed()
    model = PairModel(use_neighbors)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(SEED + int(use_neighbors))
    best_auc, best_epoch, wait = -1.0, 0, 0
    history = []
    for epoch in range(1, epochs+1):
        loss = train_epoch(model, train, optimizer, rng)
        val_auc = float(np.median([roc_auc_score(c.y, predict(model, c)) for c in valid]))
        history.append({"epoch": epoch, "train_loss": loss, "validation_median_auc": val_auc})
        print(f"{name} epoch {epoch:02d}: loss {loss:.4f}, validation median AUC {val_auc:.4f}", flush=True)
        if val_auc > best_auc + 0.0001:
            best_auc, best_epoch, wait = val_auc, epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break
    # Retrain from fresh initialization on all 175 development complexes for
    # the validation-selected number of epochs, as the paper does.
    set_seed()
    final = PairModel(use_neighbors)
    optimizer = torch.optim.Adam(final.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(SEED + int(use_neighbors))
    for epoch in range(1, best_epoch+1):
        loss = train_epoch(final, development, optimizer, rng)
        print(f"{name} final {epoch:02d}/{best_epoch}: loss {loss:.4f}", flush=True)
    return final, {"best_epoch": best_epoch, "best_validation_median_auc": best_auc,
                   "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    set_seed()
    development = load_cases("train")
    test = load_cases("test")
    train, valid = split_training(development)
    if set(c.code for c in development) & set(c.code for c in test):
        raise ValueError("Development and official test PDB IDs overlap")
    OUT.mkdir(exist_ok=True)
    report = {
        "paper": "Fout et al., NeurIPS 2017, Protein Interface Prediction using Graph Convolutional Networks",
        "data": "Zenodo 10.5281/zenodo.1127774",
        "seed": SEED, "train_ids": [c.code for c in train],
        "validation_ids": [c.code for c in valid], "official_test_ids": [c.code for c in test],
        "validation_note": "The released training labels are a 10:1 negative sample, so validation AP is not comparable with full test AP. Model selection uses median validation ROC AUC.",
        "models": {},
    }
    for name, use_neighbors in (("No-convolution MLP", False), ("Node-average GCN", True)):
        model, selection = fit_selected_model(name, use_neighbors, development, train, valid,
                                              args.epochs, args.patience, args.lr)
        summary, per_complex = summarize(test, model)
        torch.save(model.state_dict(), OUT / ("gcn.pt" if use_neighbors else "mlp.pt"))
        report["models"][name] = {"selection": selection, "test_summary": summary,
                                  "test_per_complex": per_complex}
        print(name, json.dumps(summary, indent=2), flush=True)
        (OUT / "benchmark_metrics.json").write_text(json.dumps(report, indent=2))
    print("Saved", OUT / "benchmark_metrics.json")


if __name__ == "__main__":
    main()
