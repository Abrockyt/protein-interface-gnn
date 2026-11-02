"""Paper-protocol replica of Fout et al. (NeurIPS 2017), Table 2.

Variants: No-Convolution, Node Average (Eq. 1), Node and Edge Average (Eq. 2).
Paper settings reproduced: 2 conv layers with (256, 512) filters, 512-unit dense
layer after merging, dropout 0.5, 10 random seeds, median per-complex ROC AUC on
the official 55-complex test set, 175 development complexes for final training.
Deviation (disclosed): Adam instead of SGD, one complex per step, epoch count
chosen on a 140/35 development split with seed 224 and never on test data.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from dataset import load_split
from evaluation import topk_expected

OUT = Path(__file__).resolve().parent / "results"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARIANTS = ("noconv", "node", "nodeedge")


class Case:
    def __init__(self, raw):
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=DEV)
        self.code = str(raw["complex_code"])
        self.xl, self.xr = t(raw["l_vertex"]), t(raw["r_vertex"])
        self.hl = t(raw["l_hood_indices"], torch.long).squeeze(-1)
        self.hr = t(raw["r_hood_indices"], torch.long).squeeze(-1)
        self.el, self.er = t(raw["l_edge"]), t(raw["r_edge"])
        lab = np.asarray(raw["label"], dtype=np.int64)
        assert lab.shape[1] == 3 and np.isin(lab[:, 2], [-1, 1]).all()
        self.pair = torch.as_tensor(lab[:, :2], device=DEV)
        self.y = (lab[:, 2] == 1).astype(np.uint8)
        self.yt = torch.as_tensor(self.y, dtype=torch.float32, device=DEV)


def load(split):
    return [Case(r) for r in load_split(split)[1]]


class Layer(nn.Module):
    def __init__(self, d_in, d_out, variant):
        super().__init__()
        self.variant = variant
        self.center = nn.Linear(d_in, d_out)
        if variant != "noconv":
            self.neigh = nn.Linear(d_in, d_out, bias=False)
        if variant == "nodeedge":
            self.edge = nn.Linear(2, d_out, bias=False)

    def forward(self, x, hood, edge):
        z = self.center(x)
        if self.variant != "noconv":
            z = z + self.neigh(x[hood].mean(1))
        if self.variant == "nodeedge":
            z = z + self.edge(edge.mean(1))
        return torch.relu(z)


class Net(nn.Module):
    def __init__(self, variant, filters=(256, 512), dense=512, p=0.5):
        super().__init__()
        dims = (70,) + tuple(filters)
        self.layers = nn.ModuleList(Layer(a, b, variant) for a, b in zip(dims, dims[1:]))
        self.drop = nn.Dropout(p)
        self.head = nn.Sequential(nn.Linear(2 * dims[-1], dense), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(dense, 1))

    def enc(self, x, hood, edge):
        for layer in self.layers:
            x = self.drop(layer(x, hood, edge))
        return x

    def embed(self, c):
        return self.enc(c.xl, c.hl, c.el), self.enc(c.xr, c.hr, c.er)

    def score(self, a, b, pair):
        return self.head(torch.cat((a[pair[:, 0]], b[pair[:, 1]]), 1)).squeeze(-1)


def train_epoch(model, cases, opt, rng):
    model.train()
    for i in rng.permutation(len(cases)):
        c = cases[int(i)]
        opt.zero_grad(set_to_none=True)
        a, b = model.embed(c)
        loss = nn.functional.binary_cross_entropy_with_logits(
            model.score(a, b, c.pair), c.yt, pos_weight=torch.tensor(10.0, device=DEV))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()


@torch.no_grad()
def predict(model, c, chunk=262144):
    model.eval()
    a, b = model.embed(c)
    out = [torch.sigmoid(model.score(a, b, c.pair[s:s + chunk])) for s in range(0, len(c.pair), chunk)]
    return torch.cat(out).cpu().numpy()


def evaluate(model, cases):
    per, ys, ss = {}, [], []
    for c in cases:
        s = predict(model, c)
        per[c.code] = {"auc": float(roc_auc_score(c.y, s)),
                       "ap": float(average_precision_score(c.y, s)),
                       "p10": float(topk_expected(c.y, s, 10)["precision"])}
        ys.append(c.y); ss.append(s)
    y, s = np.concatenate(ys), np.concatenate(ss)
    v = lambda k: [d[k] for d in per.values()]
    return {"median_auc": float(np.median(v("auc"))), "mean_auc": float(np.mean(v("auc"))),
            "median_ap": float(np.median(v("ap"))), "mean_p10": float(np.mean(v("p10"))),
            "pooled_ap": float(average_precision_score(y, s)),
            "pooled_auc": float(roc_auc_score(y, s)), "per_complex": per}


def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed)


def select_epoch(variant, dev, max_epochs, lr):
    perm = np.random.default_rng(224).permutation(len(dev))
    train = [dev[int(i)] for i in perm[:140]]
    valid = [dev[int(i)] for i in perm[140:]]
    seed_all(224)
    model = Net(variant).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(224)
    hist = []
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        train_epoch(model, train, opt, rng)
        auc = float(np.median([roc_auc_score(c.y, predict(model, c)) for c in valid]))
        hist.append(auc)
        print(f"[{variant}] select epoch {ep}: val median AUC {auc:.4f} ({time.time()-t0:.1f}s)", flush=True)
    best = int(np.argmax(hist)) + 1
    return best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    dev, test = load("train"), load("test")
    assert not {c.code for c in dev} & {c.code for c in test}
    print(f"dev {len(dev)} test {len(test)} device {DEV}", flush=True)
    for variant in args.variants:
        path = OUT / f"replica_{variant}.json"
        rep = json.loads(path.read_text()) if path.exists() else {}
        if "best_epoch" not in rep:
            rep["best_epoch"], rep["selection_val_median_auc"] = select_epoch(
                variant, dev, args.max_epochs, args.lr)
            rep["runs"] = []
            path.write_text(json.dumps(rep))
        print(f"[{variant}] selected epoch {rep['best_epoch']}", flush=True)
        for seed in range(len(rep["runs"]), args.seeds):
            t0 = time.time()
            seed_all(1000 + seed)
            model = Net(variant).to(DEV)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            rng = np.random.default_rng(1000 + seed)
            for _ in range(rep["best_epoch"]):
                train_epoch(model, dev, opt, rng)
            res = evaluate(model, test)
            rep["runs"].append({"seed": 1000 + seed, **res})
            path.write_text(json.dumps(rep))
            print(f"[{variant}] seed {seed}: median AUC {res['median_auc']:.4f} "
                  f"AP {res['median_ap']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        a = [r["median_auc"] for r in rep["runs"]]
        print(f"[{variant}] {len(a)} runs: median AUC {np.mean(a):.4f} +/- {np.std(a, ddof=1):.4f}", flush=True)


if __name__ == "__main__":
    main()
