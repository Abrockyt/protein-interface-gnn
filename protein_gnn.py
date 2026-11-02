"""Small, reproducible protein interface prediction project.

The model sees each partner independently. Inter-chain distances create labels
only; feeding those distances to the model would trivially reveal the answer.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 224
# Fixed, seeded sample from the public LightDock BM5 mirror. No synthetic data.
IDS = [
    "3A4S", "3V6Z", "4JCV", "3HMX", "3P57", "4HX3", "3EOA", "3L89",
    "4DN4", "4M76", "3VLB", "2VXT", "1EXB", "2YVJ", "3G6D", "4H03",
    "4G6M", "1M27", "3PC8", "3F1P", "3MXW", "3EO1", "3AAD", "4GXU",
]
AA = "ACDEFGHIKLMNPQRSTVWY"
THREE = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    AA,
))


@dataclass
class Complex:
    pdb_id: str
    x_a: torch.Tensor
    adj_a: torch.Tensor
    x_b: torch.Tensor
    adj_b: torch.Tensor
    y: np.ndarray
    names_a: list[str]
    names_b: list[str]


def download_one(pdb_id: str) -> tuple[str, str]:
    path = DATA / "raw" / f"{pdb_id}.pdb"
    if path.exists() and path.stat().st_size > 1000:
        return pdb_id, "cached"
    url = ("https://raw.githubusercontent.com/lightdock/lightdock_bm5/master/"
           f"data/{pdb_id}/{pdb_id}_segid.pdb")
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "protein-interface-gnn-education"})
    with urllib.request.urlopen(request, timeout=45) as response:
        content = response.read()
    if len(content) < 1000:
        raise ValueError(f"Downloaded PDB is unexpectedly short: {pdb_id}")
    path.write_bytes(content)
    return pdb_id, f"{len(content):,} bytes"


def parse_pdb(path: Path):
    """Return CA residue coordinates plus heavy atom coordinates for chains A/B."""
    chains = {"A": {}, "B": {}}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM  ") or len(line) < 54:
            continue
        chain = line[21]
        if chain not in chains or line[16] not in (" ", "A"):
            continue
        residue = THREE.get(line[17:20].strip())
        if residue is None:
            continue
        atom = line[12:16].strip()
        element = line[76:78].strip().upper() if len(line) >= 78 else atom[0].upper()
        if element in ("H", "D"):
            continue
        try:
            coord = np.array([float(line[i:i+8]) for i in (30, 38, 46)], dtype=np.float32)
        except ValueError:
            continue
        key = (line[22:26].strip(), line[26].strip())
        record = chains[chain].setdefault(key, {"aa": residue, "ca": None, "atoms": []})
        if atom == "CA":
            record["ca"] = coord
        record["atoms"].append(coord)
    parsed = []
    for chain in ("A", "B"):
        residues = [(key, rec) for key, rec in chains[chain].items() if rec["ca"] is not None]
        if not residues:
            raise ValueError(f"{path.name}: chain {chain} has no protein residues with CA")
        parsed.append(residues)
    return parsed


def make_graph(residues):
    ca = np.stack([rec["ca"] for _, rec in residues])
    n = len(residues)
    # Within-protein geometry only. Sequence-neighbor edges retain local backbone links.
    pairs = set(cKDTree(ca).query_pairs(10.0))
    pairs.update((i, i + 1) for i in range(n - 1))
    degree = np.zeros(n, dtype=np.float32)
    for i, j in pairs:
        degree[i] += 1
        degree[j] += 1
    x = np.zeros((n, 21), dtype=np.float32)
    for i, (_, rec) in enumerate(residues):
        x[i, AA.index(rec["aa"])] = 1.0
    x[:, 20] = np.clip(degree / 30.0, 0, 1)
    # D^-1/2 (A+I) D^-1/2: the standard symmetric GCN normalization.
    rows = list(range(n))
    cols = list(range(n))
    for i, j in pairs:
        rows.extend((i, j))
        cols.extend((j, i))
    rows, cols = np.array(rows), np.array(cols)
    norm_degree = np.bincount(rows, minlength=n).astype(np.float32)
    values = 1 / np.sqrt(norm_degree[rows] * norm_degree[cols])
    adj = torch.sparse_coo_tensor(
        torch.tensor(np.stack((rows, cols)), dtype=torch.long),
        torch.tensor(values, dtype=torch.float32), (n, n),
        check_invariants=True,
    ).coalesce()
    return torch.from_numpy(x), adj


def contact_labels(a, b):
    coords_a, index_a = [], []
    coords_b, index_b = [], []
    for i, (_, residue) in enumerate(a):
        coords_a.extend(residue["atoms"])
        index_a.extend([i] * len(residue["atoms"]))
    for j, (_, residue) in enumerate(b):
        coords_b.extend(residue["atoms"])
        index_b.extend([j] * len(residue["atoms"]))
    nearby = cKDTree(np.asarray(coords_a)).query_ball_tree(cKDTree(np.asarray(coords_b)), 5.0)
    y = np.zeros((len(a), len(b)), dtype=np.uint8)
    for atom_i, partners in enumerate(nearby):
        if partners:
            y[index_a[atom_i], np.asarray(index_b)[partners]] = 1
    return y


def load_complex(pdb_id: str) -> Complex:
    a, b = parse_pdb(DATA / "raw" / f"{pdb_id}.pdb")
    x_a, adj_a = make_graph(a)
    x_b, adj_b = make_graph(b)
    y = contact_labels(a, b)
    if y.sum() == 0:
        raise ValueError(f"{pdb_id}: no inter-protein contacts found")
    names = lambda chain, rs: [f"{chain}:{key[0]}{key[1]}:{rec['aa']}" for key, rec in rs]
    return Complex(pdb_id, x_a, adj_a, x_b, adj_b, y, names("A", a), names("B", b))


class Encoder(nn.Module):
    def __init__(self, use_graph: bool, hidden: int = 32):
        super().__init__()
        self.use_graph = use_graph
        self.first = nn.Linear(21, hidden)
        self.second = nn.Linear(hidden, hidden)

    def forward(self, x, adj):
        if self.use_graph:
            x = torch.sparse.mm(adj, x)
        x = torch.relu(self.first(x))
        if self.use_graph:
            x = torch.sparse.mm(adj, x)
        return torch.relu(self.second(x))


class PairModel(nn.Module):
    def __init__(self, use_graph: bool, hidden: int = 32):
        super().__init__()
        self.encoder = Encoder(use_graph, hidden)
        self.head = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def embed(self, complex_: Complex):
        return self.encoder(complex_.x_a, complex_.adj_a), self.encoder(complex_.x_b, complex_.adj_b)

    def score(self, h_a, h_b, pair_index):
        a, b = h_a[pair_index[:, 0]], h_b[pair_index[:, 1]]
        z = torch.cat((a, b, torch.abs(a-b), a*b), dim=1)
        return self.head(z).squeeze(-1)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))


def sample_train_pairs(y: np.ndarray, rng: np.random.Generator, negative_ratio=3):
    positive = np.argwhere(y == 1)
    negative = np.argwhere(y == 0)
    selected = negative[rng.choice(len(negative), min(len(negative), negative_ratio * len(positive)), replace=False)]
    pairs = np.concatenate((positive, selected))
    targets = np.concatenate((np.ones(len(positive)), np.zeros(len(selected))))
    shuffle = rng.permutation(len(pairs))
    return torch.as_tensor(pairs[shuffle], dtype=torch.long), torch.as_tensor(targets[shuffle], dtype=torch.float32)


@torch.no_grad()
def predict(model, complex_: Complex, chunk=65536):
    model.eval()
    h_a, h_b = model.embed(complex_)
    n, m = complex_.y.shape
    output = np.empty(n * m, dtype=np.float32)
    for start in range(0, n*m, chunk):
        flat = np.arange(start, min(start + chunk, n*m))
        pairs = torch.as_tensor(np.stack((flat // m, flat % m), axis=1), dtype=torch.long)
        output[start:start+len(flat)] = torch.sigmoid(model.score(h_a, h_b, pairs)).numpy()
    return output


def metrics(y, score, threshold=0.5):
    y = np.asarray(y, dtype=np.uint8).ravel()
    score = np.asarray(score).ravel()
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    k = int(y.sum())
    top = np.argpartition(score, -k)[-k:] if k else np.array([], dtype=int)
    return {
        "pairs": len(y), "contacts": k, "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, score)) if k else None,
        "roc_auc": float(roc_auc_score(y, score)) if 0 < k < len(y) else None,
        "precision": tp / (tp+fp) if tp+fp else 0.0,
        "recall": tp / (tp+fn) if tp+fn else 0.0,
        "f1": 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
        "precision_at_k": float(y[top].mean()) if k else None,
    }


def choose_threshold(y, score):
    # Choose on validation only, never on test. A limited grid avoids fitting noise.
    candidates = np.linspace(0.05, 0.95, 19)
    return float(max(candidates, key=lambda t: metrics(y, score, float(t))["f1"]))


def evaluate(model, complexes, threshold):
    y = np.concatenate([c.y.ravel() for c in complexes])
    scores = [predict(model, c) for c in complexes]
    pooled = metrics(y, np.concatenate(scores), threshold)
    per_complex = {c.pdb_id: metrics(c.y, s, threshold) for c, s in zip(complexes, scores)}
    return pooled, per_complex, scores


def train_model(complexes, use_graph, epochs=30, lr=0.003):
    model = PairModel(use_graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(SEED + int(use_graph))
    train, valid = complexes[:16], complexes[16:20]
    best, best_state, best_epoch = -1, None, None
    for epoch in range(1, epochs + 1):
        model.train()
        for c in rng.permutation(train):
            pairs, target = sample_train_pairs(c.y, rng)
            optimizer.zero_grad()
            h_a, h_b = model.embed(c)
            logits = model.score(h_a, h_b, pairs)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            optimizer.step()
        val_y = np.concatenate([c.y.ravel() for c in valid])
        val_score = np.concatenate([predict(model, c) for c in valid])
        val_ap = average_precision_score(val_y, val_score)
        print(f"{'GCN' if use_graph else 'MLP'} epoch {epoch:02d} validation AP {val_ap:.4f}", flush=True)
        if val_ap > best:
            best, best_epoch = float(val_ap), epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    val_y = np.concatenate([c.y.ravel() for c in valid])
    val_score = np.concatenate([predict(model, c) for c in valid])
    threshold = choose_threshold(val_y, val_score)
    return model, {"best_epoch": best_epoch, "validation_ap": best, "threshold": threshold}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "all"])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    set_seed()
    if args.command in ("prepare", "all"):
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(download_one, pdb_id) for pdb_id in IDS]
            for future in as_completed(futures):
                print(*future.result(), flush=True)
    if args.command in ("train", "all"):
        missing = [pdb_id for pdb_id in IDS if not (DATA / "raw" / f"{pdb_id}.pdb").exists()]
        if missing:
            parser.error(f"Missing structures {missing}; run 'prepare' first")
        complexes = [load_complex(pdb_id) for pdb_id in IDS]
        print("Dataset:", [(c.pdb_id, c.y.shape, int(c.y.sum())) for c in complexes], flush=True)
        print("Split by complex: train", IDS[:16], "validation", IDS[16:20], "test", IDS[20:])
        RESULTS.mkdir(exist_ok=True)
        report = {"seed": SEED, "train_ids": IDS[:16], "validation_ids": IDS[16:20], "test_ids": IDS[20:],
                  "contact_cutoff_angstrom": 5.0, "graph_cutoff_angstrom": 10.0, "models": {}}
        for name, use_graph in (("MLP", False), ("GCN", True)):
            set_seed()
            model, selection = train_model(complexes, use_graph, epochs=args.epochs)
            val_pooled, val_per, _ = evaluate(model, complexes[16:20], selection["threshold"])
            test_pooled, test_per, scores = evaluate(model, complexes[20:], selection["threshold"])
            torch.save(model.state_dict(), RESULTS / f"{name.lower()}_model.pt")
            report["models"][name] = {**selection, "validation": val_pooled,
                "validation_per_complex": val_per, "test": test_pooled, "test_per_complex": test_per}
            for c, score in zip(complexes[20:], scores):
                k = min(20, len(score))
                top = np.argsort(score)[-k:][::-1]
                m = c.y.shape[1]
                rows = [{"chain_a": c.names_a[int(i//m)], "chain_b": c.names_b[int(i%m)],
                         "score": float(score[i]), "actual_contact": bool(c.y.flat[i])} for i in top]
                (RESULTS / f"{name.lower()}_{c.pdb_id}_top_pairs.json").write_text(json.dumps(rows, indent=2))
            print(name, "TEST", json.dumps(test_pooled, indent=2), flush=True)
        (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))
        print("Saved", RESULTS / "metrics.json")


if __name__ == "__main__":
    main()
