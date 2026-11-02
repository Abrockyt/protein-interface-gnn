"""Check dataset separation and key GNN invariances."""
import numpy as np
import torch

from benchmark import PairModel, load_cases, set_seed, split_training

set_seed()
development = load_cases("train")
test = load_cases("test")
train, valid = split_training(development)
assert len(train) == 140 and len(valid) == 35 and len(test) == 55
assert not ({c.code for c in development} & {c.code for c in test})

case = development[0]
model = PairModel(True).eval()
with torch.no_grad():
    original_l = model.encode(case.x_l, case.hood_l)
    original_r = model.encode(case.x_r, case.hood_r)
    pair = torch.tensor(case.pair[:10].astype(np.int64))
    direct = model.score(original_l, original_r, pair)
    swapped = model.score(original_r, original_l, pair[:, [1, 0]])
    assert torch.allclose(direct, swapped, atol=1e-6)

    # Relabel ligand nodes. Neighbor indices must transform with the relabeling.
    n = len(case.x_l)
    permutation = torch.randperm(n)
    old_to_new = torch.empty(n, dtype=torch.long)
    old_to_new[permutation] = torch.arange(n)
    relabeled_x = case.x_l[permutation]
    relabeled_hood = old_to_new[case.hood_l[permutation]]
    relabeled_embedding = model.encode(relabeled_x, relabeled_hood)
    assert torch.allclose(relabeled_embedding, original_l[permutation], atol=1e-5)

print("PASS: official train/test IDs are separate")
print("PASS: pair scoring is partner-order invariant")
print("PASS: graph encoding is residue-relabeling equivariant")
