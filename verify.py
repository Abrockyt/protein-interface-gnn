"""Check the key no-leakage property using a real downloaded complex."""
import copy
import numpy as np

from protein_gnn import DATA, contact_labels, make_graph, parse_pdb

a, b = parse_pdb(DATA / "raw" / "1M27.pdb")
original_labels = contact_labels(a, b)
original_features, original_graph = make_graph(b)

# Move B far away. This changes cross-protein contact labels, but not B's
# internal geometry, node features, or graph available to the GCN.
shifted_b = copy.deepcopy(b)
for _, residue in shifted_b:
    residue["ca"] += np.array([1000, 0, 0], dtype=np.float32)
    residue["atoms"] = [atom + np.array([1000, 0, 0], dtype=np.float32)
                        for atom in residue["atoms"]]
shifted_labels = contact_labels(a, shifted_b)
shifted_features, shifted_graph = make_graph(shifted_b)

assert original_labels.sum() > 0
assert shifted_labels.sum() == 0
assert np.array_equal(original_features.numpy(), shifted_features.numpy())
assert np.array_equal(original_graph.to_dense().numpy(), shifted_graph.to_dense().numpy())
print("PASS: moving the other protein changes labels but not model inputs")
