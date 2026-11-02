"""Load the verified Python 2 DB5 pickles published with Fout et al. 2017."""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Accept only the NumPy constructors required by the paper's data files."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("_codecs", "encode"),
        ("copy_reg", "_reconstructor"),
        ("__builtin__", "object"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"Unexpected class in research dataset: {module}.{name}")
        return super().find_class(module, name)


def load_split(name: str):
    if name not in ("train", "test"):
        raise ValueError(name)
    with gzip.open(ROOT / "data" / f"{name}.cpkl.gz", "rb") as stream:
        codes, cases = RestrictedNumpyUnpickler(stream, encoding="latin1").load()
    if len(codes) != len(cases):
        raise ValueError("PDB code / complex count mismatch")
    for code, case in zip(codes, cases):
        if code != case["complex_code"]:
            raise ValueError(f"Complex code mismatch: {code}")
        for key in ("l_vertex", "r_vertex", "l_hood_indices", "r_hood_indices", "label"):
            if not isinstance(case[key], np.ndarray):
                raise TypeError(f"{code}: {key} is not a NumPy array")
    return codes, cases


if __name__ == "__main__":
    for split in ("train", "test"):
        codes, cases = load_split(split)
        print(split, len(codes), codes[:5], flush=True)
        first = cases[0]
        print({key: (value.shape, str(value.dtype)) for key, value in first.items()
               if isinstance(value, np.ndarray)}, flush=True)
