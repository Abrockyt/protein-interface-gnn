"""Fetch the authors' DB5 preprocessed data and check published MD5 sums."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data"
FILES = {
    "train.cpkl.gz": "c9d461dec1c4a71a2a788051fde5b85b",
    "test.cpkl.gz": "801bb8582145af116a0092d8d18a3759",
}


def fetch(name, expected):
    RAW.mkdir(exist_ok=True)
    target = RAW / name
    if target.exists() and digest(target) == expected:
        return f"{name}: verified existing file"
    url = f"https://zenodo.org/records/1127774/files/{name}?download=1"
    partial = RAW / (name + ".partial")
    request = Request(url, headers={"User-Agent": "protein-interface-gnn-course-project"})
    with urlopen(request, timeout=120) as response, partial.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    found = digest(partial)
    if found != expected:
        raise ValueError(f"{name}: MD5 mismatch; expected {expected}, found {found}")
    partial.replace(target)
    return f"{name}: verified {target.stat().st_size:,} bytes"


def digest(path):
    checksum = md5()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            checksum.update(block)
    return checksum.hexdigest()


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch, name, expected) for name, expected in FILES.items()]
        for future in futures:
            print(future.result(), flush=True)
