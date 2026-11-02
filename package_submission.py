"""Create a portable ZIP of the completed project."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parent
archive = root.parent / "Protein_Interface_GNN_Submission.zip"
skip_parts = {"__pycache__", ".git", "tmp"}
skip_names = {"report_page1_preview.png", "report_last_preview.png"}

with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in skip_parts for part in path.parts):
            continue
        if path.name in skip_names or path.suffix == ".pyc":
            continue
        zf.write(path, f"protein-interface-gnn/{path.relative_to(root).as_posix()}")
print(archive)
