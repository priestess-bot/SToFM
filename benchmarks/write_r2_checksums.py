"""Write deterministic SHA-256 manifests for R2 benchmark evidence trees."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable, List


MANIFEST_NAME = "checksums.sha256"


def _included_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        if any("invalid" in part for part in path.relative_to(root).parts):
            continue
        yield path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows: List[str] = []
    for path in _included_files(root):
        relative = path.relative_to(root).as_posix()
        rows.append(f"{_sha256(path)}  {relative}")
    target = root / MANIFEST_NAME
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.directories:
        print(write_manifest(directory))


if __name__ == "__main__":
    main()
