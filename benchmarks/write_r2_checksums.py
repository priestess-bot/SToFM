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


def verify_manifest(root: Path) -> int:
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    root_resolved = root.resolve()
    verified = 0
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        expected, separator, relative = line.partition("  ")
        if not separator or len(expected) != 64 or not relative:
            raise ValueError(f"malformed checksum row {line_number} in {manifest}")
        path = (root / relative).resolve()
        if root_resolved not in path.parents or not path.is_file():
            raise ValueError(f"checksum row {line_number} names an invalid file: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {relative}: expected {expected}, got {actual}")
        verified += 1
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify existing manifests instead of writing them")
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.directories:
        if args.verify:
            print(f"{directory / MANIFEST_NAME}: {verify_manifest(directory)} files verified")
        else:
            print(write_manifest(directory))


if __name__ == "__main__":
    main()
