#!/usr/bin/env python3
"""Build a relative-path package inventory and SHA-256 checksum list."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs" / "PUBLIC_PACKAGE_INVENTORY.csv"
CHECKSUMS = ROOT / "docs" / "SHA256SUMS.txt"
EXCLUDED = {INVENTORY.resolve(), CHECKSUMS.resolve()}
IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.resolve() in EXCLUDED:
            continue
        if any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if "outputs" in path.relative_to(ROOT).parts:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix().casefold())


def main() -> None:
    files = package_files()
    INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    with INVENTORY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "bytes", "sha256"))
        for path in files:
            writer.writerow((path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)))

    with CHECKSUMS.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("SHA-256 checksums for package files. The inventory and this checksum file are excluded.\n")
        for path in files:
            stream.write(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n")

    print(f"Inventoried {len(files)} package files.")
    print(f"Inventory: {INVENTORY.relative_to(ROOT).as_posix()}")
    print(f"Checksums: {CHECKSUMS.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
