#!/usr/bin/env python3
"""Build a relative-path package inventory and SHA-256 checksum list."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs" / "PUBLIC_PACKAGE_INVENTORY.csv"
CHECKSUMS = ROOT / "docs" / "SHA256SUMS.txt"
EXCLUDED = {INVENTORY.resolve(), CHECKSUMS.resolve()}
IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".git"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with io_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def io_path(path: Path) -> Path:
    """Return a Windows extended-length path when needed for filesystem I/O."""
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def package_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        if "outputs" in relative.parts or not io_path(path).is_file():
            continue
        if path.resolve() in EXCLUDED:
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
            writer.writerow((path.relative_to(ROOT).as_posix(), io_path(path).stat().st_size, sha256(path)))

    with CHECKSUMS.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("SHA-256 checksums for package files. The inventory and this checksum file are excluded.\n")
        for path in files:
            stream.write(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n")

    print(f"Inventoried {len(files)} package files.")
    print(f"Inventory: {INVENTORY.relative_to(ROOT).as_posix()}")
    print(f"Checksums: {CHECKSUMS.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
