#!/usr/bin/env python3
"""Build a relative-path package inventory and SHA-256 checksum list."""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs" / "PUBLIC_PACKAGE_INVENTORY.csv"
CHECKSUMS = ROOT / "docs" / "SHA256SUMS.txt"
EXCLUDED = {INVENTORY.resolve(), CHECKSUMS.resolve()}
IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".git"}


def repository_bytes(path: Path) -> bytes:
    """Return the Git-cleaned bytes that will be stored for this path.

    This makes hashes stable across Windows checkouts, where Git may present
    CRLF working-tree files even though `.gitattributes` stores LF blobs.
    `hash-object -w` also handles any other configured clean filters.
    """
    relative = path.relative_to(ROOT).as_posix()
    oid = subprocess.check_output(
        ["git", "hash-object", "-w", "--path", relative, str(path)],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()
    return subprocess.check_output(
        ["git", "cat-file", "blob", oid], cwd=ROOT
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def io_path(path: Path) -> Path:
    """Return a Windows extended-length path when needed for filesystem I/O."""
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def package_files() -> list[Path]:
    tracked = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).split(b"\0")
    files = []
    for raw in tracked:
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        path = ROOT / relative
        if not io_path(path).is_file() or path.resolve() in EXCLUDED:
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
            data = repository_bytes(path)
            writer.writerow((path.relative_to(ROOT).as_posix(), len(data), sha256(data)))

    with CHECKSUMS.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("SHA-256 checksums for package files. The inventory and this checksum file are excluded.\n")
        for path in files:
            stream.write(f"{sha256(repository_bytes(path))}  {path.relative_to(ROOT).as_posix()}\n")

    print(f"Inventoried {len(files)} package files.")
    print(f"Inventory: {INVENTORY.relative_to(ROOT).as_posix()}")
    print(f"Checksums: {CHECKSUMS.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
