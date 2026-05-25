#!/usr/bin/env python
"""Check .pt files under data/train, data/val, data/test for corruption.

Tries to torch.load each file and reports any that fail (e.g. truncated
uploads, zip-archive read errors). Optionally also computes MD5 sums.

Usage:
    python scripts/check_data_integrity.py
    python scripts/check_data_integrity.py --data_root data --splits train val test
    python scripts/check_data_integrity.py --md5 --md5_out md5sums.txt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


REQUIRED_KEYS = ("graph", "storm_boundary", "inner_boundary")


def find_pt_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.glob("*.pt") if not p.name.startswith("._"))


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def check_file(path: Path, deep: bool) -> tuple[bool, str]:
    """Return (ok, message). If deep, also validate required keys/shapes."""
    try:
        data: Any = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"torch.load failed: {type(e).__name__}: {e}"

    if not deep:
        return True, ""

    if not isinstance(data, dict):
        return False, f"expected dict, got {type(data).__name__}"
    for key in REQUIRED_KEYS:
        if key not in data:
            return False, f"missing key {key!r}; got {list(data.keys())}"
    graph = data["graph"]
    storm = data["storm_boundary"]
    inner = data["inner_boundary"]
    if graph.dim() != 3 or graph.size(-1) != 3:
        return False, f"graph shape must be (T,N,3), got {tuple(graph.shape)}"
    if storm.shape != graph.shape:
        return False, f"storm {tuple(storm.shape)} != graph {tuple(graph.shape)}"
    if (
        inner.dim() != 3
        or inner.size(0) != graph.size(0)
        or inner.size(1) != graph.size(1)
        or inner.size(-1) != 2
    ):
        return False, (
            f"inner_boundary {tuple(inner.shape)} incompatible with graph "
            f"{tuple(graph.shape)} (expected (T,N,2))"
        )
    return True, ""


def check_split(
    split_dir: Path,
    deep: bool,
    md5: bool,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Return (failures, md5_entries)."""
    files = find_pt_files(split_dir)
    failures: list[tuple[Path, str]] = []
    md5_entries: list[tuple[Path, str]] = []

    if not files:
        print(f"[{split_dir.name}] no .pt files found", file=sys.stderr)
        return failures, md5_entries

    for path in tqdm(files, desc=f"checking {split_dir.name}"):
        ok, msg = check_file(path, deep=deep)
        if not ok:
            failures.append((path, msg))
            continue
        if md5:
            try:
                md5_entries.append((path, md5_of(path)))
            except Exception as e:
                failures.append((path, f"md5 failed: {type(e).__name__}: {e}"))

    return failures, md5_entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check data .pt files for corruption across train/val/test splits."
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Subdirectories of --data_root to scan (default: train val test).",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Also validate required keys and tensor shapes (slower).",
    )
    parser.add_argument(
        "--md5",
        action="store_true",
        help="Compute MD5 for each readable file (slower; use --md5_out to save).",
    )
    parser.add_argument(
        "--md5_out",
        type=Path,
        default=None,
        help="If set with --md5, write '<md5>  <relpath>' lines compatible with md5sum -c.",
    )
    args = parser.parse_args()

    if not args.data_root.is_dir():
        print(f"error: {args.data_root} is not a directory", file=sys.stderr)
        return 2

    all_failures: list[tuple[Path, str]] = []
    all_md5: list[tuple[Path, str]] = []

    for split in args.splits:
        split_dir = args.data_root / split
        if not split_dir.is_dir():
            print(f"[skip] {split_dir} is not a directory", file=sys.stderr)
            continue
        failures, md5_entries = check_split(split_dir, deep=args.deep, md5=args.md5)
        all_failures.extend(failures)
        all_md5.extend(md5_entries)

    print()
    if all_failures:
        print(f"[FAIL] {len(all_failures)} bad file(s):")
        for path, msg in all_failures:
            rel = path.relative_to(args.data_root.parent) if path.is_absolute() else path
            print(f"  {rel}\n    -> {msg}")
    else:
        print("[OK] all files loaded successfully.")

    if args.md5 and args.md5_out is not None:
        lines = []
        for path, digest in all_md5:
            rel = path.relative_to(args.data_root.parent) if path.is_absolute() else path
            lines.append(f"{digest}  {rel}\n")
        args.md5_out.write_text("".join(lines))
        print(f"[md5] wrote {len(all_md5)} entries -> {args.md5_out}")

    return 1 if all_failures else 0


if __name__ == "__main__":
    sys.exit(main())
