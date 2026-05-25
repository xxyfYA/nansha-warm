#!/usr/bin/env python
"""Generate manifest.json for a data/<split> directory.

Usage:
    python scripts/build_manifest.py data/train
    python scripts/build_manifest.py data/test --output data/test/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


REQUIRED_KEYS = ("graph", "storm_boundary", "inner_boundary")


def find_pt_files(data_dir: Path) -> list[Path]:
    """Return sorted .pt files, excluding macOS AppleDouble metadata."""
    return sorted(path for path in data_dir.glob("*.pt") if not path.name.startswith("._"))


def read_file_metadata(path: Path) -> tuple[int, int]:
    """Load one .pt file, validate required tensors, and return (T, N)."""
    data: Any = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected torch.load() to return dict, got {type(data).__name__}")

    for key in REQUIRED_KEYS:
        if key not in data:
            raise KeyError(f"{path}: missing key {key!r}; got {list(data.keys())}")

    graph = data["graph"]
    storm = data["storm_boundary"]
    inner = data["inner_boundary"]

    if graph.dim() != 3 or graph.size(-1) != 3:
        raise ValueError(f"{path}: graph shape must be (T,N,3), got {tuple(graph.shape)}")
    if storm.shape != graph.shape:
        raise ValueError(
            f"{path}: storm_boundary {tuple(storm.shape)} != graph {tuple(graph.shape)}"
        )
    if (
        inner.dim() != 3
        or inner.size(0) != graph.size(0)
        or inner.size(1) != graph.size(1)
        or inner.size(-1) != 2
    ):
        raise ValueError(
            f"{path}: inner_boundary {tuple(inner.shape)} incompatible with graph "
            f"{tuple(graph.shape)} (expected (T,N,2))"
        )

    return int(graph.size(0)), int(graph.size(1))


def build_manifest(data_dir: Path, bundle_size_warn: int | None = None) -> dict[str, Any]:
    files = find_pt_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .pt files (excluding ._*) in {data_dir}")

    entries: list[dict[str, int | str]] = []
    num_nodes: int | None = None
    warned_files: list[tuple[str, int]] = []

    for path in tqdm(files, desc=f"scanning {data_dir.name}"):
        T, N = read_file_metadata(path)
        if num_nodes is None:
            num_nodes = N
        elif N != num_nodes:
            raise ValueError(
                f"{path}: num_nodes={N} disagrees with first-file num_nodes={num_nodes}"
            )

        if bundle_size_warn is not None and T <= bundle_size_warn:
            warned_files.append((path.name, T))
        entries.append({"path": path.name, "T": T})

    if warned_files:
        print(
            f"[manifest] warning: {len(warned_files)} files have "
            f"T <= bundle_size_warn={bundle_size_warn}:",
            file=sys.stderr,
        )
        for name, T in warned_files[:10]:
            print(f"  {name}: T={T}", file=sys.stderr)
        if len(warned_files) > 10:
            print(f"  ... and {len(warned_files) - 10} more", file=sys.stderr)

    return {
        "num_nodes": num_nodes,
        "files": entries,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build manifest.json for a data split.")
    parser.add_argument("data_dir", type=Path, help="Path to data/<split> directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path (default: <data_dir>/manifest.json).",
    )
    parser.add_argument(
        "--bundle_size_warn",
        type=int,
        default=None,
        help="Warn for files with T <= this value.",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"error: {args.data_dir} is not a directory", file=sys.stderr)
        return 2

    manifest = build_manifest(args.data_dir, bundle_size_warn=args.bundle_size_warn)
    out = args.output or (args.data_dir / "manifest.json")
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote {len(manifest['files'])} entries -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
