#!/usr/bin/env python
"""Generate manifest.json for a data/<split> directory.

Each .pt file is one pre-processed sample:
    storm_boundary  — (24, N, 3)   [P, Wx, Wy]
    inner_boundary  — (24, N, 2)   [h_bdy, q_bdy]
    target          — (N, 3)       [u, v, h] at t+24

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


REQUIRED_KEYS = ("storm_boundary", "inner_boundary", "target")


def find_pt_files(data_dir: Path) -> list[Path]:
    """Return sorted .pt files, excluding macOS AppleDouble metadata."""
    return sorted(path for path in data_dir.glob("*.pt") if not path.name.startswith("._"))


def read_file_metadata(path: Path) -> int:
    """Load one .pt file, validate required tensors, and return num_nodes."""
    data: Any = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected torch.load() to return dict, got {type(data).__name__}")

    for key in REQUIRED_KEYS:
        if key not in data:
            raise KeyError(f"{path}: missing key {key!r}; got {list(data.keys())}")

    storm = data["storm_boundary"]
    inner = data["inner_boundary"]
    target = data["target"]

    if storm.dim() != 3 or storm.size(0) != 24 or storm.size(-1) != 3:
        raise ValueError(f"{path}: storm_boundary must be (24,N,3), got {tuple(storm.shape)}")
    if inner.dim() != 3 or inner.size(0) != 24 or inner.size(1) != storm.size(1) or inner.size(-1) != 2:
        raise ValueError(
            f"{path}: inner_boundary {tuple(inner.shape)} incompatible with storm_boundary "
            f"{tuple(storm.shape)} (expected (24,N,2))"
        )
    if target.dim() != 2 or target.size(-1) != 3 or target.size(0) != storm.size(1):
        raise ValueError(
            f"{path}: target {tuple(target.shape)} must be (N,3) with N={storm.size(1)}"
        )

    return int(storm.size(1))


def build_manifest(data_dir: Path) -> dict[str, Any]:
    files = find_pt_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .pt files (excluding ._*) in {data_dir}")

    entries: list[str] = []
    num_nodes: int | None = None

    for path in tqdm(files, desc=f"scanning {data_dir.name}"):
        N = read_file_metadata(path)
        if num_nodes is None:
            num_nodes = N
        elif N != num_nodes:
            raise ValueError(
                f"{path}: num_nodes={N} disagrees with first-file num_nodes={num_nodes}"
            )
        entries.append(path.name)

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
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"error: {args.data_dir} is not a directory", file=sys.stderr)
        return 2

    manifest = build_manifest(args.data_dir)
    out = args.output or (args.data_dir / "manifest.json")
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote {len(manifest['files'])} entries -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
