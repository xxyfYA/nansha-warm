"""Single-step direct prediction test for Geo-FNO warm-up model (h-only).

Each .pt file is one pre-processed sample:
    storm_boundary  — (24, N, 3)
    inner_boundary  — (24, N, 2)
    target          — (N, 3)   [u, v, h] at t+24

The script loads each file, builds the 120-channel feature from the 24h forcing
window, predicts h at t+24, and compares against the ground-truth target.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io
import torch
from tqdm import tqdm

from dataset import build_features, load_pt, load_static_coords
from main import set_seed
from model import GeoFNO2d
from temporal_utils import (
    CHANNEL_NAME,
    C_IN,
    C_OUT,
    build_checkpoint_name,
)


METRIC_SPACES = ("physical", "normalized")
WATER_LEVEL_CHANNEL = 2
DRY_WATER_LEVEL_THRESHOLD = 0.005
GRAPH_STATS_KEYS = ("graph_mean", "graph_std")
LEGACY_STATS_KEYS = ("u_mean", "u_std", "v_mean", "v_std", "h_mean", "h_std")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Geo-FNO warm-up single-step test across a split (h-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_dir", type=str, default="data/test", help="Path to test split directory.")
    parser.add_argument("--coords", type=str, default="data/coordinates.mat", help="Path to coordinates.mat.")
    parser.add_argument("--norm", type=str, default="data/normalization.mat", help="Path to normalization.mat.")
    parser.add_argument("--model", type=str, default=None, help="Checkpoint path.")
    parser.add_argument("--output", type=str, default="geofno_warmup_results.txt", help="Base output path.")
    parser.add_argument("--num_files", type=int, default=None, help="Limit number of files for smoke tests.")
    parser.add_argument("--allow_random_weights", action="store_true", help="Run without a checkpoint.")
    parser.add_argument("--modes", type=int, default=24, help="Fourier modes per axis.")
    parser.add_argument("--width", type=int, default=48, help="Model width.")
    parser.add_argument("--s1", type=int, default=64, help="Internal grid size along axis 1.")
    parser.add_argument("--s2", type=int, default=64, help="Internal grid size along axis 2.")
    parser.add_argument("--num_fno_layers", type=int, default=4, help="Number of FNO layers.")
    parser.add_argument("--fc1_hidden", type=int, default=256, help="Hidden dim of the post-FNO FC1 layer.")
    parser.add_argument("--device", type=str, default="auto", help="Device string or auto.")
    return parser.parse_args()


def metric_output_path(base_path, metric_space):
    path = Path(base_path)
    suffix = path.suffix or ".txt"
    return path.with_name(f"{path.stem}_{metric_space}{suffix}")


def load_normalization_stats(stats_path, device):
    print(f"[test] loading normalization stats from {stats_path}")
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {path}")

    stats = scipy.io.loadmat(path)
    if all(key in stats for key in GRAPH_STATS_KEYS):
        mean_values = np.asarray(stats["graph_mean"], dtype=np.float32).reshape(-1)
        std_values = np.asarray(stats["graph_std"], dtype=np.float32).reshape(-1)
        if mean_values.size < 3 or std_values.size < 3:
            raise ValueError(
                f"{path} graph_mean/graph_std must contain at least 3 channels; "
                f"got graph_mean={mean_values.size}, graph_std={std_values.size}"
            )
        mean_values = mean_values[:3]
        std_values = std_values[:3]
    elif all(key in stats for key in LEGACY_STATS_KEYS):
        mean_values = np.asarray(
            [stats["u_mean"].item(), stats["v_mean"].item(), stats["h_mean"].item()],
            dtype=np.float32,
        )
        std_values = np.asarray(
            [stats["u_std"].item(), stats["v_std"].item(), stats["h_std"].item()],
            dtype=np.float32,
        )
    else:
        raise KeyError(
            f"{path} missing normalization stats. Acceptable key sets: "
            f"{GRAPH_STATS_KEYS} or {LEGACY_STATS_KEYS}. "
            f"Available keys: {sorted(key for key in stats if not key.startswith('__'))}"
        )

    mean_full = torch.tensor(mean_values, device=device, dtype=torch.float32).view(1, 1, 3)
    std_full = torch.tensor(std_values, device=device, dtype=torch.float32).view(1, 1, 3)
    mean_sub = mean_full[..., 2:3]
    std_sub = std_full[..., 2:3]
    return mean_sub, std_sub, mean_full, std_full


def denormalize(tensor, mean, std):
    return tensor * std + mean


def apply_dry_grid_error_mask(diff, target_full_norm, mean_full, std_full):
    """Set the metric diff to zero on dry physical target nodes."""
    target_wl = denormalize(
        target_full_norm[..., WATER_LEVEL_CHANNEL],
        mean_full[..., WATER_LEVEL_CHANNEL],
        std_full[..., WATER_LEVEL_CHANNEL],
    )
    dry_mask = target_wl < DRY_WATER_LEVEL_THRESHOLD
    return diff.masked_fill(dry_mask.unsqueeze(-1), 0.0)


def strip_module_prefix(state_dict):
    if not all(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key[len("module."):]: value for key, value in state_dict.items()}


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def resolve_checkpoint_path(explicit, default_name):
    if explicit is not None:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"Checkpoint not found: {explicit}")
        return explicit

    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    candidates = [
        Path.cwd() / default_name,
        base_dir / default_name,
        script_dir / default_name,
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(
            "Checkpoint not found. Checked: " + ", ".join(str(p) for p in candidates)
        )
    return str(max(existing, key=lambda p: p.stat().st_mtime))


def load_checkpoint(model, ckpt_path, device, model_args):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        args_text = ", ".join(f"{key}={value}" for key, value in model_args.items())
        raise RuntimeError(
            f"Failed to load checkpoint {ckpt_path} with model args ({args_text}). "
            "Check for architecture or shape mismatch."
        ) from exc


def find_test_files(test_dir):
    return sorted(path for path in Path(test_dir).glob("*.pt") if not path.name.startswith("._"))


def evaluate_one_file(
    model,
    file_path,
    coords_2d_device,
    mean_sub,
    std_sub,
    mean_full,
    std_full,
    device,
):
    """Evaluate one pre-processed .pt file (one sample)."""
    entry = load_pt(file_path)
    storm = entry["storm"].to(device)
    inner = entry["inner"].to(device)
    target_full_norm = entry["target"].unsqueeze(0).to(device)  # (1, N, 3)

    x_in = coords_2d_device.unsqueeze(0)  # (1, N, 2)
    features = build_features(storm, inner).unsqueeze(0)  # (1, N, 120)
    target_norm_sub = target_full_norm[..., 2:3]  # (1, N, 1)

    with torch.no_grad():
        pred_norm = model(features, x_in)

    step_result = {}
    for metric_space in METRIC_SPACES:
        if metric_space == "physical":
            pred_metric = denormalize(pred_norm, mean_sub, std_sub)
            target_metric = denormalize(target_norm_sub, mean_sub, std_sub)
        else:
            pred_metric = pred_norm
            target_metric = target_norm_sub

        diff = pred_metric - target_metric
        diff = apply_dry_grid_error_mask(diff, target_full_norm, mean_full, std_full)

        sse = (diff ** 2).sum().item()
        sae = diff.abs().sum().item()
        sum_gt = target_metric.sum().item()
        sum_sq_gt = (target_metric ** 2).sum().item()

        l2_err = torch.norm(diff.reshape(1, -1), p=2, dim=1).item()
        l2_gt = max(torch.norm(target_metric.reshape(1, -1), p=2, dim=1).item(), 1e-8)
        rel_l2 = l2_err / l2_gt

        step_result[metric_space] = {
            "sse": sse, "sae": sae,
            "sum_gt": sum_gt, "sum_sq_gt": sum_sq_gt,
            "rel_l2": rel_l2,
        }

    return [{"step": 1, "metrics": step_result}]


def compute_aggregate_stats(all_results, num_nodes):
    """Aggregate per-file metrics."""
    by_space = {ms: {"sse": 0.0, "sae": 0.0, "sum_gt": 0.0, "sum_sq_gt": 0.0, "rel_l2_sum": 0.0, "count": 0}
                for ms in METRIC_SPACES}

    for file_results in all_results:
        for entry in file_results:
            for ms in METRIC_SPACES:
                m = entry["metrics"][ms]
                by_space[ms]["sse"] += m["sse"]
                by_space[ms]["sae"] += m["sae"]
                by_space[ms]["sum_gt"] += m["sum_gt"]
                by_space[ms]["sum_sq_gt"] += m["sum_sq_gt"]
                by_space[ms]["rel_l2_sum"] += m["rel_l2"]
                by_space[ms]["count"] += 1

    stats = {}
    for ms in METRIC_SPACES:
        b = by_space[ms]
        count = max(b["count"], 1)
        num_values = count * num_nodes
        mse = b["sse"] / num_values
        rmse = np.sqrt(mse)
        mae = b["sae"] / num_values
        ss_tot = max(b["sum_sq_gt"] - (b["sum_gt"] ** 2) / num_values, 1e-8)
        r2 = 1.0 - (b["sse"] / ss_tot)
        rel_l2 = b["rel_l2_sum"] / count
        stats[ms] = {
            "mse": mse, "rmse": rmse, "mae": mae, "r2": r2,
            "rel_l2": rel_l2, "count": b["count"],
        }
    return stats


def write_results(stats, output_path, model_path, total_files, evaluated_files, skipped_files):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("Geo-FNO Warm-up Single-Step Test Results\n")
        f.write(f"Metric Spaces: {', '.join(METRIC_SPACES)}\n")
        f.write(f"Channels: {CHANNEL_NAME}\n")
        f.write(f"Checkpoint: {model_path if model_path is not None else 'random weights'}\n")
        f.write(
            f"Total files: {total_files} "
            f"(evaluated: {evaluated_files}, skipped: {skipped_files})\n"
        )
        f.write("=" * 80 + "\n")
        f.write(f"{'Metric Space':<14} | {'MSE':<12} | {'RMSE':<12} | {'MAE':<12} | {'R2':<12} | {'Rel L2':<12} | {'N':<8}\n")
        f.write("-" * 80 + "\n")
        for ms in METRIC_SPACES:
            s = stats[ms]
            f.write(
                f"{ms:<14} | {s['mse']:<12.6f} | {s['rmse']:<12.6f} | "
                f"{s['mae']:<12.6f} | {s['r2']:<12.6f} | "
                f"{s['rel_l2']:<12.6f} | {s['count']:<8}\n"
            )
        f.write("=" * 80 + "\n")
    print(f"[test] results -> {output_path}")


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[test] device={device}")

    set_seed(3407)

    coords_2d_cpu = load_static_coords(args.coords)
    coords_2d_device = coords_2d_cpu.to(device)
    num_nodes = coords_2d_cpu.size(0)

    mean_sub, std_sub, mean_full, std_full = load_normalization_stats(args.norm, device=device)

    in_channels = C_IN
    out_channels = C_OUT
    model_args = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "modes": args.modes,
        "width": args.width,
        "s1": args.s1,
        "s2": args.s2,
        "num_fno_layers": args.num_fno_layers,
        "fc1_hidden": args.fc1_hidden,
    }
    model = GeoFNO2d(
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        in_channels=in_channels,
        out_channels=out_channels,
        s1=args.s1,
        s2=args.s2,
        num_fno_layers=args.num_fno_layers,
        fc1_hidden=args.fc1_hidden,
    ).to(device)
    print(f"[test] model params={sum(p.numel() for p in model.parameters()):,}")

    default_checkpoint = build_checkpoint_name()
    try:
        model_path = resolve_checkpoint_path(args.model, default_checkpoint)
    except FileNotFoundError:
        if not args.allow_random_weights:
            raise
        model_path = None

    if model_path is not None:
        print(f"[test] loading checkpoint {model_path}")
        load_checkpoint(model, model_path, device, model_args)
    else:
        print("[test] warning: using random weights (--allow_random_weights)")

    model.eval()
    test_files = find_test_files(args.test_dir)
    if args.num_files is not None:
        test_files = test_files[: args.num_files]
    if not test_files:
        raise FileNotFoundError(f"No .pt files in {args.test_dir}")

    print(f"[test] evaluating {len(test_files)} files")

    all_results = []
    for path in tqdm(test_files, desc="Test files"):
        file_results = evaluate_one_file(
            model, path, coords_2d_device,
            mean_sub, std_sub, mean_full, std_full,
            device,
        )
        all_results.append(file_results)

    stats = compute_aggregate_stats(all_results, num_nodes)

    for ms in METRIC_SPACES:
        s = stats[ms]
        print(
            f"[{ms}] N={s['count']} "
            f"mse={s['mse']:.6f} rmse={s['rmse']:.6f} "
            f"mae={s['mae']:.6f} r2={s['r2']:.6f} rel_l2={s['rel_l2']:.6f}"
        )

    write_results(
        stats, args.output,
        model_path,
        total_files=len(test_files),
        evaluated_files=len(test_files),
        skipped_files=0,
    )
    print("[test] done.")


if __name__ == "__main__":
    main()
