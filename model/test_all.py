"""Autoregressive multi-step test for Geo-FNO bundle model across a test split (h-only)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io
import torch
from tqdm import tqdm

from dataset import load_static_coords
from main import set_seed
from model import GeoFNO2d
from temporal_utils import (
    CHANNEL_NAME,
    build_checkpoint_name,
    input_channels_for_bundle,
    output_channels_for_bundle,
    validate_temporal_params,
)


METRIC_SPACES = ("physical", "normalized")
WATER_LEVEL_CHANNEL = 2
DRY_WATER_LEVEL_THRESHOLD = 0.005
REQUIRED_KEYS = ("graph", "storm_boundary", "inner_boundary")
GRAPH_STATS_KEYS = ("graph_mean", "graph_std")
LEGACY_STATS_KEYS = ("u_mean", "u_std", "v_mean", "v_std", "h_mean", "h_std")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Geo-FNO autoregressive test across a split (h-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_dir", type=str, default="data/test", help="Path to test split directory.")
    parser.add_argument("--coords", type=str, default="data/coordinates.mat", help="Path to coordinates.mat.")
    parser.add_argument("--norm", type=str, default="data/normalization.mat", help="Path to normalization.mat.")
    parser.add_argument("--model", type=str, default=None, help="Checkpoint path.")
    parser.add_argument("--output", type=str, default="geofno_autoregressive_results.txt", help="Base output path.")
    parser.add_argument("--max_rollout", type=int, default=72, help="Max autoregressive rollout length per file (steps).")
    parser.add_argument("--num_files", type=int, default=None, help="Limit number of files for smoke tests.")
    parser.add_argument("--bundle_size", type=int, default=8, help="Bundle prediction size.")
    parser.add_argument("--allow_random_weights", action="store_true", help="Run without a checkpoint.")
    parser.add_argument("--modes", type=int, default=16, help="Fourier modes per axis.")
    parser.add_argument("--width", type=int, default=32, help="Model width.")
    parser.add_argument("--s1", type=int, default=64, help="Internal grid size along axis 1.")
    parser.add_argument("--s2", type=int, default=64, help="Internal grid size along axis 2.")
    parser.add_argument("--num_fno_layers", type=int, default=3, help="Number of FNO layers.")
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
    """Legacy eval: set only the metric diff to zero on dry physical target nodes."""
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
        raise FileNotFoundError("Checkpoint not found. Checked: " + ", ".join(str(p) for p in candidates))
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


def init_bucket(device):
    return {
        "sse": torch.zeros(1, device=device),
        "sae": torch.zeros(1, device=device),
        "sum_gt": torch.zeros(1, device=device),
        "sum_sq_gt": torch.zeros(1, device=device),
        "rel_l2_sum": torch.zeros(1, device=device),
        "count": 0,
    }


def compute_stats(bucket, num_nodes):
    count = bucket["count"]
    if count == 0:
        zeros = np.zeros(1, dtype=np.float64)
        return {
            "mse_channels": zeros,
            "rmse_channels": zeros,
            "mae_channels": zeros,
            "r2_channels": zeros,
            "rel_l2_channels": zeros,
        }

    num_values = count * num_nodes
    sse = bucket["sse"]
    sae = bucket["sae"]
    mse_channels = sse / num_values
    rmse_channels = np.sqrt(mse_channels)
    mae_channels = sae / num_values
    ss_tot = bucket["sum_sq_gt"] - (bucket["sum_gt"] ** 2) / num_values
    ss_tot = np.maximum(ss_tot, 1e-8)
    r2_channels = 1.0 - (sse / ss_tot)
    rel_l2_channels = bucket["rel_l2_sum"] / count
    return {
        "mse_channels": mse_channels,
        "rmse_channels": rmse_channels,
        "mae_channels": mae_channels,
        "r2_channels": r2_channels,
        "rel_l2_channels": rel_l2_channels,
    }


def compute_auc(results):
    auc = {"h": {}}
    steps = [entry["step"] for entry in results]
    if len(steps) < 2:
        for metric in ("mse", "rmse", "mae", "r2", "rel_l2"):
            auc["h"][metric] = 0.0
        return auc

    metrics = [key for key in results[0]["h"].keys() if key != "step"]
    for metric in metrics:
        values = [entry["h"][metric] for entry in results]
        auc["h"][metric] = float(np.trapz(values, steps))
    return auc


def build_features_batch(state_t, storm_window, inner_window, btype_oh):
    """Build batched node features for one autoregressive bundle block."""
    batch_size, _, num_nodes, _ = storm_window.shape
    storm_flat = storm_window.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
    inner_flat = inner_window.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
    btype_batch = btype_oh.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.cat([state_t, storm_flat, inner_flat, btype_batch], dim=-1).contiguous()


def find_test_files(test_dir):
    return sorted(path for path in Path(test_dir).glob("*.pt") if not path.name.startswith("._"))


def prescan_files(file_paths, bundle_size, max_rollout):
    """Pass-1 scan: load each .pt to read graph T, classify evaluable vs skipped.

    Returns:
        evaluable: list of (path, T, target_steps) for files with T > bundle_size
        skipped: list of (path, T) for files with T <= bundle_size
        bucket_len: max target_steps across evaluable files (0 if none)
    """
    evaluable = []
    skipped = []
    for path in tqdm(file_paths, desc="Pre-scan T"):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if "graph" not in data:
            raise KeyError(f"{path}: missing 'graph' key; got {list(data.keys())}")
        num_time = int(data["graph"].shape[0])
        del data
        if num_time <= bundle_size:
            skipped.append((path, num_time))
            continue
        target_steps = min(max_rollout, num_time - 1)
        evaluable.append((path, num_time, target_steps))
    bucket_len = max((target_steps for _, _, target_steps in evaluable), default=0)
    return evaluable, skipped, bucket_len


def load_event_file(file_path, expected_nodes):
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    for key in REQUIRED_KEYS:
        if key not in data:
            raise KeyError(f"{file_path}: missing key {key!r}; got {list(data.keys())}")

    graph = data["graph"].float()
    storm = data["storm_boundary"].float()
    inner = data["inner_boundary"].float()

    if graph.dim() != 3 or graph.size(-1) != 3:
        raise ValueError(f"{file_path}: graph must be (T,N,3), got {tuple(graph.shape)}")
    if storm.shape != graph.shape:
        raise ValueError(f"{file_path}: storm_boundary {tuple(storm.shape)} != graph {tuple(graph.shape)}")
    if (
        inner.dim() != 3
        or inner.size(0) != graph.size(0)
        or inner.size(1) != graph.size(1)
        or inner.size(-1) != 2
    ):
        raise ValueError(
            f"{file_path}: inner_boundary {tuple(inner.shape)} incompatible with graph "
            f"{tuple(graph.shape)}; expected (T,N,2)"
        )
    if graph.size(1) != expected_nodes:
        raise ValueError(f"{file_path}: N={graph.size(1)} != coordinates N={expected_nodes}")

    return graph, storm, inner


def autoregressive_one_file(
    model,
    file_path,
    coords_2d_device,
    btype_oh_device,
    mean_sub,
    std_sub,
    mean_full,
    std_full,
    device,
    target_steps,
    bundle_size,
    per_step_metrics_by_space,
):
    """Run a single autoregressive rollout (h-only)."""
    graph_all, storm_all, inner_all = load_event_file(file_path, coords_2d_device.size(0))
    num_time = graph_all.size(0)
    if target_steps > num_time - 1:
        raise ValueError(
            f"{file_path}: target_steps={target_steps} exceeds T-1={num_time - 1}"
        )

    x_in = coords_2d_device.unsqueeze(0)
    real_start = graph_all[0:1, :, 2:3].to(device)
    predictions = [None] * target_steps

    covered = 0
    with torch.no_grad():
        while covered < target_steps:
            remaining = target_steps - covered
            if remaining >= bundle_size:
                input_rel = covered
                input_state = real_start if covered == 0 else predictions[covered - 1]
                use_full_block = True
            else:
                input_rel = target_steps - bundle_size
                input_state = predictions[input_rel - 1]
                use_full_block = False

            storm_window = storm_all[input_rel : input_rel + bundle_size + 1].unsqueeze(0).to(device)
            inner_window = inner_all[input_rel : input_rel + bundle_size + 1].unsqueeze(0).to(device)
            features = build_features_batch(input_state, storm_window, inner_window, btype_oh_device)
            bundle_out = model(features, x_in)

            if use_full_block:
                for i in range(bundle_size):
                    predictions[covered + i] = bundle_out[:, i]
                covered += bundle_size
            else:
                for j in range(remaining):
                    bundle_idx = bundle_size - remaining + j
                    predictions[covered + j] = bundle_out[:, bundle_idx]
                covered = target_steps

    with torch.no_grad():
        for step in range(target_steps):
            rel_idx = step + 1
            pred_norm = predictions[step]
            target_full_norm = graph_all[rel_idx : rel_idx + 1].to(device)
            target_norm_sub = target_full_norm[..., 2:3]

            for metric_space in METRIC_SPACES:
                if metric_space == "physical":
                    pred_metric = denormalize(pred_norm, mean_sub, std_sub)
                    target_metric = denormalize(target_norm_sub, mean_sub, std_sub)
                else:
                    pred_metric = pred_norm
                    target_metric = target_norm_sub

                diff = pred_metric - target_metric
                diff = apply_dry_grid_error_mask(diff, target_full_norm, mean_full, std_full)
                bucket = per_step_metrics_by_space[metric_space][step]
                bucket["sse"] += torch.sum(diff ** 2, dim=(0, 1))
                bucket["sae"] += torch.sum(torch.abs(diff), dim=(0, 1))
                bucket["sum_gt"] += torch.sum(target_metric, dim=(0, 1))
                bucket["sum_sq_gt"] += torch.sum(target_metric ** 2, dim=(0, 1))

                l2_err = torch.norm(diff.permute(0, 2, 1), p=2, dim=2)
                l2_gt = torch.norm(target_metric.permute(0, 2, 1), p=2, dim=2).clamp(min=1e-8)
                bucket["rel_l2_sum"] += (l2_err / l2_gt).sum(dim=0)
                bucket["count"] += 1


def write_results(
    results_by_space,
    output_path,
    max_rollout,
    bundle_size,
    model_path,
    total_files,
    evaluated_files,
    skipped_files,
):
    for metric_space in METRIC_SPACES:
        out = metric_output_path(output_path, metric_space)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write("Autoregressive Test Results\n")
            f.write(f"Max rollout: {max_rollout}\n")
            f.write(f"Bundle size: {bundle_size}\n")
            f.write(f"Channels: {CHANNEL_NAME}\n")
            f.write(f"Checkpoint: {model_path if model_path is not None else 'random weights'}\n")
            f.write(f"Metric Space: {metric_space}\n")
            f.write(
                f"Total files: {total_files} "
                f"(evaluated: {evaluated_files}, skipped: {skipped_files})\n"
            )
            f.write("=" * 110 + "\n")
            f.write(
                f"{'Step':<6} | {'Channel':<7} | {'MSE':<12} | {'RMSE':<12} | "
                f"{'MAE':<12} | {'R2':<12} | {'Rel L2':<12} | {'N':<6}\n"
            )
            f.write("-" * 110 + "\n")

            for result in results_by_space[metric_space]:
                step = result["step"]
                count = result["count"]
                metrics = result["h"]
                f.write(
                    f"{step:<6} | {'wl':<7} | {metrics['mse']:<12.6f} | "
                    f"{metrics['rmse']:<12.6f} | {metrics['mae']:<12.6f} | "
                    f"{metrics['r2']:<12.6f} | {metrics['rel_l2']:<12.6f} | {count:<6}\n"
                )
                f.write("-" * 110 + "\n")

            auc = compute_auc(results_by_space[metric_space])
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"AUC Summary Over {len(results_by_space[metric_space])} Steps\n")
            f.write("-" * 100 + "\n")
            f.write(
                f"{'Channel':<11} | {'MSE Area':<12} | {'RMSE Area':<12} | "
                f"{'MAE Area':<12} | {'R2 Area':<12} | {'Rel L2 Area':<12}\n"
            )
            f.write("-" * 100 + "\n")
            metrics = auc["h"]
            f.write(
                f"{'wl':<11} | {metrics['mse']:<12.6f} | {metrics['rmse']:<12.6f} | "
                f"{metrics['mae']:<12.6f} | {metrics['r2']:<12.6f} | "
                f"{metrics['rel_l2']:<12.6f}\n"
            )
            f.write("=" * 100 + "\n")
        print(f"[test] results -> {out}")


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[test] device={device}")

    set_seed(3407)
    validate_temporal_params(args.bundle_size)
    if args.max_rollout < args.bundle_size:
        raise ValueError(
            f"max_rollout must be >= bundle_size: max_rollout={args.max_rollout}, "
            f"bundle_size={args.bundle_size}"
        )

    coords_2d_cpu, btype_oh_cpu = load_static_coords(args.coords)
    coords_2d_device = coords_2d_cpu.to(device)
    btype_oh_device = btype_oh_cpu.to(device)
    num_nodes = coords_2d_cpu.size(0)

    mean_sub, std_sub, mean_full, std_full = load_normalization_stats(args.norm, device=device)

    in_channels = input_channels_for_bundle(args.bundle_size)
    out_channels = output_channels_for_bundle(args.bundle_size)
    model_args = {
        "bundle_size": args.bundle_size,
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

    default_checkpoint = build_checkpoint_name(args.bundle_size)
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

    evaluable, skipped, bucket_len = prescan_files(
        test_files, args.bundle_size, args.max_rollout
    )
    print(
        f"[test] prescan: total={len(test_files)} "
        f"evaluable={len(evaluable)} skipped={len(skipped)} bucket_len={bucket_len}"
    )
    if not evaluable:
        raise RuntimeError(
            f"No files eligible for evaluation. total={len(test_files)}, "
            f"skipped={len(skipped)}, bundle_size={args.bundle_size}"
        )

    per_step_metrics_by_space = {
        metric_space: [init_bucket(device) for _ in range(bucket_len)]
        for metric_space in METRIC_SPACES
    }

    for path, _T, target_steps in tqdm(evaluable, desc="Test files"):
        autoregressive_one_file(
            model,
            path,
            coords_2d_device,
            btype_oh_device,
            mean_sub,
            std_sub,
            mean_full,
            std_full,
            device,
            target_steps,
            args.bundle_size,
            per_step_metrics_by_space,
        )

    for metric_space in METRIC_SPACES:
        for step in range(bucket_len):
            bucket = per_step_metrics_by_space[metric_space][step]
            for key in ("sse", "sae", "sum_gt", "sum_sq_gt", "rel_l2_sum"):
                bucket[key] = bucket[key].detach().cpu().numpy()

    results_by_space = {metric_space: [] for metric_space in METRIC_SPACES}
    for metric_space in METRIC_SPACES:
        for step, bucket in enumerate(per_step_metrics_by_space[metric_space]):
            stats = compute_stats(bucket, num_nodes)
            result = {"step": step + 1, "count": int(bucket["count"])}
            result["h"] = {
                "mse": float(stats["mse_channels"][0]),
                "rmse": float(stats["rmse_channels"][0]),
                "mae": float(stats["mae_channels"][0]),
                "r2": float(stats["r2_channels"][0]),
                "rel_l2": float(stats["rel_l2_channels"][0]),
            }
            results_by_space[metric_space].append(result)

            metrics = result["h"]
            print(
                f"[step {step + 1:02d}][{metric_space}][N={int(bucket['count'])}] "
                f"wl: mse={metrics['mse']:.6f} rmse={metrics['rmse']:.6f} "
                f"mae={metrics['mae']:.6f} r2={metrics['r2']:.6f} "
                f"rel_l2={metrics['rel_l2']:.6f}"
            )

    write_results(
        results_by_space,
        args.output,
        args.max_rollout,
        args.bundle_size,
        model_path,
        total_files=len(test_files),
        evaluated_files=len(evaluable),
        skipped_files=len(skipped),
    )
    print("[test] done.")


if __name__ == "__main__":
    main()
