"""Training entrypoint for Geo-FNO storm-surge warm-up prediction model.

Single GPU:
    python model/main.py

Single-node multi-GPU DDP:
    torchrun --nproc_per_node=4 model/main.py

Before running, build split manifests explicitly:
    python scripts/build_manifest.py data/train
    python scripts/build_manifest.py data/val
    python scripts/build_manifest.py data/test
"""
from __future__ import annotations

import inspect
import os
import platform
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import (
    FileChunkedDistributedSampler,
    MultiStormSurgeDataset,
    load_static_coords,
)
from model import GeoFNO2d
from scheduler import build_scheduler
from temporal_utils import C_IN, C_OUT, build_checkpoint_name
from train import train_model


# ===================== CONFIG =====================
CONFIG = {
    "train_dir": "data/train",
    "val_dir": "data/val",
    "test_dir": "data/test",
    "coords_path": "data/coordinates.mat",
    "norm_path": "data/normalization.mat",
    "tb_dir": "runs",

    "seed": 42,

    "batch_size": 16,
    "num_workers": 4,
    "lru_files_per_worker": 2,

    "modes": 32,
    "width": 96,
    "s1": 64,
    "s2": 64,
    "num_fno_layers": 4,
    "fc1_hidden": 256,

    "num_epochs": 200,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "warmup_ratio": 0.05,
    "min_lr_ratio": 0.01,
    "grad_clip": 1.0,
    "accum_steps": 1,
    "loss_type": "rel_l2",
    "ema_decay": 0.999,
}
# ==================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def init_distributed() -> dict:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA, but CUDA is unavailable")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        rank = 0
        local_rank = 0
    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_rank0": rank == 0,
    }


def cleanup_distributed(dist_ctx: dict):
    if dist_ctx.get("distributed", False) and dist.is_initialized():
        dist.destroy_process_group()


def get_device(dist_ctx: dict) -> torch.device:
    if dist_ctx["distributed"]:
        return torch.device("cuda", dist_ctx["local_rank"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def per_device_batch_size(global_batch_size: int, dist_ctx: dict) -> int:
    world_size = dist_ctx["world_size"]
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"CONFIG['batch_size']={global_batch_size} must be divisible by "
            f"world_size={world_size}"
        )
    return global_batch_size // world_size


def rank0_print(dist_ctx: dict, *args, **kwargs):
    if dist_ctx["is_rank0"]:
        print(*args, **kwargs)


def build_manifest_commands(config: dict) -> list[str]:
    return [
        f"python scripts/build_manifest.py {config[split_key]}"
        for split_key in ("train_dir", "val_dir", "test_dir")
    ]


def preflight_manifest_files(config: dict) -> None:
    split_dirs = [
        Path(config[split_key])
        for split_key in ("train_dir", "val_dir")
    ]
    missing = [
        split_dir / "manifest.json"
        for split_dir in split_dirs
        if not (split_dir / "manifest.json").exists()
    ]
    if not missing:
        return

    missing_lines = "\n".join(f"  - {path}" for path in missing)
    command_lines = "\n".join(f"  {command}" for command in build_manifest_commands(config))
    raise FileNotFoundError(
        "Required manifest file(s) are missing; main.py does not scan data directories "
        "automatically.\n"
        f"Missing:\n{missing_lines}\n"
        "Build manifests before training:\n"
        f"{command_lines}"
    )


def main():
    dist_ctx = init_distributed()
    writer = None
    try:
        if CONFIG["accum_steps"] < 1:
            raise ValueError(f"accum_steps must be >= 1, got {CONFIG['accum_steps']}")
        preflight_manifest_files(CONFIG)
        set_seed(CONFIG["seed"])

        in_channels = C_IN
        out_channels = C_OUT
        checkpoint_name = build_checkpoint_name()
        run_tag = "GeoFNO_warmup_" + datetime.now().strftime("%Y%m%d-%H%M%S")

        device = get_device(dist_ctx)
        rank0_print(
            dist_ctx,
            f"[main] device={device}, distributed={dist_ctx['distributed']}, "
            f"world_size={dist_ctx['world_size']}",
        )
        rank0_print(
            dist_ctx,
            f"[main] input_window={in_channels // 5}h, "
            f"in_channels={in_channels}, out_channels={out_channels}",
        )
        rank0_print(dist_ctx, f"[main] checkpoint name: {checkpoint_name}")

        rank0_print(dist_ctx, f"[main] loading coords from {CONFIG['coords_path']}")
        coords_2d_cpu = load_static_coords(CONFIG["coords_path"])
        coords_2d_device = coords_2d_cpu.to(device)
        rank0_print(dist_ctx, f"[main] coords shape={tuple(coords_2d_cpu.shape)}")

        rank0_print(dist_ctx, f"[main] loading train manifest from {CONFIG['train_dir']}")
        train_dataset = MultiStormSurgeDataset(
            data_dir=CONFIG["train_dir"],
            lru_files_per_worker=CONFIG["lru_files_per_worker"],
        )
        rank0_print(dist_ctx, f"[main] loading val manifest from {CONFIG['val_dir']}")
        val_dataset = MultiStormSurgeDataset(
            data_dir=CONFIG["val_dir"],
            lru_files_per_worker=CONFIG["lru_files_per_worker"],
        )
        rank0_print(
            dist_ctx,
            f"[main] train samples={len(train_dataset)}, val samples={len(val_dataset)}",
        )
        rank0_print(dist_ctx, f"[main] nodes per sample={train_dataset.num_nodes}")

        batch_size = per_device_batch_size(CONFIG["batch_size"], dist_ctx)
        train_sampler = FileChunkedDistributedSampler(
            train_dataset,
            num_replicas=dist_ctx["world_size"],
            rank=dist_ctx["rank"],
            shuffle=True,
            seed=CONFIG["seed"],
            drop_last=True,
        )
        val_sampler = FileChunkedDistributedSampler(
            val_dataset,
            num_replicas=dist_ctx["world_size"],
            rank=dist_ctx["rank"],
            shuffle=False,
            seed=CONFIG["seed"],
            drop_last=False,
            pad_to_equal_length=False,
        )

        loader_kwargs = {"num_workers": CONFIG["num_workers"], "pin_memory": True}
        if CONFIG["num_workers"] > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            drop_last=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            drop_last=False,
            **loader_kwargs,
        )

        model = GeoFNO2d(
            modes1=CONFIG["modes"],
            modes2=CONFIG["modes"],
            width=CONFIG["width"],
            in_channels=in_channels,
            out_channels=out_channels,
            s1=CONFIG["s1"],
            s2=CONFIG["s2"],
            num_fno_layers=CONFIG["num_fno_layers"],
            fc1_hidden=CONFIG["fc1_hidden"],
        ).to(device)
        rank0_print(dist_ctx, f"[main] model params={sum(p.numel() for p in model.parameters()):,}")

        if dist_ctx["distributed"]:
            model = DDP(
                model,
                device_ids=[dist_ctx["local_rank"]],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )
        optimizer_steps_per_epoch = len(train_loader) // CONFIG["accum_steps"]
        if optimizer_steps_per_epoch < 1:
            raise ValueError(
                f"accum_steps={CONFIG['accum_steps']} too large for "
                f"steps_per_epoch={len(train_loader)}"
            )
        scheduler = build_scheduler(
            optimizer,
            num_epochs=CONFIG["num_epochs"],
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            warmup_ratio=CONFIG["warmup_ratio"],
            min_lr_ratio=CONFIG["min_lr_ratio"],
        )
        total_steps = CONFIG["num_epochs"] * optimizer_steps_per_epoch
        warmup_steps = int(CONFIG["warmup_ratio"] * total_steps)
        rank0_print(
            dist_ctx,
            f"[main] Cosine: total_steps={total_steps}, warmup_steps={warmup_steps}, "
            f"min_lr={CONFIG['lr'] * CONFIG['min_lr_ratio']:.2e}",
        )

        if dist_ctx["is_rank0"]:
            tb_run_dir = os.path.join(CONFIG["tb_dir"], run_tag)
            os.makedirs(tb_run_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_run_dir)
            rank0_print(dist_ctx, f"[main] tensorboard log dir={tb_run_dir}")

            config_md = "### Training Configuration\n| Parameter | Value |\n|---|---|\n"
            for key, value in CONFIG.items():
                config_md += f"| {key} | {value} |\n"
            config_md += f"| in_channels | {in_channels} |\n"
            config_md += f"| out_channels | {out_channels} |\n"
            config_md += "\n### System\n| Parameter | Value |\n|---|---|\n"
            config_md += f"| OS | {platform.system()} {platform.release()} |\n"
            config_md += f"| CPU Cores | {os.cpu_count()} |\n"
            config_md += f"| World Size | {dist_ctx['world_size']} |\n"
            try:
                config_md += f"| GPU | {torch.cuda.get_device_name(0)} |\n"
            except Exception:
                pass
            writer.add_text("config/all", config_md, 0)

        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            num_epochs=CONFIG["num_epochs"],
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            coords_2d_device=coords_2d_device,
            writer=writer,
            grad_clip=CONFIG["grad_clip"],
            loss_type=CONFIG["loss_type"],
            ema_decay=CONFIG.get("ema_decay"),
            checkpoint_path=checkpoint_name,
            train_sampler=train_sampler,
            dist_ctx=dist_ctx,
            accum_steps=CONFIG["accum_steps"],
        )

        rank0_print(dist_ctx, "[main] done.")

    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
