"""Training loop for Geo-FNO warm-up prediction model."""
from __future__ import annotations

import copy
from contextlib import nullcontext

import torch
import torch.distributed as dist
from tqdm import tqdm


def is_distributed(dist_ctx: dict | None) -> bool:
    return bool(dist_ctx and dist_ctx.get("distributed", False))


def is_rank0(dist_ctx: dict | None) -> bool:
    return dist_ctx is None or dist_ctx.get("is_rank0", True)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_sums(values, device, dist_ctx: dict | None):
    totals = torch.tensor(values, dtype=torch.float64, device=device)
    if is_distributed(dist_ctx):
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals.cpu().tolist()


def barrier_if_distributed(dist_ctx: dict | None):
    if is_distributed(dist_ctx):
        dist.barrier()


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float):
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"EMA decay must satisfy 0 <= decay < 1, got {decay}")
        self.decay = float(decay)
        self.live = unwrap_model(model)
        self.shadow = copy.deepcopy(self.live).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module | None = None):
        live = self.live if model is None else unwrap_model(model)
        keep = self.decay
        blend = 1.0 - keep
        for shadow_param, live_param in zip(self.shadow.parameters(), live.parameters(), strict=True):
            shadow_param.mul_(keep).add_(live_param, alpha=blend)
        for shadow_buffer, live_buffer in zip(self.shadow.buffers(), live.buffers(), strict=True):
            shadow_buffer.copy_(live_buffer)


class RMSELoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)


def rel_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Single-channel relative L2 loss, averaged over batch."""
    diff = (pred - target).reshape(pred.size(0), -1)
    base = target.reshape(pred.size(0), -1)
    num = torch.linalg.vector_norm(diff, ord=2, dim=1)
    den = torch.linalg.vector_norm(base, ord=2, dim=1).clamp(min=eps)
    return (num / den).mean()


def evaluate_model(model, test_loader, device, coords_2d_device, dist_ctx: dict | None = None):
    """Single-step evaluation in normalized space (h-only)."""
    model.eval()
    total_sse = 0.0
    total_sae = 0.0
    total_rel_l2 = 0.0
    num_samples = 0
    total_elements = 0

    x_in_base = coords_2d_device.to(device, non_blocking=True).unsqueeze(0)
    with torch.no_grad():
        for features, target in test_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            batch_size = features.shape[0]
            x_in = x_in_base.expand(batch_size, -1, -1)

            pred = model(features, x_in)
            diff = pred - target

            total_sse += (diff ** 2).sum().item()
            total_sae += diff.abs().sum().item()

            diff_flat = diff.reshape(batch_size, -1)
            target_flat = target.reshape(batch_size, -1)
            diff_norm = torch.linalg.vector_norm(diff_flat, ord=2, dim=1)
            target_norm = torch.linalg.vector_norm(target_flat, ord=2, dim=1).clamp(min=1e-8)
            total_rel_l2 += (diff_norm / target_norm).sum().item()

            num_samples += batch_size
            total_elements += target.numel()

    totals = reduce_sums(
        [total_sse, total_sae, total_rel_l2, num_samples, total_elements],
        device,
        dist_ctx,
    )
    sse, sae, rel_l2, sample_count, element_count = totals
    sample_count = max(1.0, sample_count)
    element_count = max(1.0, element_count)
    mse = sse / element_count
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": sae / element_count,
        "rel_l2": rel_l2 / sample_count,
    }


def _ddp_sync_context(model, should_sync: bool, dist_ctx: dict | None):
    if should_sync or not is_distributed(dist_ctx):
        return nullcontext()
    if not hasattr(model, "no_sync"):
        return nullcontext()
    return model.no_sync()


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs,
    device,
    optimizer,
    scheduler,
    coords_2d_device,
    writer,
    grad_clip=None,
    loss_type: str = "rel_l2",
    ema_decay: float | None = None,
    checkpoint_path: str = "best_geofno.pt",
    train_sampler=None,
    dist_ctx: dict | None = None,
    accum_steps: int = 1,
):
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")
    if loss_type == "rmse":
        criterion = RMSELoss()
    elif loss_type == "rel_l2":
        criterion = None
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    ema = None
    if ema_decay is not None:
        ema = ExponentialMovingAverage(model, decay=ema_decay)

    global_step = 0
    best_loss = float("inf")
    x_in_base = coords_2d_device.to(device, non_blocking=True).unsqueeze(0)

    for epoch in range(num_epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        model.train()
        local_loss_sum = 0.0
        local_n = 0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False,
            disable=not is_rank0(dist_ctx),
        )

        steps_per_epoch = len(train_loader)
        optimizer_steps_per_epoch = steps_per_epoch // accum_steps
        usable_micro_batches = optimizer_steps_per_epoch * accum_steps

        optimizer.zero_grad(set_to_none=True)

        for micro_idx, (features, target_block) in enumerate(pbar):
            if micro_idx >= usable_micro_batches:
                break

            features = features.to(device, non_blocking=True)
            target_block = target_block.to(device, non_blocking=True)
            batch_size = features.shape[0]

            should_sync = (micro_idx + 1) % accum_steps == 0
            with _ddp_sync_context(model, should_sync, dist_ctx):
                x_in = x_in_base.expand(batch_size, -1, -1)
                pred = model(features, x_in)
                if loss_type == "rmse":
                    loss = criterion(pred, target_block)
                else:
                    loss = rel_l2_loss(pred, target_block)
                loss = loss / accum_steps
                loss.backward()

            loss_unscaled = loss.item() * accum_steps
            local_loss_sum += loss_unscaled * batch_size
            local_n += batch_size

            if should_sync:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

                if is_rank0(dist_ctx) and writer is not None:
                    writer.add_scalar("train/loss_step", loss_unscaled, global_step)
                    writer.add_scalar("train/lr_step", optimizer.param_groups[0]["lr"], global_step)
                global_step += 1

            if is_rank0(dist_ctx):
                pbar.set_postfix({"loss": f"{loss_unscaled:.6f}"})

        global_loss_sum, global_n = reduce_sums([local_loss_sum, local_n], device, dist_ctx)
        avg_loss = global_loss_sum / max(1.0, global_n)
        if is_rank0(dist_ctx) and writer is not None:
            writer.add_scalar("train/loss_epoch", avg_loss, epoch)

        eval_model = ema.shadow if ema is not None else model
        test_metrics = evaluate_model(
            eval_model,
            test_loader,
            device,
            coords_2d_device,
            dist_ctx=dist_ctx,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        if is_rank0(dist_ctx):
            if writer is not None:
                writer.add_scalar("val/loss_epoch", test_metrics["rel_l2"], epoch)
                writer.add_scalar("val/rel_l2", test_metrics["rel_l2"], epoch)
                writer.add_scalar("val/mse", test_metrics["mse"], epoch)
                writer.add_scalar("val/rmse", test_metrics["rmse"], epoch)
                writer.add_scalar("val/mae", test_metrics["mae"], epoch)
                writer.add_scalar("train/lr", current_lr, epoch)
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {avg_loss:.6f} | "
                f"Test RMSE: {test_metrics['rmse']:.6f} | "
                f"Test Rel-L2: {test_metrics['rel_l2']:.6f} | "
                f"LR: {current_lr:.2e}"
            )

        current_test_loss = test_metrics["rmse"] if loss_type == "rmse" else test_metrics["rel_l2"]
        if current_test_loss < best_loss:
            best_loss = current_test_loss
            if is_rank0(dist_ctx):
                save_target = ema.shadow if ema is not None else unwrap_model(model)
                torch.save(save_target.state_dict(), checkpoint_path)
                print(f"  -> Saved best model to {checkpoint_path} (metric={best_loss:.6f})")

        barrier_if_distributed(dist_ctx)

    if is_rank0(dist_ctx):
        print("Training finished.")
