"""
Training engine.
Runs the PyTorch training loop in a thread and streams metrics
back to the WebSocket handler via a thread-safe asyncio.Queue.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    ExponentialLR,
)


# ── Message helpers ───────────────────────────────────────────────────────────

def _msg(type_: str, **kwargs: Any) -> dict:
    return {"type": type_, **kwargs}


# ── Single epoch helpers ──────────────────────────────────────────────────────

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    send: Callable[[dict], None],
    should_stop: Callable[[], bool],
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = len(loader)

    for batch_idx, (data, target) in enumerate(loader):
        if should_stop():
            return total_loss / max(batch_idx, 1)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        try:
            output = model(data)
        except RuntimeError as exc:
            msg = str(exc)
            if "mat1 and mat2 shapes cannot be multiplied" in msg or "size mismatch" in msg.lower():
                raise RuntimeError(
                    f"Shape mismatch in forward pass: {msg}. "
                    "Check that your Input node dimensions match your data "
                    "(for a CSV with N features set channels=N, height=1, width=1)."
                ) from exc
            if "out of memory" in msg.lower():
                raise RuntimeError(
                    "CUDA out of memory. Try: reduce batch size, use a smaller model, "
                    "or add more Dropout/pooling layers."
                ) from exc
            raise

        loss = criterion(output, target)

        loss_val = loss.item()
        if math.isnan(loss_val):
            raise RuntimeError(
                f"Loss became NaN at epoch {epoch}, batch {batch_idx + 1}. "
                "Try: lower the learning rate, add BatchNorm, check for zero-length "
                "sequences, or ensure your target labels are in the correct range."
            )
        if math.isinf(loss_val):
            raise RuntimeError(
                f"Loss exploded to Inf at epoch {epoch}, batch {batch_idx + 1}. "
                "Try: lower the learning rate or add gradient clipping."
            )

        loss.backward()

        # Warn on gradient explosion (but don't stop — user may have clipping)
        max_grad = max(
            (p.grad.abs().max().item() for p in model.parameters() if p.grad is not None),
            default=0.0,
        )
        if max_grad > 1e4:
            send(_msg("warning", message=f"Very large gradient detected ({max_grad:.1e}) at epoch {epoch} batch {batch_idx+1} — consider adding gradient clipping or lowering lr."))

        optimizer.step()

        total_loss += loss_val

        if batch_idx % max(1, total_batches // 20) == 0 or batch_idx == total_batches - 1:
            send(_msg(
                "batch",
                epoch=epoch,
                totalEpochs=total_epochs,
                batch=batch_idx + 1,
                totalBatches=total_batches,
                loss=round(total_loss / (batch_idx + 1), 5),
            ))

    return total_loss / total_batches


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_preds: bool = False,
) -> tuple[float, float, float | None, list[int] | None, list[int] | None]:
    """Returns (val_loss, val_acc, top5_acc_or_None, all_preds, all_targets)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += criterion(output, target).item()

        num_classes = output.size(1) if output.dim() > 1 else 1
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if collect_preds:
            all_preds.extend(pred.cpu().tolist())
            all_targets.extend(target.cpu().tolist())

        # Top-5 only makes sense when we have ≥5 classes
        if num_classes >= 5:
            top5 = output.topk(5, dim=1).indices
            correct_top5 += sum(t.item() in r.tolist() for t, r in zip(target, top5))

    val_loss = total_loss / len(loader)
    val_acc = correct / total if total else 0.0
    top5_acc = (correct_top5 / total) if (total > 0 and output.size(1) >= 5) else None

    return (
        val_loss,
        val_acc,
        top5_acc,
        all_preds if collect_preds else None,
        all_targets if collect_preds else None,
    )


# ── Scheduler factory ─────────────────────────────────────────────────────────

def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.optim.lr_scheduler.LRScheduler | ReduceLROnPlateau | None:
    scheduler_type: str = config.get("scheduler", "none")
    if scheduler_type == "step":
        step_size = int(config.get("schedulerStepSize", 5))
        gamma = float(config.get("schedulerGamma", 0.5))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "cosine":
        t_max = int(config.get("schedulerTMax", 10))
        return CosineAnnealingLR(optimizer, T_max=t_max)
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    elif scheduler_type == "exponential":
        gamma = float(config.get("schedulerGamma", 0.9))
        return ExponentialLR(optimizer, gamma=gamma)
    return None


def _get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return float(param_group["lr"])
    return 0.0


# ── Main training function (runs in a thread) ─────────────────────────────────

def run_training_sync(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    send: Callable[[dict], None],
    should_stop: Callable[[], bool],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    epochs: int = int(config.get("epochs", 10))
    lr: float = float(config.get("learningRate", 0.001))
    opt_name: str = config.get("optimizer", "adam").lower()
    loss_fn: str = config.get("lossFunction", "cross_entropy").lower()

    # Optimizer
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss function
    if loss_fn == "mse":
        criterion: nn.Module = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Scheduler (optional)
    scheduler = _make_scheduler(optimizer, config)
    has_plateau = isinstance(scheduler, ReduceLROnPlateau)

    send(_msg("status", message=f"Training on {device} · {epochs} epochs"
              + (f" · scheduler={config.get('scheduler', 'none')}" if scheduler else "")))

    epoch_times: list[float] = []

    for epoch in range(1, epochs + 1):
        if should_stop():
            send(_msg("stopped", epoch=epoch - 1, totalEpochs=epochs))
            return

        t0 = time.time()
        send(_msg("epochStart", epoch=epoch, totalEpochs=epochs))

        train_loss = _train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, epochs, send, should_stop,
        )

        if should_stop():
            send(_msg("stopped", epoch=epoch, totalEpochs=epochs))
            return

        is_last = epoch == epochs
        val_loss, val_acc, top5_acc, _, _ = _validate(
            model, val_loader, criterion, device,
            collect_preds=is_last,
        )

        # Step the scheduler
        if scheduler is not None:
            if has_plateau:
                scheduler.step(val_loss)  # type: ignore[arg-type]
            else:
                scheduler.step()  # type: ignore[union-attr]

        current_lr = _get_lr(optimizer)

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_time = sum(epoch_times) / len(epoch_times)
        eta_secs = int(avg_time * (epochs - epoch))

        epoch_msg: dict = dict(
            epoch=epoch,
            totalEpochs=epochs,
            trainLoss=round(train_loss, 5),
            valLoss=round(val_loss, 5),
            valAccuracy=round(val_acc, 5),
            epochTime=round(elapsed, 2),
            etaSecs=eta_secs,
            currentLR=current_lr,
        )
        if top5_acc is not None:
            epoch_msg["top5Accuracy"] = round(top5_acc, 5)

        send(_msg("epochEnd", **epoch_msg))

    # Final pass to collect confusion matrix
    _, _, _, final_preds, final_targets = _validate(
        model, val_loader, criterion, device, collect_preds=True
    )
    confusion_matrix: list[list[int]] | None = None
    if final_preds is not None and final_targets is not None:
        try:
            from sklearn.metrics import confusion_matrix as sk_cm  # type: ignore[import]
            labels = sorted(set(final_targets))
            cm = sk_cm(final_targets, final_preds, labels=labels)
            confusion_matrix = cm.tolist()
        except Exception:
            pass

    complete_msg: dict = {"totalEpochs": epochs}
    if confusion_matrix is not None:
        complete_msg["confusionMatrix"] = confusion_matrix

    send(_msg("complete", **complete_msg))


# ── Async wrapper ─────────────────────────────────────────────────────────────

async def run_training_async(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    queue: asyncio.Queue,
    stop_event: asyncio.Event,
) -> None:
    loop = asyncio.get_event_loop()

    def send(msg: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, msg)

    def should_stop() -> bool:
        return stop_event.is_set()

    await loop.run_in_executor(
        None,
        run_training_sync,
        model, train_loader, val_loader, config, send, should_stop,
    )
