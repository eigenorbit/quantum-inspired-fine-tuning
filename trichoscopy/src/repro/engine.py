"""Train and evaluation loops.

One epoch of `train_one_epoch`, then `evaluate` over the non-augmented test
images. Nothing here is specific to a step or an architecture.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from .metrics import Predictions


def build_optimizer(model: nn.Module, config):
    """Adam with the learning rate and weight decay of Table S2."""
    if config.optimizer != "adam":
        raise ValueError(f"unsupported optimizer {config.optimizer!r}")
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer, config):
    """Step LR scheduler: multiply by `gamma` every `step_size` epochs."""
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
    )


def build_criterion(config):
    if config.loss != "cross_entropy":
        raise ValueError(f"unsupported loss {config.loss!r}")
    return nn.CrossEntropyLoss()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None,
                    log_every: int = 50, epoch: int | None = None) -> dict:
    """One pass over the training set. Returns mean loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    seen = 0
    started = time.perf_counter()

    for batch_index, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += batch_size

        if log_every and batch_index % log_every == 0:
            elapsed = time.perf_counter() - started
            rate = seen / elapsed if elapsed else 0.0
            prefix = f"epoch {epoch} " if epoch is not None else ""
            print(
                f"  {prefix}batch {batch_index + 1}/{len(loader)}  "
                f"loss {running_loss / seen:.4f}  acc {correct / seen:.4f}  "
                f"{rate:.0f} img/s",
                flush=True,
            )

    return {
        "loss": running_loss / seen,
        "accuracy": correct / seen,
        "seconds": time.perf_counter() - started,
        "images": seen,
    }


@torch.no_grad()
def evaluate(model, loader, device, criterion=None) -> tuple[Predictions, dict]:
    """Run the model over a loader; return predictions and summary statistics."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    running_loss = 0.0
    seen = 0
    started = time.perf_counter()

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        if criterion is not None:
            running_loss += criterion(outputs, labels).item() * labels.size(0)
        probabilities = torch.softmax(outputs, dim=1)

        all_true.append(labels.cpu().numpy())
        all_pred.append(outputs.argmax(dim=1).cpu().numpy())
        all_prob.append(probabilities.cpu().numpy())
        seen += labels.size(0)

    predictions = Predictions(
        y_true=np.concatenate(all_true),
        y_pred=np.concatenate(all_pred),
        probabilities=np.concatenate(all_prob),
    )
    stats = {
        "accuracy": float((predictions.y_true == predictions.y_pred).mean()),
        "loss": running_loss / seen if criterion is not None else None,
        "seconds": time.perf_counter() - started,
        "images": seen,
    }
    return predictions, stats
