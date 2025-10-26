import numpy as np
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_


def train_one_epoch(
    model,
    loader,
    device,
    optimizer,
    criterion,
    scheduler,
    scaler,
    autocast,
    grad_clip_norm,
    ema=None,
):
    """Train the model for one epoch"""

    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    nonblock = torch.cuda.is_available()

    # Iterate over batches
    for imgs, labels in loader:

        # Transfer images/labels to device (GPU/CPU/MPS)
        imgs, labels = imgs.to(device, non_blocking=nonblock), labels.to(
            device, non_blocking=nonblock
        )

        # Zeroes gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward
        with autocast:
            logits = model(imgs)  # forward pass
            loss = criterion(logits, labels)  # loss computed

        # Backward
        if scaler: # CUDA
            scaler.scale(loss).backward()  # backward pass with loss scaling
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip_norm) # clip grad norms
            scaler.step(optimizer)  # optimizer step
            scaler.update()  # update scaler for next iteration
        else: # MPS and CPU
            loss.backward()  # backward pass
            if grad_clip_norm and grad_clip_norm > 0:
                clip_grad_norm_(model.parameters(), grad_clip_norm) # clip grad norms
            optimizer.step()  # optimizer step

        if scheduler:
            scheduler.step()

        if ema:
            ema.update(model)

        loss_sum += loss.item() * labels.size(0)  # sum up batch loss
        correct += (logits.argmax(1) == labels).sum().item()  # count correct
        total += labels.size(0)  # count samples

    tr_loss = loss_sum / total
    tr_acc = correct / total

    return tr_loss, tr_acc


@torch.no_grad()
def evaluate(model, loader, device, metrics=False, teacher=False):
    """Evaluate the model on the validation set"""

    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []
    ce = CrossEntropyLoss()
    nonblock = torch.cuda.is_available()

    for imgs, labels in loader:

        # Transfer images/labels to device (GPU/CPU/MPS)
        imgs, labels = imgs.to(device, non_blocking=nonblock), labels.to(
            device, non_blocking=nonblock
        )

        if teacher:
            imgs = F.interpolate(imgs, size=224, mode='bilinear', align_corners=False)

        logits = model(imgs)  # forward pass
        loss_sum += ce(logits, labels).item() * labels.size(0)  # sum up batch loss
        correct += ((logits.argmax(1) == labels).sum().item())  # count correct predictions
        total += labels.size(0)  # count samples

        if metrics:
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = loss_sum / total
    acc = correct / total

    preds = np.concatenate(all_preds) if metrics else None
    gts = np.concatenate(all_labels) if metrics else None

    return avg_loss, acc, preds, gts
