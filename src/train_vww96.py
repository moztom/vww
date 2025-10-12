#!/usr/bin/env python3
import argparse, json, time, yaml, subprocess, platform, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small
from sklearn.metrics import classification_report, confusion_matrix
from torch.amp import GradScaler, autocast
from contextlib import nullcontext

# -----------------------
# 1) Helpers
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train VWW from pre-exported vww96/{train,val}/{0,1}")
    ap.add_argument("--data", type=Path, required=True, help="Path to vww96 root (contains train/ val/)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    return ap.parse_args()

def set_seed(seed=42):
    '''Set random seeds and deterministic pytorch for reproducibility.'''
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    
    # Faster, and reproducable on MPS (not strictly reproducible on CUDA)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def init_run_dir(cfg: dict, run_path: Path, device_str: str, tag: str = ""):
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S") + (f"_{tag}" if tag else "")
    run_dir = run_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)

    # Save system info
    sysinfo = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": device_str,
        "cuda_name": torch.cuda.get_device_name(0) if device_str == "cuda" and torch.cuda.is_available() else "NA",
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "seed": 42,
        "git_commit": try_cmd(["git", "rev-parse", "--short", "HEAD"]),
        "start_time": run_id
    }
    with open(run_dir / "system.txt", "w") as f:
        for k, v in sysinfo.items():
            f.write(f"{k}: {v}\n")

    # TensorBoard
    writer = SummaryWriter(log_dir=str(run_dir / "curves.tb"))
    return run_dir, writer


def try_cmd(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "NA"


def log_epoch(writer, run_dir: Path, epoch: int, *, tr_loss, tr_acc, va_loss, va_acc, lr=None):
    writer.add_scalar("loss/train", float(tr_loss), epoch)
    writer.add_scalar("acc/train", float(tr_acc), epoch)
    writer.add_scalar("loss/val", float(va_loss), epoch)
    writer.add_scalar("acc/val", float(va_acc), epoch)
    if lr is not None: writer.add_scalar("lr", float(lr), epoch)
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps({
            "epoch": epoch, "train_loss": float(tr_loss), "train_acc": float(tr_acc),
            "val_loss": float(va_loss), "val_acc": float(va_acc), "lr": None if lr is None else float(lr)
        }) + "\n")


# -----------------------
# 2) Data
# -----------------------
def build_loaders(data_root: Path, batch: int, num_workers: int):
    # 96x96 already; keep it simple. Add Resize(224) only if you choose a 224-pretrained backbone.

    # Computed with scripts/compute_mean_std.py
    MEAN, STD = (0.4699, 0.4469, 0.4077), (0.2653, 0.2618, 0.2768)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(), # randomly flip some training images
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_root / "val",   transform=val_tf)

    # Confirm expected mapping: folder "0" -> class 0, "1" -> class 1
    assert train_ds.class_to_idx == {"0": 0, "1": 1}, f"class mapping is {train_ds.class_to_idx}"


    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=False)
    
    # Loss weighting alternative (if not using sampler)
    '''
    labels = [y for _, y in train_ds.samples]
    counts = np.bincount(labels, minlength=2)
    total = float(sum(counts))
    # Balanced-ish: N/(2*n_c) so average weight ~1
    class_weight_tensor = torch.tensor([total/(2*counts[0]+1e-9), total/(2*counts[1]+1e-9)], dtype=torch.float32)
    '''

    class_weight_tensor = None  # data is near balanced, so don’t weight

    # No shuffle as we want determininism for val
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, class_weight_tensor

# -----------------------
# 3) Model
# -----------------------
def build_model(num_classes=2):
    # 96-native: use MobileNetV3-Small from scratch (weights=None)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes) # change final layer to 2 classes
    return model

# -----------------------
# 4) Train/Eval
# -----------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler: GradScaler | None):

    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for imgs, labels in loader:

        # Transfer images/labels to device (GPU/CPU/MPS)
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Zeroes gradients (with set_to_none=True for slightly better performance)
        optimizer.zero_grad(set_to_none=True)

        # Forward + backward + optimize
        use_amp = scaler is not None and device == "cuda"
        amp_context = autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        with amp_context:
            logits = model(imgs)                # forward pass
            loss = criterion(logits, labels)    # loss computed

        if scaler is None:
            # No loss scaling path (CPU BF16 or AMP disabled)
            loss.backward() # backward pass
            optimizer.step() # optimizer step
        else:
            # Loss scaling path (CUDA/MPS FP16)
            scaler.scale(loss).backward() # backward pass with loss scaling
            scaler.step(optimizer) # optimizer step
            scaler.update() # update scale for next iteration
            
        loss_sum += loss.item() * labels.size(0) # sum up batch loss
        correct += (logits.argmax(1) == labels).sum().item() # count correct
        total += labels.size(0) # count samples

    return loss_sum/total, correct/total # avg loss, accuracy

@torch.no_grad() # disable gradient calculation for evaluation
def evaluate(model, loader, device):

    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []
    ce = nn.CrossEntropyLoss(reduction="sum")  # compute sum to average manually
    
    for imgs, labels in loader:

        # Transfer images/labels to device (GPU/CPU/MPS)
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        logits = model(imgs) # forward pass
        loss_sum += ce(logits, labels).item() # sum up batch loss
        pred = logits.argmax(1) # predicted class
        correct += (pred == labels).sum().item() # count correct predictions
        total += labels.size(0) # count samples
        all_preds.append(pred.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = loss_sum/total
    acc = correct/total
    preds = np.concatenate(all_preds)
    gts = np.concatenate(all_labels)
    return avg_loss, acc, preds, gts

# -----------------------
# 5) Main
# -----------------------
def main():
    args = parse_args()
    set_seed(42)
    args.out.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg["device"] = device
    run_dir, writer = init_run_dir(cfg, args.out, device, tag="mbv3-baseline")

    train_loader, val_loader, class_weight_tensor = build_loaders(
        args.data, args.batch, args.num_workers
    )

    model = build_model(num_classes=2).to(device)

    # Define loss function (cross-entropy)
    # change to weight=None if not using sampler (i.e. class_weight_tensor is always None)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor.to(device) if class_weight_tensor is not None else None, label_smoothing=0.05)
    
    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Simple schedule; feels good for 10–30 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Enables automatic mixed precision. Set --no_amp to disable (slower, but worth trying))
    use_amp = (device == "cuda") and (not args.no_amp)
    scaler = GradScaler() if use_amp else None

    best_acc = 0.0
    best_epoch = 0
    best_path = run_dir / "model.pt"

    patience, max_patience = 0, 5  # early stopping

    # Training loop
    overall_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        va_loss, va_acc, preds, gts = evaluate(model, val_loader, device)
        scheduler.step()
        epoch_elapsed = time.perf_counter() - epoch_start
        elapsed_total = time.perf_counter() - overall_start

        log_epoch(writer, run_dir, epoch, tr_loss=tr_loss, tr_acc=tr_acc, va_loss=va_loss, va_acc=va_acc, lr=optimizer.param_groups[0]["lr"])

        print(f"[{epoch}/{args.epochs}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"epoch {epoch_elapsed:.1f}s | elapsed {elapsed_total/60:.1f}m")

        # Save best by val accuracy
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            patience = 0
            torch.save({"model": model.state_dict(),
                        "acc": va_acc,
                        "epoch": epoch}, best_path)
        else:
            patience += 1
            if patience >= max_patience:
                print(f"No improvement in {max_patience} epochs, stopping early")
                break
    
    total_elapsed = time.perf_counter() - overall_start

    # Final metrics on val
    va_loss, va_acc, preds, gts = evaluate(model, val_loader, device)
    print("\nValidation summary:")
    print(f"acc: {va_acc:.4f}  loss: {va_loss:.4f}")
    labels = [0, 1]
    target_names = ["no_person(0)", "person(1)"]
    cm = confusion_matrix(gts, preds, labels=labels)
    print("Confusion matrix (rows = true [0,1], cols = pred [0,1]):")
    print(cm)
    print("\nClassification report:")
    report = classification_report(gts, preds, labels=labels, target_names=target_names)
    print(report)
    print(f"\nBest checkpoint: {best_path} (acc={best_acc:.4f}) (epoch {best_epoch})")
    print(f"Total training time: {total_elapsed/60:.1f}mins ({total_elapsed:.1f}s)")

    # Save final metrics to metrics.jsonl
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"best_epoch": best_epoch, "best_val_acc": best_acc, "total_train_time": total_elapsed}) + "\n")

    # Save confusion matrix to JSON
    cm_path = run_dir / "confusion_matrix.json"
    with open(cm_path, "w") as f:
        json.dump({"key": "(rows = true [0,1], cols = pred [0,1])", "matrix": cm.tolist(), "target_names": target_names}, f)

    # Close TensorBoard writer
    writer.flush(); writer.close()

if __name__ == "__main__":
    main()
