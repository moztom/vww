#!/usr/bin/env python3
import argparse, random, time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small
from sklearn.metrics import classification_report, confusion_matrix
from torch.amp import GradScaler, autocast
from contextlib import nullcontext

# -----------------------
# 1) CLI + Repro
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train VWW from pre-exported vww96/{train,val}/{0,1}")
    ap.add_argument("--data", type=Path, required=True, help="Path to vww96 root (contains train/ val/)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs/vww96"))
    ap.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    return ap.parse_args()

def set_seed(seed=42):
    '''Set random seeds and deterministic pytorch for reproducibility.'''
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # keep fast kernels
    torch.backends.cudnn.benchmark = True

# -----------------------
# 2) Data
# -----------------------
def build_loaders(data_root: Path, batch: int, num_workers: int):
    # 96x96 already; keep it simple. Add Resize(224) only if you choose a 224-pretrained backbone.

    # Computed with scripts/compute_mean_std.py
    MEAN, STD = (0.4699, 0.4469, 0.4077), (0.2653, 0.2618, 0.2768)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(), # randomly flip some training images
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_root / "val",   transform=val_tf)

    # Confirm expected mapping: folder "0" -> class 0, "1" -> class 1
    assert train_ds.class_to_idx == {"0": 0, "1": 1}, f"class mapping is {train_ds.class_to_idx}"


    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True)
    
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
                            num_workers=num_workers, pin_memory=True)

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

        '''
        if scaler is not None:
            with autocast(device_type=device, dtype=torch.float16):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with autocast(device_type=device, dtype=torch.float16):
                logits = model(imgs)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        '''
            
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

    train_loader, val_loader, class_weight_tensor = build_loaders(
        args.data, args.batch, args.num_workers
    )

    model = build_model(num_classes=2).to(device)

    # Define loss function (cross-entropy)
    # change to weight=None if not using sampler (i.e. class_weight_tensor is always None)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor.to(device) if class_weight_tensor is not None else None)

    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Simple schedule; feels good for 10–30 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Enables automatic mixed precision. Set --no_amp to disable (slower, but worth trying))
    use_amp = (device == "cuda") and (not args.no_amp)
    scaler = GradScaler() if use_amp else None

    best_acc, best_path = 0.0, args.out / "best.pt"

    # Training loop
    overall_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        va_loss, va_acc, preds, gts = evaluate(model, val_loader, device)
        scheduler.step()
        epoch_elapsed = time.perf_counter() - epoch_start
        elapsed_total = time.perf_counter() - overall_start

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"epoch {epoch_elapsed:.1f}s | elapsed {elapsed_total/60:.1f}m")

        # Save best by val accuracy
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(),
                        "acc": va_acc,
                        "epoch": epoch}, best_path)

    # Final metrics on val
    va_loss, va_acc, preds, gts = evaluate(model, val_loader, device)
    print("\nValidation summary:")
    print(f"acc: {va_acc:.4f}  loss: {va_loss:.4f}")
    print("Confusion matrix (rows = true [0,1], cols = pred [0,1]):")
    print(confusion_matrix(gts, preds))
    print("\nClassification report:")
    print(classification_report(gts, preds, target_names=["no_person(0)", "person(1)"]))

    print(f"\nBest checkpoint: {best_path} (acc={best_acc:.4f})")
    total_elapsed = time.perf_counter() - overall_start
    print(f"Total training time: {total_elapsed/60:.1f}mins ({total_elapsed:.1f}s)")

    # Save summary to results/
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"vww96_train_{timestamp}.txt"
    report_path = results_dir / run_name
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Train VWW96 run summary\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Device: {device}\n")
        f.write("CLI arguments:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
        f.write("\nMetrics:\n")
        f.write(f"  Final val acc: {va_acc:.4f}\n")
        f.write(f"  Final val loss: {va_loss:.4f}\n")
        f.write(f"  Best val acc: {best_acc:.4f}\n")
        f.write(f"  Best checkpoint: {best_path}\n")
        f.write(f"  Total time: {total_elapsed/60:.1f}m ({total_elapsed:.1f}s)\n")
    print(f"Saved run report to {report_path}")

if __name__ == "__main__":
    main()
