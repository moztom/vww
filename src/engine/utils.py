import random, time, platform, subprocess, json
from pathlib import Path
import yaml

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


REPO_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed=42):
    """ Set random seeds and deterministic pytorch for reproducibility """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    
    # Faster, and reproducable on MPS (not strictly reproducible on CUDA)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def init_logging(config: dict, device: str):
    """ Create a new run directory with timestamp, save config and system info """

    # Check runs directory exists
    (REPO_ROOT / "runs").mkdir(exist_ok=True)

    run_id = time.strftime("%Y-%m-%d_%H-%M-%S") + (f"_{config['meta']['name']}")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.yaml", "w") as file:
        yaml.safe_dump(config, file, sort_keys=True)

    # Save system info
    sysinfo = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else "NA",
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "seed": 42,
        "git_commit": _try_cmd(["git", "rev-parse", "--short", "HEAD"]),
        "start_time": run_id
    }
    with open(run_dir / "system.txt", "w") as file:
        for k, v in sysinfo.items():
            file.write(f"{k}: {v}\n")

    # TensorBoard
    writer = SummaryWriter(log_dir=str(run_dir / "curves.tb"))
    return run_dir, writer


def log_epoch(
    writer,
    run_dir: Path,
    epoch: int,
    tr_loss,
    tr_acc,
    va_loss,
    va_acc,
    lr=None,
    ce=None,
    kl=None,
    alpha=None,
    margin=None,
    margin_weight=None,
):
    """ Log epoch metrics to TensorBoard and JSONL file """

    writer.add_scalar("train loss", float(tr_loss), epoch)
    if ce is not None:
        writer.add_scalar("ce_loss", float(ce), epoch)
    if kl is not None:
        writer.add_scalar("kl_loss", float(kl), epoch)
    if margin is not None:
        writer.add_scalar("margin_loss", float(margin), epoch)
    if margin_weight is not None:
        writer.add_scalar("margin_weight", float(margin_weight), epoch)
    writer.add_scalar("train acc", float(tr_acc), epoch)
    writer.add_scalar("val loss", float(va_loss), epoch)
    writer.add_scalar("val acc", float(va_acc), epoch)
    if lr is not None:
        writer.add_scalar("lr", float(lr), epoch)

    if (ce is not None) and (kl is not None) and (alpha is not None):
        with open(run_dir / "metrics.jsonl", "a") as file:
            file.write(json.dumps({
                "epoch": epoch,
                "alpha": float(alpha),
                "margin_weight": None if margin_weight is None else float(margin_weight),
                "train_loss": f"{float(tr_loss)} (ce {float(ce)}, kl {float(kl)}{'' if margin is None else f', margin {float(margin)}'})",
                "train_acc": float(tr_acc),
                "val_loss": float(va_loss),
                "val_acc": float(va_acc),
                "lr": float(lr) if lr is not None else None
            }) + "\n")
    else:
        with open(run_dir / "metrics.jsonl", "a") as file:
            file.write(json.dumps({
                "epoch": epoch, "train_loss": float(tr_loss), "train_acc": float(tr_acc),
                "val_loss": float(va_loss), "val_acc": float(va_acc), "lr": float(lr) if lr else None
            }) + "\n")


def save_checkpt(output_path, epoch, model, va_acc, save_full_checkpt, *, optimizer, scheduler, scaler, va_loss):
    """ Save model checkpoint """

    if save_full_checkpt:
        to_save = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "va_acc": va_acc,
            "va_loss": va_loss,
        }
    else:
        to_save = {
            "epoch": epoch,
            "model": model.state_dict(),
            "va_acc": va_acc
        }

    torch.save(
        to_save,
        output_path,
    )


def _try_cmd(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "NA"
