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

    run_id = time.strftime("%Y-%m-%d_%H-%M-%S") + (f"_{config["meta"]["name"]}")
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


def log_epoch(writer, run_dir: Path, epoch: int, tr_loss, tr_acc, va_loss, va_acc, lr=None):
    """ Log epoch metrics to TensorBoard and JSONL file """

    writer.add_scalar("loss/train", float(tr_loss), epoch)
    writer.add_scalar("acc/train", float(tr_acc), epoch)
    writer.add_scalar("loss/val", float(va_loss), epoch)
    writer.add_scalar("acc/val", float(va_acc), epoch)
    if lr is not None: 
        writer.add_scalar("lr", float(lr), epoch)

    with open(run_dir / "metrics.jsonl", "a") as file:
        file.write(json.dumps({
            "epoch": epoch, "train_loss": float(tr_loss), "train_acc": float(tr_acc),
            "val_loss": float(va_loss), "val_acc": float(va_acc), "lr": None if lr is None else float(lr)
        }) + "\n")


def _try_cmd(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "NA"
