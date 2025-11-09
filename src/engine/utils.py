import os
import random, time, platform, subprocess, json
from pathlib import Path
from typing import Dict, Optional
import yaml

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from thop import profile


REPO_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed=42):
    """Set random seeds and make CUDA execution deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    info = {}

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        # cuBLAS requires this environment variable for deterministic matmul
        # os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # info["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

        #torch.backends.cudnn.benchmark = False
        #torch.backends.cudnn.deterministic = True
        #info["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        #info["cudnn_deterministic"] = torch.backends.cudnn.deterministic

    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        info["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        info["cudnn_deterministic"] = torch.backends.cudnn.deterministic

    return info


def init_logging(config: dict, device: str, seed: int, determinism: Optional[Dict[str, object]] = None):
    """Create a new run directory with timestamp, save config, determinism info, and system details."""

    # Check runs directory exists
    (REPO_ROOT / "runs").mkdir(exist_ok=True)

    run_id = time.strftime("%Y-%m-%d_%H-%M-%S") + (f"_{config['meta']['name']}")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.yaml", "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    # Save system info
    sysinfo = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else "NA",
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "seed": seed,
        "git_commit": _try_cmd(["git", "rev-parse", "--short", "HEAD"]),
        "start_time": run_id
    }
    if determinism:
        for key, value in determinism.items():
            sysinfo[f"determinism.{key}"] = value
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


def _try_cmd(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "NA"


def _peek_example(loader) -> Optional[torch.Tensor]:
    """ Return a single example tensor from a DataLoader on CPU """

    iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        return None

    example = batch[0] if isinstance(batch, (tuple, list)) else batch
    if not isinstance(example, torch.Tensor):
        return None

    return example[:1].detach()


def compute_model_complexity(
    model: torch.nn.Module,
    loader=None,
) -> Optional[Dict[str, int]]:
    """ Compute parameter count and MACs for a model """

    example_input = _peek_example(loader)

    if example_input is None:
        return None

    example_input = example_input.to("cpu")

    try:
        model.to("cpu")
        model.eval()
        with torch.no_grad():
            macs, params = profile(model, inputs=(example_input,), verbose=False)
    except Exception:
        return None

    try:
        params = int(params)
        macs = int(macs)
    except (TypeError, ValueError):
        return None

    return {"param_count": params, "macs": macs}
