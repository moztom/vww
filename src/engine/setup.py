from pathlib import Path
import yaml
from contextlib import nullcontext

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler

from src.engine.utils import set_seed, init_logging
from src.data import build_dataloaders
from src.models import build_model


def build_context(config_path: Path, stage: str = None):
    """Build training context from config file"""

    # load config
    config = _load_config(config_path)

    # seed
    set_seed(config["meta"].get("seed", 42))

    # device
    device = _pick_device(config["meta"].get("device", "auto"))

    # data
    tr_loader, val_loader, class_weight_tensor = build_dataloaders(
        data_path=Path(config["data"]["path"]),
        batch_size=config["data"].get("batch_size", 256),
        num_workers=config["data"].get("num_workers", 4),
        mean=config["data"]["mean"],
        std=config["data"]["std"],
        rhf=config["data"]["aug"]["rand_hflip"],
        cj=config["data"]["aug"]["color_jitter"],
        re=config["data"]["aug"]["random_erasing"],
    )

    # model
    model = build_model(config["model"]["type"], config["model"]["pretrained"]).to(device)

    # criterion/loss
    criterion = CrossEntropyLoss(
        weight=class_weight_tensor.to(device) if class_weight_tensor else None,
        label_smoothing=config["train"]["label_smoothing"],
    )

    # what is class_weight_tensor for?
    # It's for handling class imbalance in the dataset by assigning different weights to each class during loss

    # optimizer
    optimizer = _make_optimizer(config, model)

    # scheduler
    scheduler = _make_scheduler(config, optimizer, tr_loader)

    # logging / run dir
    run_dir, writer = init_logging(config, device)

    # AMP (disabled for mps due to gradient underflow with float16 autocast)
    use_amp = config["meta"]["amp"] and (device != "mps")
    scaler = GradScaler() if use_amp and (device == "cuda") else None
    autocast = torch.autocast(device_type=device) if use_amp else nullcontext()
    # NEEDS TO BE DIFFERENT FOR INFERENCE

    context = {
        "device": device,
        "tr_loader": tr_loader,
        "val_loader": val_loader,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "run_dir": run_dir,
        "writer": writer,
        "scaler": scaler,
        "autocast": autocast,
        "max_patience": config["train"]["early_stop_patience"],
        "epochs": config["train"]["epochs"],
        "grad_clip_norm": config["train"].get("grad_clip_norm", 0.0),
        "save_full_checkpt": config["train"]["save_full_checkpt"],
    }

    if stage == "kd":
        teacher = build_model(config["kd"]["teacher"]["arch"], config["kd"]["teacher"]["pretrained"])
        checkpt = torch.load(config["kd"]["teacher"]["checkpt"], map_location="cpu")
        teacher.load_state_dict(checkpt["model"])
        teacher.to(device)
        teacher.eval()

        for p in teacher.parameters():
            p.requires_grad = False
        
        context.update({
            "teacher": teacher,
            "kd_alpha": float(config["kd"]["alpha"]),
            "kd_temp": float(config["kd"]["temperature"]),
        })

    if stage == "prune":
        pass

    if stage == "quant":
        pass


    return context


def _load_config(path: Path):
    """Load and validate YAML config file"""
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    for key in ["meta", "data", "model", "train"]:
        assert key in config, f"Missing top-level '{key}' in {path}"

    return config


def _pick_device(name: str):
    """Pick device based on config string"""
    if name == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    return name


def _make_optimizer(config, model):
    """Create optimizer based on config"""
    if config["train"]["optimizer"]["name"] == "adamw":
        optimizer = AdamW(
            params=model.parameters(),
            lr=config["train"]["optimizer"]["lr"],
            weight_decay=config["train"]["optimizer"]["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer {config['train']['optimizer']}")

    return optimizer


def _make_scheduler(config, optimizer, tr_loader):
    """Create scheduler based on config"""
    if config["train"]["scheduler"]["name"] == "onecycle":
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=config["train"]["scheduler"]["max_lr"],
            epochs=config["train"]["epochs"],
            steps_per_epoch=len(tr_loader),
            pct_start=config["train"]["scheduler"]["pct_start"],
            div_factor=config["train"]["scheduler"]["div_factor"],
            final_div_factor=config["train"]["scheduler"]["div_factor"],
            anneal_strategy="cos",
        )
    else:
        raise ValueError(f"Unknown scheduler {config['train']['scheduler']}")

    return scheduler
