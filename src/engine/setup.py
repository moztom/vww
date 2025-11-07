from pathlib import Path
import yaml

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.engine.utils import set_seed, init_logging
from src.engine.data import build_dataloaders
from src.engine.models import build_model


def build_context(config_path: Path, stage: str = None):
    """Build training context from config file"""

    # load config
    cfg = _load_config(config_path)

    # seed and determinism
    determinism = set_seed(cfg["meta"]["seed"])

    # device
    device = _pick_device(cfg["meta"].get("device", "auto"))

    # data
    tr_loader, val_loader, class_weight_tensor = build_dataloaders(
        data_path=Path(cfg["data"]["path"]),
        batch_size=cfg["data"]["batch"],
        num_workers=cfg["data"]["num_workers"],
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
        rhf=cfg["data"]["aug"]["rand_hflip"],
        cj=cfg["data"]["aug"]["color_jitter"],
        re=cfg["data"]["aug"]["random_erasing"],
    )

    # model
    model = build_model(cfg["model"]["type"], cfg["model"]["pretrained"]).to(device)

    # load initial checkpoint (if specified)
    init_checkpoint = cfg["train"].get("init_checkpoint")
    if init_checkpoint:
        checkpt = torch.load(Path(init_checkpoint), map_location="cpu")
        checkpt = checkpt.get("model") # TEMPORARY remove when i retrain my models (to not contain full checkpoint)
        model.load_state_dict(checkpt)

    # criterion/loss
    criterion = CrossEntropyLoss(
        weight=class_weight_tensor.to(device) if class_weight_tensor else None,
        label_smoothing=cfg["train"]["label_smoothing"],
    )

    # what is class_weight_tensor for?
    # It's for handling class imbalance in the dataset by assigning different weights to each class during loss

    # optimizer
    optimizer = _make_optimizer(cfg, model)

    # scheduler
    scheduler = _make_scheduler(cfg, optimizer, tr_loader)

    # logging / run dir
    run_dir, writer = init_logging(cfg, device, cfg["meta"]["seed"], determinism)

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
        "max_patience": cfg["train"]["early_stop_patience"],
        "epochs": cfg["train"]["epochs"],
        "grad_clip_norm": cfg["train"].get("grad_clip_norm", 0.0),
        "freeze_backbone_epochs": cfg["train"].get("freeze_backbone_epochs", 0),
        "ema_decay": cfg["train"].get("ema_decay"),
        "bn_recalibrate_epoch": cfg["train"].get("bn_recalibrate_epoch"),
        "bn_recalibrate_max_batches": cfg["train"].get("bn_recalibrate_max_batches"),
        "config": cfg,
    }

    if stage in ("kd", "prune"):
        kd_cfg = cfg["kd"]
        teacher_cfg = kd_cfg["teacher"]

        teacher = build_model(teacher_cfg["arch"], teacher_cfg["pretrained"])
        checkpt = torch.load(teacher_cfg["checkpt"], map_location="cpu")
        checkpt = checkpt.get("model") # TEMPORARY remove when i retrain my models (to not contain full checkpoint)
        teacher.load_state_dict(checkpt)
        teacher.to(device)
        teacher.eval()

        for p in teacher.parameters():
            p.requires_grad_(False)

        context.update({
            "teacher": teacher,
            "kd_alpha": float(cfg.get("alpha", 0.5)),
            "kd_temp": float(cfg.get("temperature", 4.0)),
            # Optional scheduling controls
            "kd_alpha_start": kd_cfg.get("alpha_start", None),
            "kd_alpha_end": kd_cfg.get("alpha_end", None),
            "kd_alpha_warmup_epochs": kd_cfg.get("alpha_warmup_epochs", None),
            "kd_alpha_decay_end_epoch": kd_cfg.get("alpha_decay_end_epoch", None),
            "kd_alpha_constant": kd_cfg.get("alpha_constant", None),
            "kd_teacher_input_size": kd_cfg.get("teacher_input_size"),
            "kd_confidence_gamma": kd_cfg.get("confidence_gamma"),
            "kd_margin_weight": float(kd_cfg.get("margin_weight", 0.0)),
            "kd_margin_weight_start": kd_cfg.get("margin_weight_start"),
            "kd_margin_weight_end": kd_cfg.get("margin_weight_end"),
            "kd_margin_decay_end_epoch": kd_cfg.get("margin_weight_decay_end_epoch"),
            "kd_label_smoothing": kd_cfg.get("label_smoothing", 0.0),
        })

    if stage == "prune":
        context["prune_cfg"] = cfg["prune"]

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


def _make_optimizer(cfg, model):
    """Create optimizer based on config"""
    if cfg["train"]["optimizer"]["name"] == "adamw":
        optimizer = AdamW(
            params=model.parameters(),
            lr=cfg["train"]["optimizer"]["lr"],
            weight_decay=cfg["train"]["optimizer"]["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer {cfg['train']['optimizer']}")

    return optimizer


def _make_scheduler(cfg, optimizer, tr_loader):
    """Create scheduler based on config"""
    if cfg["train"]["scheduler"]["name"] == "onecycle":
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=cfg["train"]["scheduler"]["max_lr"],
            epochs=cfg["train"]["epochs"],
            steps_per_epoch=len(tr_loader),
            pct_start=cfg["train"]["scheduler"]["pct_start"],
            div_factor=cfg["train"]["scheduler"]["div_factor"],
            final_div_factor=cfg["train"]["scheduler"]["final_div_factor"],
            anneal_strategy="cos",
        )
    else:
        raise ValueError(f"Unknown scheduler {cfg['train']['scheduler']}")

    return scheduler
