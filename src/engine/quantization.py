import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.export import Dim, export
from torchao.quantization.pt2e import move_exported_model_to_eval
from torchao.quantization.pt2e.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
    prepare_qat_pt2e,
)

from torch.ao.quantization import allow_exported_model_train_eval


from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

'''
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
  get_symmetric_quantization_config,
  XNNPACKQuantizer,
)
'''
from src.engine.train_loops import evaluate, train_one_epoch
from src.engine.kd import kd_train_one_epoch


def run_quantization(ctx: Dict) -> Dict:
    """run PTQ or QAT based on the loaded config."""
    quant_cfg = ctx.get("quant_cfg")
    if not quant_cfg:
        raise ValueError("Quantization context missing 'quant_cfg'.")

    mode = str(quant_cfg.get("mode", "ptq")).lower()
    if mode == "ptq":
        summary = _run_ptq(ctx, quant_cfg)
    elif mode == "qat":
        summary = _run_qat(ctx, quant_cfg)
    else:
        raise ValueError(f"Unsupported quantization mode '{mode}'.")

    _write_quant_summary(ctx["run_dir"], summary)
    return summary


def _run_ptq(ctx: Dict, quant_cfg: Dict) -> Dict:
    model = ctx["model"].to("cpu").eval()

    example_input = _example_input(ctx["val_loader"])
    batch_dim = Dim.AUTO
    exported = export(
        model,
        (example_input,),
        dynamic_shapes=({0: batch_dim},),
    ).module(check_guards=False)
    ptq_cfg = quant_cfg.get("ptq", {})
    exclude_first_last = bool(ptq_cfg.get("exclude_first_last", True))

    quantizer = _build_xnnpack_quantizer(model, exclude_first_last, is_qat=False)
    prepared = prepare_pt2e(exported, quantizer)
    # Allow calling train()/eval() on exported models inside our loops
    allow_exported_model_train_eval(prepared)

    calibrate_loader = _pick_loader(ctx, quant_cfg.get("calibrate_split", "val"))
    calibrate_batches = int(quant_cfg.get("calibrate_batches", 200))
    observed_batches = _calibrate_model(prepared, calibrate_loader, calibrate_batches)

    quantized = convert_pt2e(prepared)
    allow_exported_model_train_eval(quantized)
    metrics = _evaluate_on_cpu(quantized, ctx["val_loader"])

    state_path, full_path = _save_quantized_model(quantized, ctx["run_dir"], suffix="ptq")
    state_bytes = state_path.stat().st_size
    full_bytes = full_path.stat().st_size if full_path and full_path.exists() else None
    summary = {
        "mode": "ptq",
        "backend": "xnnpack",
        "exclude_first_last": exclude_first_last,
        "calibrate_split": quant_cfg.get("calibrate_split", "val"),
        "calibrate_batches": calibrate_batches,
        "calibrate_batches_observed": observed_batches,
        "val_loss": metrics["loss"],
        "val_acc": metrics["acc"],
        "state_dict_path": str(state_path),
        "full_model_path": str(full_path) if full_path else None,
        "state_dict_bytes": state_bytes,
        "full_model_bytes": full_bytes,
    }
    return summary


def _run_qat(ctx: Dict, quant_cfg: Dict) -> Dict:
    device = ctx["device"]

    qat_cfg = quant_cfg.get("qat", {})
    exclude_first_last = bool(qat_cfg.get("exclude_first_last", False))

    model = ctx["model"].to(device=device, dtype=torch.float32)
    model.train()
    example_input = _example_input(ctx["val_loader"]).to(device)
    batch_dim = Dim.AUTO
    exported = export(
        model,
        (example_input,),
        dynamic_shapes=({0: batch_dim},),
    ).module(check_guards=False)
    quantizer = _build_xnnpack_quantizer(model, exclude_first_last, is_qat=True)
    prepared = prepare_qat_pt2e(exported, quantizer)
    prepared.to(device)
    allow_exported_model_train_eval(prepared)

    epochs = int(qat_cfg.get("epochs", 5))
    lr = float(qat_cfg.get("lr", 1e-4))
    weight_decay = float(qat_cfg.get("weight_decay", 1e-5))
    patience = int(qat_cfg.get("patience", 0))
    grad_clip = float(qat_cfg.get("grad_clip_norm", ctx.get("grad_clip_norm", 0.0)))
    optimizer = AdamW(prepared.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None

    kd_block = quant_cfg.get("kd", {})
    use_kd = bool(kd_block.get("use_kd", True))
    teacher = ctx.get("teacher")
    if use_kd and teacher is None:
        raise ValueError("QAT requested KD but teacher is missing from context.")
    use_kd = use_kd and teacher is not None

    kd_alpha = float(kd_block.get("alpha", ctx.get("kd_alpha", 0.5)))
    kd_temp = float(kd_block.get("temperature", ctx.get("kd_temp", 4.0)))
    kd_margin_weight = float(kd_block.get("margin_weight", ctx.get("kd_margin_weight", 0.0)))
    kd_label_smoothing = float(kd_block.get("label_smoothing", ctx.get("kd_label_smoothing", 0.0)))

    best_state = None
    best_acc = -1.0
    best_epoch = 0
    patience_ctr = 0

    writer = ctx.get("writer")

    for epoch in range(1, epochs + 1):
        if use_kd:
            tr_loss, tr_acc, tr_ce, tr_kl, tr_margin = kd_train_one_epoch(
                prepared,
                teacher,
                ctx["tr_loader"],
                device,
                optimizer,
                scheduler,
                kd_alpha,
                kd_temp,
                grad_clip,
                label_smoothing=kd_label_smoothing,
                teacher_input_size=ctx.get("kd_teacher_input_size"),
                confidence_gamma=ctx.get("kd_confidence_gamma"),
                margin_weight=kd_margin_weight,
            )
        else:
            tr_loss, tr_acc = train_one_epoch(
                prepared,
                ctx["tr_loader"],
                device,
                optimizer,
                ctx["criterion"],
                scheduler,
                grad_clip,
            )
            tr_ce = tr_kl = tr_margin = None

        va_loss, va_acc, *_ = evaluate(prepared, ctx["val_loader"], device)

        if writer:
            writer.add_scalar("train loss", float(tr_loss), epoch)
            writer.add_scalar("train acc", float(tr_acc), epoch)
            writer.add_scalar("val loss", float(va_loss), epoch)
            writer.add_scalar("val acc", float(va_acc), epoch)
            writer.add_scalar("lr", float(optimizer.param_groups[0]["lr"]), epoch)
            if tr_ce is not None:
                writer.add_scalar("ce_loss", float(tr_ce), epoch)
            if tr_kl is not None:
                writer.add_scalar("kl_loss", float(tr_kl), epoch)
            if tr_margin is not None:
                writer.add_scalar("margin_loss", float(tr_margin), epoch)

        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            patience_ctr = 0
            best_state = {k: v.cpu() for k, v in prepared.state_dict().items()}
        else:
            patience_ctr += 1
            if patience > 0 and patience_ctr >= patience:
                break

    if best_state is not None:
        prepared.load_state_dict(best_state)

    prepared.to("cpu")
    recal_loader = _pick_loader(ctx, quant_cfg.get("calibrate_split", "val"))
    calibrate_batches = int(quant_cfg.get("calibrate_batches", 200))
    observed_batches = _calibrate_model(prepared, recal_loader, calibrate_batches)

    quantized = convert_pt2e(prepared)
    allow_exported_model_train_eval(quantized)
    metrics = _evaluate_on_cpu(quantized, ctx["val_loader"])

    state_path, full_path = _save_quantized_model(quantized, ctx["run_dir"], suffix="qat")
    state_bytes = state_path.stat().st_size
    full_bytes = full_path.stat().st_size if full_path and full_path.exists() else None
    summary = {
        "mode": "qat",
        "backend": "xnnpack",
        "exclude_first_last": exclude_first_last,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "use_kd": use_kd,
        "calibrate_split": quant_cfg.get("calibrate_split", "val"),
        "calibrate_batches": calibrate_batches,
        "calibrate_batches_observed": observed_batches,
        "val_loss": metrics["loss"],
        "val_acc": metrics["acc"],
        "state_dict_path": str(state_path),
        "full_model_path": str(full_path) if full_path else None,
        "state_dict_bytes": state_bytes,
        "full_model_bytes": full_bytes,
    }
    return summary


def _example_input(loader) -> torch.Tensor:
    iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("Dataloader is empty; cannot obtain example input for FX prepare.") from exc

    example = batch[0] if isinstance(batch, (tuple, list)) else batch
    if not isinstance(example, torch.Tensor):
        raise TypeError("Expected tensor batch from loader for example input.")
    return example.to("cpu")


def _first_last_module_names(model: nn.Module) -> Tuple[Optional[str], Optional[str]]:
    first_conv = None
    last_linear = None
    for name, module in model.named_modules():
        if first_conv is None and isinstance(module, nn.Conv2d):
            first_conv = name
        if isinstance(module, nn.Linear):
            last_linear = name
    return first_conv, last_linear


def _build_xnnpack_quantizer(model: nn.Module, exclude_first_last: bool, *, is_qat: bool) -> XNNPACKQuantizer:
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_qat=is_qat))
    # XNNPACKQuantizer does not support setting layers to None :(

    return quantizer


def _pick_loader(ctx: Dict, split: str):
    split = str(split or "val").lower()
    if split == "train":
        return ctx["tr_loader"]
    return ctx["val_loader"]


def _calibrate_model(model: torch.nn.Module, loader, max_batches: int) -> int:
    move_exported_model_to_eval(model)
    count = 0
    max_batches = max(0, int(max_batches))
    with torch.inference_mode():
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            model(images.to("cpu"))
            count += 1
            if max_batches and count >= max_batches:
                break
    return count


def _evaluate_on_cpu(model: torch.nn.Module, loader) -> Dict[str, float]:
    move_exported_model_to_eval(model)
    loss, acc, *_ = evaluate(model, loader, device="cpu")
    return {"loss": float(loss), "acc": float(acc)}


def _save_quantized_model(model: torch.nn.Module, run_dir: Path, suffix: str) -> Tuple[Path, Optional[Path]]:
    run_dir = Path(run_dir)
    state_path = run_dir / f"model_int8_{suffix}_state.pt"
    torch.save({"model_state": model.state_dict()}, state_path)
    full_path = run_dir / f"model_int8_{suffix}_full.pt"
    try:
        torch.save(model, full_path)
    except Exception as exc:
        warnings.warn(f"Skipping full-model save for {suffix}: {exc}")
        full_path = None
    return state_path, full_path


def _write_quant_summary(run_dir: Path, payload: Dict) -> None:
    run_dir = Path(run_dir)
    summary_path = run_dir / "quant_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(payload, fh, indent=2)
