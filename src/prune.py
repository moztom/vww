"""
Iterative structured pruning for MobileNetV3-S

Default config: src/config/student_mbv3s_vww96_prune.yaml

Example usage: python -m src.prune
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW

from src.engine.setup import build_context
from src.engine.utils import log_epoch, compute_model_complexity
from src.engine.train_loops import evaluate, train_one_epoch
from src.engine.kd import kd_train_one_epoch
from src.engine.pruning import MobilenetV3ChannelPruner, ChannelScore
from src.engine.finetune_utils import recalibrate_batch_norm


def run_pruning(args: argparse.Namespace) -> None:
    ctx = build_context(args.config_path, stage="prune")
    prune_cfg = ctx.get("prune_cfg")
    if not prune_cfg:
        raise ValueError("Missing prune configuration. Add a prune section to the YAML config.")

    device = ctx["device"]
    model = ctx["model"]
    teacher = ctx.get("teacher")
    run_dir: Path = ctx["run_dir"]
    writer = ctx["writer"]

    finetune_cfg = prune_cfg.get("finetune", {})
    acceptance_cfg = prune_cfg.get("acceptance", {})

    targets = sorted(float(t) for t in prune_cfg.get("targets"))
    importance = prune_cfg.get("importance", "bn_gamma")
    protect_cfg = prune_cfg.get("protect", {})
    expand_only = bool(prune_cfg.get("expand_only", True))

    use_kd = bool(finetune_cfg.get("use_kd", True)) and teacher is not None
    if finetune_cfg.get("use_kd", True) and teacher is None:
        print("Warning: KD recovery requested but teacher not available. Falling back to CE fine-tune.")
        use_kd = False

    grad_clip = prune_cfg.get("grad_clip_norm", ctx.get("grad_clip_norm", 0.0))
    weight_decay = float(finetune_cfg.get("weight_decay", 1e-5))
    max_patience = int(finetune_cfg.get("patience", 2))
    max_epochs = int(finetune_cfg.get("epochs", 4))
    bn_recalibrate_batches = finetune_cfg.get("bn_recalibrate_batches", 32)
    bn_recalibrate_loader = str(finetune_cfg.get("bn_recalibrate_loader", "val")).strip().lower()
    if bn_recalibrate_loader not in ("train", "val"):
        bn_recalibrate_loader = "val"

    alpha = float(ctx.get("kd_alpha", 0.5))
    temperature = float(ctx.get("kd_temp", 4.0))
    label_smoothing = float(ctx.get("kd_label_smoothing", 0.0))
    teacher_input_size = ctx.get("kd_teacher_input_size")
    confidence_gamma = ctx.get("kd_confidence_gamma")
    margin_weight = float(ctx.get("kd_margin_weight", 0.0))

    pruner = MobilenetV3ChannelPruner(
        model=model,
        protect_cfg=protect_cfg,
        importance=importance,
        expand_only=expand_only,
    )
    total_prunable = pruner.total_initial_prunable
    print(f"Total prunable channels (expand convs): {total_prunable}")

    model.to(device)

    # Baseline evaluation before any pruning
    baseline_start = time.perf_counter()
    base_loss, base_acc, *_ = evaluate(model, ctx["val_loader"], device)
    baseline_time = time.perf_counter() - baseline_start
    print(f"Baseline accuracy: {base_acc:.4f} (eval {baseline_time:.1f}s)")
    base_checkpoint_full = run_dir / "model_pruned_base_full.pt"
    torch.save({"model": model}, base_checkpoint_full)

    complexity = compute_model_complexity(model, ctx["val_loader"])
    if complexity:
        params = complexity["param_count"]
        macs = complexity["macs"]
        print(
            f"Baseline complexity: params={params:,} ({params/1e6:.2f}M) | "
            f"MACs={macs:,} ({macs/1e6:.2f}M)"
        )
        model_size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        model_size_bytes += sum(b.nelement() * b.element_size() for b in model.buffers())
        model_size_mb = model_size_bytes / (1024 ** 2)
        with open(run_dir / "metrics.jsonl", "a") as file:
            file.write(json.dumps({
                "Baseline params": f"{params:,} ({params/1e6:.2f}M)",
                "Basleine MACs": f"{macs:,} ({macs/1e6:.2f}M)",
                "Baseline model size": f"{model_size_bytes:,} bytes ({model_size_mb:.2f} MB)",
            }) + "\n")

    prev_best_acc = base_acc
    global_epoch = 0

    for target in targets:
        print("\n" + "-" * 70)
        print(f"Pruning towards {target:.2%} global channel sparsity (expand convs only)")

        plan: List[ChannelScore] = pruner.plan_for_fraction(target, importance=importance)
        to_remove = len(plan)
        if to_remove == 0:
            print("No additional channels to prune for this target; skipping.")
            continue

        print(f"Removing {to_remove} channels (cumulative fraction before removal: {_format_fraction(pruner.current_fraction())})")
        plan_per_block = pruner.apply(plan)
        cumulative = pruner.current_fraction()
        print(f"Cumulative removed fraction: {_format_fraction(cumulative)}")

        if bn_recalibrate_batches:
            recal_loader = ctx["val_loader"] if bn_recalibrate_loader == "val" else ctx["tr_loader"]
            print(f"Recalibrating BatchNorm statistics using the {bn_recalibrate_loader} loader...")
            recalibrate_batch_norm(model, recal_loader, device, bn_recalibrate_batches)

        pre_va_loss, pre_va_acc, *_ = evaluate(model, ctx["val_loader"], device)
        print(f"Post-surgery pre-recovery accuracy: {pre_va_acc:.4f}")

        finetune_lr = _lr_for_target(finetune_cfg, target)
        optimizer = AdamW(model.parameters(), lr=finetune_lr, weight_decay=weight_decay)
        scheduler = None
        patience = 0
        best_step_acc = -1.0
        best_step_loss = float("inf")
        best_epoch = 0
        step_checkpoint_full = run_dir / f"model_pruned_{int(round(target * 100))}_full.pt"

        for epoch in range(1, max_epochs + 1):
            global_epoch += 1
            epoch_start = time.perf_counter()

            if use_kd:
                tr_loss, tr_acc, tr_ce, tr_kl, tr_margin = kd_train_one_epoch(
                    model,
                    teacher,
                    ctx["tr_loader"],
                    device,
                    optimizer,
                    scheduler,
                    alpha,
                    temperature,
                    grad_clip,
                    label_smoothing=label_smoothing,
                    teacher_input_size=teacher_input_size,
                    confidence_gamma=confidence_gamma,
                    margin_weight=margin_weight,
                )
            else:
                tr_loss, tr_acc = train_one_epoch(
                    model,
                    ctx["tr_loader"],
                    device,
                    optimizer,
                    ctx["criterion"],
                    scheduler,
                    grad_clip,
                )

            va_loss, va_acc, *_ = evaluate(model, ctx["val_loader"], device)
            elapsed = time.perf_counter() - epoch_start

            log_epoch(
                writer,
                run_dir,
                global_epoch,
                tr_loss,
                tr_acc,
                va_loss,
                va_acc,
                optimizer.param_groups[0]["lr"],
                ce=tr_ce if tr_ce else None,
                kl=tr_kl if tr_kl else None,
                alpha=alpha if use_kd else None,
                margin=tr_margin if tr_margin else None,
                margin_weight=margin_weight if use_kd else None,
            )

            print(
                f"[target {_format_fraction(target)} | epoch {epoch}/{max_epochs}] "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
                f"time {elapsed:.1f}s"
            )

            if va_acc > best_step_acc:
                best_step_acc = va_acc
                best_step_loss = va_loss
                best_epoch = epoch
                torch.save({"model": model}, step_checkpoint_full)
                patience = 0
            else:
                patience += 1
                if max_patience and patience >= max_patience:
                    print(f"Early stopping recovery after {epoch} epochs (patience reached).")
                    break

        if step_checkpoint_full.exists():
            obj = torch.load(step_checkpoint_full, map_location="cpu", weights_only=False)
            saved_model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
            model.load_state_dict(saved_model.state_dict())
            model.to(device)

        final_va_loss, final_va_acc, *_ = evaluate(model, ctx["val_loader"], device)
        print(
            f"Best recovery epoch {best_epoch}: val acc {best_step_acc:.4f} | "
            f"final eval {final_va_acc:.4f}"
        )

        complexity = compute_model_complexity(model, ctx["val_loader"])
        params = complexity["param_count"]
        macs = complexity["macs"]
        model_size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        model_size_bytes += sum(b.nelement() * b.element_size() for b in model.buffers())
        model_size_mb = model_size_bytes / (1024 ** 2)
        print(
            "Step complexity: "
            f"params={params:,} ({params/1e6:.2f}M), "
            f"MACs={macs:,} ({macs/1e6:.2f}M), "
            f"model size={model_size_bytes:,} bytes ({model_size_mb:.2f} MB)"
        )
        with open(run_dir / "metrics.jsonl", "a") as file:
            file.write(json.dumps({
                "Target completed": target,
                "Step params": f"{params:,} ({params/1e6:.2f}M)",
                "Step MACs": f"{macs:,} ({macs/1e6:.2f}M)",
                "Model size": f"{model_size_bytes:,} bytes ({model_size_mb:.2f} MB)",
            }) + "\n")

        acceptance_msg = None
        baseline_threshold = acceptance_cfg.get("baseline_accuracy")
        max_drop = acceptance_cfg.get("max_drop_after_step")
        if baseline_threshold is not None and final_va_acc < float(baseline_threshold):
            acceptance_msg = (
                f"Stopping: accuracy {final_va_acc:.4f} below baseline threshold {baseline_threshold:.4f}."
            )
        elif max_drop is not None and (prev_best_acc - final_va_acc) > float(max_drop):
            acceptance_msg = (
                f"Stopping: accuracy drop {prev_best_acc - final_va_acc:.4f} "
                f"exceeds permitted {max_drop:.4f}."
            )

        model.to(device)

        summary = {
            "target_fraction": target,
            "removed_channels": len(plan),
            "cumulative_fraction": pruner.current_fraction(),
            "pre_recovery_accuracy": pre_va_acc,
            "post_recovery_accuracy": final_va_acc,
            "best_recovery_accuracy": best_step_acc,
            "best_recovery_epoch": best_epoch,
            "learning_rate": finetune_lr,
            "complexity": complexity,
            "model size": {"bytes": model_size_bytes, "MB": model_size_mb},
            "per_block": _serialize_plan(plan_per_block),
            "acceptance_message": acceptance_msg,
        }
        summary_path = run_dir / f"prune_step_{int(round(target * 100))}.json"
        _save_json(summary_path, summary)

        prev_best_acc = final_va_acc

        if acceptance_msg:
            print(acceptance_msg)
            break

    if writer:
        writer.flush()
        writer.close()


def _lr_for_target(cfg: Dict, target: float) -> float:
    lr = float(cfg.get("lr", 1e-4))
    high = cfg.get("lr_high")
    if high is not None:
        threshold = float(cfg.get("lr_high_threshold", 0.3))
        if target >= threshold:
            lr = float(high)
    return lr


def _save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def _serialize_plan(plan_per_block: Dict[int, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    serialized = {}
    for block_idx, info in plan_per_block.items():
        serialized[str(block_idx)] = {
            "removed_channels": info.get("removed_channels", []),
            "remaining": info.get("remaining"),
        }
    return serialized


def _format_fraction(frac: float) -> str:
    return f"{frac * 100:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    default_config = Path("src") / "config" / "student_mbv3s_vww96_prune.yaml"

    parser.add_argument(
        "--config_path",
        type=Path,
        default=default_config,
        help="Path to pruning YAML config."
        )
    args = parser.parse_args()

    run_pruning(args)

if __name__ == "__main__":
    main()
