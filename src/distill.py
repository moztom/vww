"""
Knowledge distillation for training of student models

Default config: src/config/student.yaml

Example usage: python -m src.distill
"""

import argparse, time, json
from pathlib import Path
import numpy as np

import torch
from sklearn.metrics import confusion_matrix, classification_report

from src.engine.setup import build_context
from src.engine.kd import kd_train_one_epoch
from src.engine.train_loops import evaluate
from src.engine.utils import log_epoch, compute_model_complexity


def run_distillation(args: argparse.Namespace):
    ctx = build_context(args.config_path, stage="kd")
    if args.debug == True:
        print("Config load complete:")
        print(ctx)

    best_acc = 0.0
    best_epoch = 0
    patience = 0
    overall_start = time.perf_counter()

    for epoch in range(1, ctx["epochs"] + 1):
        epoch_start = time.perf_counter()
        epoch_alpha = float(ctx["kd_alpha"])
        epoch_margin = float(ctx.get("kd_margin_weight", 0.0))

        tr_loss, tr_acc, tr_ce, tr_kl, tr_margin = kd_train_one_epoch(
            ctx["model"],
            ctx["teacher"],
            ctx["tr_loader"],
            ctx["device"],
            ctx["optimizer"],
            ctx["scheduler"],
            epoch_alpha,
            ctx["kd_temp"],
            ctx["grad_clip_norm"],
            label_smoothing=ctx.get("kd_label_smoothing", 0.0),
            teacher_input_size=ctx.get("kd_teacher_input_size"),
            confidence_gamma=ctx.get("kd_confidence_gamma"),
            margin_weight=epoch_margin,
        )

        va_loss, va_acc, *_ = evaluate(ctx["model"], ctx["val_loader"], ctx["device"])

        epoch_elapsed = time.perf_counter() - epoch_start
        elapsed_total = time.perf_counter() - overall_start

        log_epoch(
            ctx["writer"],
            ctx["run_dir"],
            epoch,
            tr_loss,
            tr_acc,
            va_loss,
            va_acc,
            ctx["optimizer"].param_groups[0]["lr"],
            ce=tr_ce,
            kl=tr_kl,
            alpha=epoch_alpha,
            margin=tr_margin,
            margin_weight=epoch_margin,
        )

        margin_str = f", margin {tr_margin:.4f}" if tr_margin > 0 else ""
        print(
            f"[{epoch}/{ctx['epochs']}] "
            f"train loss {tr_loss:.4f} (ce {tr_ce:.4f}, kl {tr_kl:.4f}{margin_str}) acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
            f"epoch time {epoch_elapsed:.1f}s | elapsed time {elapsed_total/60:.1f}m"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            patience = 0

            torch.save(
                ctx["model"].state_dict(),
                ctx["run_dir"] / "model.pt"
            )

        else:
            patience += 1
            if patience >= ctx["max_patience"]:
                print(f"No improvement in {ctx['max_patience']} epochs, stopping early")
                break

    total_elapsed = time.perf_counter() - overall_start

    # Final metrics ----------
    
    # Reload the best checkpoint before computing final metrics
    final_model = ctx["model"]
    best_ckpt_path = ctx["run_dir"] / "model.pt"
    checkpt = torch.load(best_ckpt_path, map_location="cpu")
    final_model.load_state_dict(checkpt, strict=True)

    va_loss, va_acc, preds, gts = evaluate(
        final_model, ctx["val_loader"], ctx["device"], metrics=True
    )
    print("\nVALIDATION SUMMARY")
    print(
        f"\nBest checkpoint: val acc = {best_acc:.4f} (epoch {best_epoch}) ({ctx['run_dir'] / 'model.pt'})"
    )
    print(f"Total training time: {total_elapsed/60:.1f}mins ({total_elapsed:.1f}s)")

    labels = [0, 1]
    target_names = ["no_person(0)", "person(1)"]

    print("\nConfusion matrix:")
    cm = confusion_matrix(gts, preds, labels=labels)
    print(cm)
    print("Key:")
    key = np.matrix(
        [
            ["True neg (pred=0)", "False pos (pred=1)"],
            ["False neg (pred=0)", "True pos (pred=1)"],
        ]
    )
    print(key)

    print("\nClassification report:")
    print(classification_report(gts, preds, labels=labels, target_names=target_names))

    complexity = compute_model_complexity(final_model, ctx["val_loader"])
    param_count = macs = None
    if complexity:
        param_count = complexity["param_count"]
        macs = complexity["macs"]
        print(
            f"\nModel complexity: params={param_count:,} ({param_count/1e6:.2f}M) | "
            f"MACs={macs:,} ({macs/1e6:.2f}M)"
        )

    # Save final metrics to metrics.jsonl
    with open(ctx["run_dir"] / "metrics.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "best_epoch": best_epoch,
                    "best_val_acc": best_acc,
                    "total_train_time": total_elapsed,
                    "model_param_count": param_count,
                    "model_macs": macs,
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "classification_report": classification_report(
                        gts,
                        preds,
                        labels=labels,
                        target_names=target_names,
                        output_dict=True,
                    )
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "labels": target_names,
                    "confusion matrix": cm.tolist(),
                    "key": key.tolist(),
                }
            )
        )

    # Close TensorBoard writer
    ctx["writer"].flush()
    ctx["writer"].close()

    # ---------------------


def main():
    parser = argparse.ArgumentParser()
    default_config = Path("src") / "config" / "student.yaml"

    parser.add_argument(
        "--config_path",
        type=Path,
        default=default_config,
        help="Path to kd YAML config"
    )
    parser.add_argument(
        "--debug", action="store_true"
    )
    args = parser.parse_args()

    run_distillation(args)

if __name__ == "__main__":
    main()
