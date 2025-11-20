"""
Training script for binary image classification on the VWW dataset
Used to train baseline and teacher models

Default config: src/config/baseline_mbv3s_vww96.yaml

Example usage: python -m src.train

"""

import argparse, time, json
from pathlib import Path
import numpy as np

import torch
from sklearn.metrics import confusion_matrix, classification_report

from src.engine.setup import build_context
from src.engine.train_loops import train_one_epoch, evaluate
from src.engine.utils import log_epoch, compute_model_complexity
from src.engine.ema import ModelEMA
from src.engine.finetune_utils import set_backbone_trainable, recalibrate_batch_norm


def run_training(args: argparse.Namespace):
    ctx = build_context(args.config_path)
    if args.debug == True:
        print("Config load complete:")
        print(ctx)

    best_acc = 0.0
    best_epoch = 0
    patience = 0
    overall_start = time.perf_counter()
    ema = None
    if ctx.get("ema_decay"):
        ema = ModelEMA(ctx["model"], ctx["ema_decay"])
        ema.to(ctx["device"])

    freeze_epochs = ctx.get("freeze_backbone_epochs", 0) or 0
    backbone_frozen = False
    if freeze_epochs > 0:
        set_backbone_trainable(ctx["model"], False)
        backbone_frozen = True
        print(f"Freezing backbone for first {freeze_epochs} epochs")

    bn_recal_epoch = ctx.get("bn_recalibrate_epoch")
    bn_recal_batches = ctx.get("bn_recalibrate_max_batches")
    bn_recal_done = False

    for epoch in range(1, ctx["epochs"] + 1):
        epoch_start = time.perf_counter()

        tr_loss, tr_acc = train_one_epoch(
            ctx["model"],
            ctx["tr_loader"],
            ctx["device"],
            ctx["optimizer"],
            ctx["criterion"],
            ctx["scheduler"],
            ctx["grad_clip_norm"],
            ema=ema,
        )

        if bn_recal_epoch and (not bn_recal_done) and epoch == bn_recal_epoch:
            print("Recalibrating BatchNorm running statistics...")
            recalibrate_batch_norm(
                ctx["model"], ctx["tr_loader"], ctx["device"], bn_recal_batches
            )
            if ema:
                ema.update(ctx["model"])
            bn_recal_done = True

        model_for_eval = ema.ema_model if ema else ctx["model"]
        va_loss, va_acc, *_ = evaluate(model_for_eval, ctx["val_loader"], ctx["device"])

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
        )

        print(
            f"[{epoch}/{ctx['epochs']}] "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
            f"epoch time {epoch_elapsed:.1f}s | elapsed time {elapsed_total/60:.1f}m"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            patience = 0

            torch.save(
                model_for_eval.state_dict(),
                ctx["run_dir"] / "model.pt"
            )

        else:
            patience += 1
            if patience >= ctx["max_patience"]:
                print(f"No improvement in {ctx['max_patience']} epochs, stopping early")
                break

        if backbone_frozen and epoch >= freeze_epochs:
            set_backbone_trainable(ctx["model"], True)
            backbone_frozen = False
            print("Backbone unfrozen")

    total_elapsed = time.perf_counter() - overall_start

    # Final metrics -----

    # Reload the best checkpoint before computing final metrics
    final_model = ema.ema_model if ema else ctx["model"]
    try:
        best_ckpt_path = ctx["run_dir"] / "model.pt"
        checkpt = torch.load(best_ckpt_path, map_location="cpu")
        final_model.load_state_dict(checkpt, strict=True)
    except Exception:
        # If loading fails, fall back to current in-memory weights
        pass

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
    
    # ---------------------

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


def main():
    ap = argparse.ArgumentParser()
    default_config = Path("src") / "config" / "baseline_mbv3s_vww96.yaml"

    ap.add_argument(
        "--config_path",
        type=Path,
        default=default_config,
        help="Path to YAML config file"
    )
    ap.add_argument("--debug", required=False, type=bool, default=False)
    args = ap.parse_args()

    run_training(args)

if __name__ == "__main__":
    main()
