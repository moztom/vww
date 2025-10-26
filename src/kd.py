import argparse, time, json
from pathlib import Path
import numpy as np

import torch
from sklearn.metrics import confusion_matrix, classification_report

from src.engine.setup import build_context
from src.engine.kd import kd_train_one_epoch
from src.engine.train_loops import evaluate
from src.engine.utils import log_epoch, save_checkpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config_path", required=True, type=Path, help="Path to config file e.g. src\config\student_mbv3s_vww96.yaml"
    )
    ap.add_argument("--debug", required=False, type=bool, default=False)
    args = ap.parse_args()

    ctx = build_context(args.config_path, stage="kd")

    if args.debug == True:
        print("Config load complete:")
        print(ctx)

    best_acc = 0.0
    best_epoch = 0
    output_path = ctx["run_dir"] / "model.pt"
    patience = 0
    overall_start = time.perf_counter()

    def _compute_alpha(epoch: int) -> float:
        """Epoch-wise KD alpha schedule: CE-heavy warmup then linear to target.

        Optional context keys (if present in config):
          - kd_alpha_start
          - kd_alpha_end
          - kd_alpha_warmup_epochs
          - kd_alpha_decay_end_epoch

        Defaults:
          start=0.9, end=ctx["kd_alpha"], warmup=5, decay_end=ctx["epochs"].
        """

        # Don't change alpha if constant flag set
        if ctx["kd_alpha_constant"]:
            return float(ctx["kd_alpha"])
        
        total_epochs = ctx["epochs"]
        start = ctx.get("kd_alpha_start", None)
        end = ctx.get("kd_alpha_end", None)
        if start is None:
            start = 0.9
        if end is None:
            end = ctx["kd_alpha"]
        warmup = ctx.get("kd_alpha_warmup_epochs", None)
        if warmup is None:
            warmup = 5
        decay_end = ctx.get("kd_alpha_decay_end_epoch", None)
        if decay_end is None:
            decay_end = total_epochs

        # Clamp and cast
        warmup = max(0, int(warmup))
        decay_end = max(warmup + 1, int(decay_end))
        epoch = int(epoch)

        if warmup > 0 and epoch <= warmup:
            return float(start)

        if epoch >= decay_end:
            return float(end)

        # Linear decay from start -> end between warmup+1 and decay_end
        span = max(1, decay_end - warmup)
        t = (epoch - warmup) / span
        return float(start + t * (end - start))

    for epoch in range(1, ctx["epochs"] + 1):
        epoch_start = time.perf_counter()
        epoch_alpha = _compute_alpha(epoch)

        tr_loss, tr_acc, tr_ce, tr_kl = kd_train_one_epoch(
            ctx["model"],
            ctx["teacher"],
            ctx["tr_loader"],
            ctx["device"],
            ctx["optimizer"],
            ctx["scheduler"],
            ctx["scaler"],
            ctx["autocast"],
            epoch_alpha,
            ctx["kd_temp"],
            ctx["grad_clip_norm"],
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
        )

        print(
            f"[{epoch}/{ctx['epochs']}] "
            f"alpha {epoch_alpha:.3f} | "
            f"train loss {tr_loss:.4f} (ce {tr_ce:.4f}, kl {tr_kl:.4f}) acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
            f"epoch time {epoch_elapsed:.1f}s | elapsed time {elapsed_total/60:.1f}m"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            patience = 0

            save_checkpt(
                output_path,
                epoch,
                ctx["model"],
                va_acc,
                ctx["save_full_checkpt"],
                optimizer=ctx["optimizer"],
                scheduler=ctx["scheduler"],
                scaler=ctx["scaler"],
                va_loss=va_loss,
            )

        else:
            patience += 1
            if patience >= ctx["max_patience"]:
                print(f"No improvement in {ctx['max_patience']} epochs, stopping early")
                break

    total_elapsed = time.perf_counter() - overall_start

    # Final metrics -----
    va_loss, va_acc, preds, gts = evaluate(
        ctx["model"], ctx["val_loader"], ctx["device"], metrics=True
    )
    print("\nVALIDATION SUMMARY")
    print(
        f"\nBest checkpoint: val acc = {best_acc:.4f} (epoch {best_epoch}) ({output_path})"
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

    # Save final metrics to metrics.jsonl
    with open(ctx["run_dir"] / "metrics.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "best_epoch": best_epoch,
                    "best_val_acc": best_acc,
                    "total_train_time": total_elapsed,
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


if __name__ == "__main__":
    main()
