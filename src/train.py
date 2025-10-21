import argparse, time, json
from pathlib import Path
import numpy as np

import torch
from sklearn.metrics import confusion_matrix, classification_report

from src.engine.setup import build_context
from src.engine.train_loops import train_one_epoch, evaluate
from src.engine.utils import log_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config_path", required=True, type=Path, help="Path to config file (.yaml)"
    )
    ap.add_argument("--debug", required=False, type=bool, default=False)
    args = ap.parse_args()

    ctx = build_context(args.config_path)

    if args.debug == True:
        print("Config load complete:")
        print(ctx)

    best_acc = 0.0
    best_epoch = 0
    output_path = ctx["run_dir"] / "model.pt"
    patience = 0
    overall_start = time.perf_counter()

    for epoch in range(1, ctx["epochs"] + 1):
        epoch_start = time.perf_counter()

        tr_loss, tr_acc = train_one_epoch(
            ctx["model"],
            ctx["tr_loader"],
            ctx["device"],
            ctx["optimizer"],
            ctx["criterion"],
            ctx["scheduler"],
            ctx["scaler"],
            ctx["autocast"],
            ctx["grad_clip_norm"]
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
        )

        print(
            f"[{epoch}/{ctx["epochs"]}] "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
            f"epoch time {epoch_elapsed:.1f}s | elapsed time {elapsed_total/60:.1f}m"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            patience = 0
            torch.save(
                {"model": ctx["model"].state_dict(), "acc": va_acc, "epoch": epoch},
                output_path,
            )
        else:
            patience += 1
            if patience >= ctx["max_patience"]:
                print(f"No improvement in {ctx["max_patience"]} epochs, stopping early")
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
    key=np.matrix([["True neg (pred=0)", "False pos (pred=1)"], ["False neg (pred=0)", "True pos (pred=1)"]])
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
                    "key": key.tolist()
                }
            )
        )

    # Close TensorBoard writer
    ctx["writer"].flush()
    ctx["writer"].close()


if __name__ == "__main__":
    main()
