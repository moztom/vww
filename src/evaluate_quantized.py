import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.engine.data import build_dataloaders
from src.engine.train_loops import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved quantized (int8) model on VWW.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to quantized .pt file (full module).")
    parser.add_argument("--data_path", type=Path, required=True, help="Dataset root containing val/ subfolders.")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = "cpu"
    model = torch.load(args.model_path, map_location=device)
    model.eval()

    val_loader = build_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        eval_only=True,
    )

    loss, acc, preds, gts = evaluate(model, val_loader, device, metrics=True)
    print(f"val loss: {loss:.4f}")
    print(f"val acc: {acc:.4f}")

    labels = [0, 1]
    target_names = ["no_person(0)", "person(1)"]
    print(confusion_matrix(gts, preds, labels=labels))
    print(classification_report(gts, preds, labels=labels, target_names=target_names))

    size_bytes = args.model_path.stat().st_size
    print(f"model bytes: {size_bytes:,} (~{size_bytes/1024/1024:.2f} MB)")

    metrics_path = args.model_path.parent / "quant_eval_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(
            {
                "model_path": str(args.model_path),
                "val_loss": loss,
                "val_acc": acc,
                "confusion_matrix": confusion_matrix(gts, preds, labels=labels).tolist(),
                "classification_report": classification_report(
                    gts, preds, labels=labels, target_names=target_names, output_dict=True
                ),
                "model_bytes": size_bytes,
            },
            fh,
            indent=2,
        )
    print(f"Saved detailed metrics to {metrics_path}")


if __name__ == "__main__":
    main()
