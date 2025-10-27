import argparse
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, classification_report

from src.engine.train_loops import evaluate
from src.data import build_dataloaders
from src.models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["mobilenet_v3_small", "mobilenet_v3_large"], required=True, type=str)
    ap.add_argument("--model_path", required=True, type=Path, help="Path to model (.pt)")
    ap.add_argument("--data_path", required=True, type=Path, help="Path to data (e.g. data/vww96)")
    ap.add_argument("--batch_size", required=True, type=int, help="Batch size (128 or 256)")
    ap.add_argument("--teacher", required=False, type=bool, help="Evaluating teacher model?")
    args = ap.parse_args()

    model = build_model(args.model_type)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    checkpt = torch.load(args.model_path, map_location="cpu")
    state = checkpt["model"]

    print(checkpt["va_acc"])
    print(checkpt["epoch"])

    model.load_state_dict(state, strict=True)
    model.to(device)

    val_loader = build_dataloaders(args.data_path, args.batch_size, eval_only=True)

    avg_loss, acc, preds, gts = evaluate(model, val_loader, device, metrics=True, teacher=args.teacher)

    labels = [0, 1]
    target_names = ["no_person(0)", "person(1)"]
    print(f"val loss: {avg_loss} \nval acc: {acc}")
    print(confusion_matrix(gts, preds, labels=labels))
    print(classification_report(gts, preds, labels=labels, target_names=target_names))


if __name__ == "__main__":
    main()
