import argparse
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, classification_report

from src.engine.train_loops import evaluate
from src.engine.utils import set_seed, compute_model_complexity
from src.data import build_dataloaders
from src.models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["mobilenet_v3_small", "mobilenet_v3_large"], required=True, type=str)
    ap.add_argument("--model_path", required=True, type=Path, help="Path to model (model.pt)")
    ap.add_argument("--data_path", required=True, type=Path, help="Path to data (e.g. data/vww96)")
    ap.add_argument("--batch_size", required=True, type=int, help="Batch size (128 or 256)")
    ap.add_argument("--teacher_224", required=False, default=False, type=bool, help="Evaluating a 224 teacher model?")
    args = ap.parse_args()

    set_seed(42)

    model = build_model(args.model_type)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    checkpt = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpt, strict=True)
    model.to(device)

    val_loader = build_dataloaders(args.data_path, args.batch_size, eval_only=True)

    avg_loss, acc, preds, gts = evaluate(model, val_loader, device, metrics=True, teacher_224=args.teacher_224)

    labels = [0, 1]
    target_names = ["no_person(0)", "person(1)"]
    print(f"val loss: {avg_loss}")
    print(f"val acc: {acc}")

    complexity = compute_model_complexity(model, loader=val_loader)
    if complexity:
        param_count = complexity["param_count"]
        macs = complexity["macs"]
        print(f"model params: {param_count:,} ({param_count/1e6:.2f}M)")
        print(f"model macs: {macs:,} ({macs/1e6:.2f}M)")

    print(confusion_matrix(gts, preds, labels=labels))
    print(classification_report(gts, preds, labels=labels, target_names=target_names))


if __name__ == "__main__":
    main()
