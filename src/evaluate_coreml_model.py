"""
Evaluate an exported Core ML .mlpackage model on the VWW validation set.

Examples:
# baseline
python -m src.evaluate_coreml_model --model_path coreml_models/baseline_fp32.mlpackage --data_path data/vww96
# student
python -m src.evaluate_coreml_model --model_path coreml_models/student_fp32.mlpackage --data_path data/vww96
# pruned
python -m src.evaluate_coreml_model --model_path coreml_models/pruned_fp32.mlpackage --data_path data/vww96
# quantized (int8 weights, still float inputs)
python -m src.evaluate_coreml_model --model_path coreml_models/pruned_int8.mlpackage --data_path data/vww96 --compute_unit cpu_only
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import coremltools as ct
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.engine.data import build_dataloaders


def _resolve_io_names(mlmodel: ct.models.MLModel, input_name: Optional[str], output_name: Optional[str]) -> Tuple[str, str]:
    """Return concrete input/output names, falling back to the first entries in the spec."""
    spec = mlmodel.get_spec()

    if input_name is None:
        if not spec.description.input:
            raise ValueError("Core ML model has no inputs.")
        input_name = spec.description.input[0].name

    if output_name is None:
        if not spec.description.output:
            raise ValueError("Core ML model has no outputs.")
        output_name = spec.description.output[0].name

    return input_name, output_name


def _compute_unit(name: str) -> ct.ComputeUnit:
    lookup = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "neural_engine": ct.ComputeUnit.CPU_AND_NE,
    }
    key = name.lower()
    if key not in lookup:
        raise ValueError(f"Unsupported compute unit '{name}'. Choose from: {', '.join(lookup)}")
    return lookup[key]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Core ML .mlpackage model on the VWW val set.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the Core ML .mlpackage or .mlmodel file.")
    parser.add_argument("--data_path", type=Path, required=True, help="Dataset root containing val/0 and val/1 folders (e.g., data/vww96).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation. Must match the batch dimension used during export.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--compute_unit", type=str, default="cpu_only",
        choices=["all", "cpu_and_gpu", "cpu_only", "neural_engine"],
        help="Target compute units for Core ML execution.",
    )
    parser.add_argument("--input_name", type=str, default=None, help="Override input feature name. Defaults to the first model input.")
    parser.add_argument("--output_name", type=str, default=None, help="Override output feature name. Defaults to the first model output.")
    parser.add_argument(
        "--metrics_output",
        type=Path,
        default=None,
        help="Optional path to save metrics JSON. Default: <model_dir>/coreml_eval_metrics.json",
    )
    args = parser.parse_args()

    model_path = args.model_path.expanduser()
    data_path = args.data_path.expanduser()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    mlmodel = ct.models.MLModel(str(model_path), compute_units=_compute_unit(args.compute_unit))
    input_name, output_name = _resolve_io_names(mlmodel, args.input_name, args.output_name)

    val_loader = build_dataloaders(
        data_path=data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_only=True,
    )

    preds = []
    gts = []

    for x, y in val_loader:
        # Core ML expects NCHW float32 arrays; dataloader already applied normalization.
        x_np = np.ascontiguousarray(x.numpy().astype(np.float32))
        out = mlmodel.predict({input_name: x_np})

        if output_name not in out:
            raise KeyError(f"Output '{output_name}' not found in Core ML outputs: {list(out.keys())}")

        logits = np.array(out[output_name])
        if logits.ndim == 1:
            logits = logits[None, :]
        batch_preds = np.argmax(logits, axis=1)

        preds.extend(batch_preds.tolist())
        gts.extend(y.numpy().astype(int).tolist())

    preds_arr = np.array(preds)
    gts_arr = np.array(gts)
    acc = float((preds_arr == gts_arr).mean())

    labels = [0, 1]
    target_names = ["no_person(0)", "person(1)"]
    cm = confusion_matrix(gts_arr, preds_arr, labels=labels)
    cls_report = classification_report(gts_arr, preds_arr, labels=labels, target_names=target_names)

    print(f"val acc: {acc:.4f}")
    print(cm)
    print(cls_report)

    default_metrics = model_path.parent / f"{model_path.stem}_coreml_eval_metrics.json"
    metrics_path = args.metrics_output or default_metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as fh:
        json.dump(
            {
                "model_path": str(model_path),
                "data_path": str(data_path),
                "val_acc": acc,
                "confusion_matrix": cm.tolist(),
                "classification_report": classification_report(
                    gts_arr, preds_arr, labels=labels, target_names=target_names, output_dict=True
                ),
                "num_samples": int(len(gts_arr)),
                "batch_size": args.batch_size,
                "compute_unit": args.compute_unit,
                "input_name": input_name,
                "output_name": output_name,
            },
            fh,
            indent=2,
        )
    print(f"Saved detailed metrics to {metrics_path}")


if __name__ == "__main__":
    main()
