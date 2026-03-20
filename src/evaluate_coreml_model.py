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
python -m src.evaluate_coreml_model --model_path coreml_models/pruned_int8.mlpackage --data_path data/vww96 --cpu_only

# optional confusion matrix figure + per-class metrics CSV/LaTeX
python -m src.evaluate_coreml_model --model_path coreml_models/baseline_fp32.mlpackage --data_path data/vww96 \
  --save_cm_plot --cm_normalize \
  --save_per_class_metrics
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import coremltools as ct
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.engine.data import build_dataloaders


def _get_io_names(mlmodel: ct.models.MLModel) -> Tuple[str, str]:
    """Return the first declared Core ML input and output names."""
    spec = mlmodel.get_spec()

    if not spec.description.input:
        raise ValueError("Core ML model has no inputs.")
    input_name = spec.description.input[0].name

    if not spec.description.output:
        raise ValueError("Core ML model has no outputs.")
    output_name = spec.description.output[0].name

    return input_name, output_name


def _normalize_confusion_matrix(cm: np.ndarray, normalize: bool) -> np.ndarray:
    """Return raw counts or a confusion matrix normalized over all entries."""
    cm = cm.astype(np.float64)
    if not normalize:
        return cm

    eps = 1e-12
    return cm / (cm.sum() + eps)


def _save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    out_path: Path,
    normalize: bool = False,
) -> None:
    """Save a confusion matrix plot to out_path (pdf/png/etc based on suffix)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_disp = _normalize_confusion_matrix(cm, normalize)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm_disp, interpolation="nearest")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    if not normalize:
        fmt = "{:d}"
        values = cm.astype(int)
    else:
        fmt = "{:.2f}"
        values = cm_disp

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                fmt.format(values[i, j]),
                ha="center",
                va="center",
            )

    title = "Confusion matrix"
    if normalize:
        title += " (normalized)"
    ax.set_title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _get_per_class_metrics(
    report_dict: Dict,
    class_keys: List[str],
) -> List[Dict[str, object]]:
    """
    Get per-class precision/recall/f1/support rows from sklearn classification_report(..., output_dict=True).
    """
    rows: List[Dict[str, object]] = []
    for key in class_keys:
        if key not in report_dict:
            raise KeyError(
                f"Expected class key '{key}' not found in classification_report output_dict keys: "
                f"{list(report_dict.keys())}"
            )
        rows.append(
            {
                "class": key,
                "precision": float(report_dict[key]["precision"]),
                "recall": float(report_dict[key]["recall"]),
                "f1": float(report_dict[key]["f1-score"]),
                "support": int(report_dict[key]["support"]),
            }
        )
    return rows


def _write_per_class_csv(
    model_tag: str,
    rows: List[Dict[str, object]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "class", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": model_tag,
                    "class": row["class"],
                    "precision": f'{row["precision"]:.6f}',
                    "recall": f'{row["recall"]:.6f}',
                    "f1": f'{row["f1"]:.6f}',
                    "support": row["support"],
                }
            )


def _write_per_class_latex_rows(
    model_tag: str,
    rows: List[Dict[str, object]],
    out_path: Path,
) -> None:
    """
    Write LaTeX table rows (no header/tabular env) for easy copy/paste.
    Format: Model & Class & Precision & Recall & F1 \\\\
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in rows:
            fh.write(
                f"{model_tag} & {row['class']} & {row['precision']:.4f} & "
                f"{row['recall']:.4f} & {row['f1']:.4f} \\\\\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Core ML .mlpackage model on the VWW val set.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the Core ML .mlpackage file.")
    parser.add_argument("--data_path", type=Path, required=True, help="Dataset root containing val/0 and val/1 folders (e.g., data/vww96).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation. Must match the batch dimension used during export.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="If set, force Core ML evaluation to use CPU only.",
    )
    parser.add_argument(
        "--save_cm_plot",
        action="store_true",
        help="If set, save a confusion matrix figure (PDF/PNG based on output suffix).",
    )
    parser.add_argument(
        "--cm_normalize",
        action="store_true",
        help="If set, normalize the confusion matrix over all entries before plotting.",
    )
    parser.add_argument(
        "--save_per_class_metrics",
        action="store_true",
        help="If set, export per-class precision/recall/F1 as CSV and LaTeX rows.",
    )
    args = parser.parse_args()

    model_path = args.model_path.expanduser()
    data_path = args.data_path.expanduser()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if args.cpu_only:
        mlmodel = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    else:
        mlmodel = ct.models.MLModel(str(model_path))
    input_name, output_name = _get_io_names(mlmodel)

    val_loader = build_dataloaders(
        data_path=data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_only=True,
    )

    preds: List[int] = []
    gts: List[int] = []

    for x, y in val_loader:
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
    cls_report_text = classification_report(gts_arr, preds_arr, labels=labels, target_names=target_names)
    cls_report_dict = classification_report(
        gts_arr, preds_arr, labels=labels, target_names=target_names, output_dict=True
    )

    print(f"val acc: {acc:.4f}")
    print(cm)
    print(cls_report_text)

    metrics_path = model_path.parent / f"{model_path.stem}_coreml_eval_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w") as fh:
        json.dump(
            {
                "model_path": str(model_path),
                "data_path": str(data_path),
                "val_acc": acc,
                "confusion_matrix": cm.tolist(),
                "classification_report": cls_report_dict,
                "num_samples": int(len(gts_arr)),
                "batch_size": args.batch_size,
                "cpu_only": args.cpu_only,
                "input_name": input_name,
                "output_name": output_name,
            },
            fh,
            indent=2,
        )
    print(f"Saved detailed metrics to {metrics_path}")

    model_tag = model_path.stem

    if args.save_cm_plot:
        cm_out = Path(__file__).resolve().parents[1] / "figures" / f"{model_path.stem}_cm.pdf"
        _save_confusion_matrix_plot(cm=cm, class_names=target_names, out_path=cm_out, normalize=args.cm_normalize)
        print(f"Saved confusion matrix figure to {Path("figures") / cm_out.name}")

    if args.save_per_class_metrics:
        per_class_rows = _get_per_class_metrics(cls_report_dict, class_keys=target_names)

        csv_out = metrics_path.parent / f"{model_path.stem}_per_class.csv"
        _write_per_class_csv(model_tag=model_tag, rows=per_class_rows, out_path=csv_out)
        print(f"Saved per-class metrics CSV to {csv_out}")

        tex_out = metrics_path.parent / f"{model_path.stem}_per_class_rows.tex"
        _write_per_class_latex_rows(model_tag=model_tag, rows=per_class_rows, out_path=tex_out)
        print(f"Saved per-class metrics LaTeX rows to {tex_out}")


if __name__ == "__main__":
    main()
