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

# also emit a confusion matrix figure + per-class metrics CSV/LaTeX
python -m src.evaluate_coreml_model --model_path coreml_models/baseline_fp32.mlpackage --data_path data/vww96 \
  --save_cm_plot --cm_normalize true \
  --save_per_class_metrics
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import coremltools as ct
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.engine.data import build_dataloaders


def _resolve_io_names(
    mlmodel: ct.models.MLModel,
    input_name: Optional[str],
    output_name: Optional[str],
) -> Tuple[str, str]:
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


def _normalize_confusion_matrix(cm: np.ndarray, mode: str) -> np.ndarray:
    """
    Normalize confusion matrix counts.

    mode:
      - "none": raw counts
      - "true": row-normalized (per true label)
      - "pred": column-normalized (per predicted label)
      - "all": normalized by total count
    """
    cm = cm.astype(np.float64)
    if mode == "none":
        return cm

    eps = 1e-12
    if mode == "true":
        return cm / (cm.sum(axis=1, keepdims=True) + eps)
    if mode == "pred":
        return cm / (cm.sum(axis=0, keepdims=True) + eps)
    if mode == "all":
        return cm / (cm.sum() + eps)

    raise ValueError(f"Unsupported normalization mode: {mode}")


def _save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    out_path: Path,
    normalize: str = "true",
) -> None:
    """
    Save a confusion matrix plot to out_path (pdf/png/etc based on suffix).
    Uses matplotlib only when called (so the script runs without it unless requested).
    """
    import matplotlib

    matplotlib.use("Agg")  # safe for headless/CLI runs
    import matplotlib.pyplot as plt  # noqa: WPS433

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

    # Annotate values
    if normalize == "none":
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
    if normalize != "none":
        title += f" (normalized: {normalize})"
    ax.set_title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _extract_per_class_metrics(
    report_dict: Dict,
    class_keys: List[str],
) -> List[Dict[str, object]]:
    """
    Extract per-class precision/recall/f1/support rows from sklearn classification_report(..., output_dict=True).
    """
    rows: List[Dict[str, object]] = []
    for k in class_keys:
        if k not in report_dict:
            raise KeyError(f"Expected class key '{k}' not found in classification_report output_dict keys: {list(report_dict.keys())}")
        rows.append(
            {
                "class": k,
                "precision": float(report_dict[k]["precision"]),
                "recall": float(report_dict[k]["recall"]),
                "f1": float(report_dict[k]["f1-score"]),
                "support": int(report_dict[k]["support"]),
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
        for r in rows:
            writer.writerow(
                {
                    "model": model_tag,
                    "class": r["class"],
                    "precision": f'{r["precision"]:.6f}',
                    "recall": f'{r["recall"]:.6f}',
                    "f1": f'{r["f1"]:.6f}',
                    "support": r["support"],
                }
            )


def _write_per_class_latex_rows(
    model_tag: str,
    rows: List[Dict[str, object]],
    out_path: Path,
) -> None:
    """
    Writes LaTeX table rows (no header/tabular env) for easy copy/paste.
    Format: Model & Class & Precision & Recall & F1 \\\\
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(
                f"{model_tag} & {r['class']} & {r['precision']:.4f} & {r['recall']:.4f} & {r['f1']:.4f} \\\\\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Core ML .mlpackage model on the VWW val set.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the Core ML .mlpackage or .mlmodel file.")
    parser.add_argument("--data_path", type=Path, required=True, help="Dataset root containing val/0 and val/1 folders (e.g., data/vww96).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation. Must match the batch dimension used during export.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument(
        "--compute_unit",
        type=str,
        default="cpu_only",
        choices=["all", "cpu_and_gpu", "cpu_only", "neural_engine"],
        help="Target compute units for Core ML execution.",
    )
    parser.add_argument("--input_name", type=str, default=None, help="Override input feature name. Defaults to the first model input.")
    parser.add_argument("--output_name", type=str, default=None, help="Override output feature name. Defaults to the first model output.")
    parser.add_argument(
        "--metrics_output",
        type=Path,
        default=None,
        help="Optional path to save metrics JSON. Default: <model_dir>/<model_stem>_coreml_eval_metrics.json",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="Optional short name used in exported per-class CSV/LaTeX rows. Default: model_path.stem",
    )

    # Optional exports for dissertation artefacts
    parser.add_argument(
        "--save_cm_plot",
        action="store_true",
        help="If set, save a confusion matrix figure (PDF/PNG based on output suffix).",
    )
    parser.add_argument(
        "--cm_plot_output",
        type=Path,
        default=None,
        help="Optional output path for confusion matrix figure. Default: <metrics_dir>/<model_stem>_cm_<normalize>.pdf",
    )
    parser.add_argument(
        "--cm_normalize",
        type=str,
        default="true",
        choices=["none", "true", "pred", "all"],
        help="Normalization mode for the confusion matrix figure (raw counts or normalized).",
    )
    parser.add_argument(
        "--save_per_class_metrics",
        action="store_true",
        help="If set, export per-class precision/recall/F1 as CSV and LaTeX rows.",
    )
    parser.add_argument(
        "--per_class_csv_output",
        type=Path,
        default=None,
        help="Optional output path for per-class metrics CSV. Default: <metrics_dir>/<model_stem>_per_class.csv",
    )
    parser.add_argument(
        "--per_class_latex_output",
        type=Path,
        default=None,
        help="Optional output path for per-class metrics LaTeX rows. Default: <metrics_dir>/<model_stem>_per_class_rows.tex",
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

    preds: List[int] = []
    gts: List[int] = []

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

    cls_report_text = classification_report(gts_arr, preds_arr, labels=labels, target_names=target_names)
    cls_report_dict = classification_report(
        gts_arr, preds_arr, labels=labels, target_names=target_names, output_dict=True
    )

    print(f"val acc: {acc:.4f}")
    print(cm)
    print(cls_report_text)

    default_metrics = model_path.parent / f"{model_path.stem}_coreml_eval_metrics.json"
    metrics_path = args.metrics_output or default_metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w") as fh:
        json.dump(
            {
                "model_path": str(model_path),
                "data_path": str(data_path),
                "val_acc": acc,
                "confusion_matrix": cm.tolist(),  # raw counts
                "classification_report": cls_report_dict,
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

    # Optional outputs for dissertation
    model_tag = args.model_tag or model_path.stem

    if args.save_cm_plot:
        default_cm_path = metrics_path.parent / f"{model_path.stem}_cm_{args.cm_normalize}.pdf"
        cm_out = (args.cm_plot_output.expanduser() if args.cm_plot_output else default_cm_path)
        _save_confusion_matrix_plot(cm=cm, class_names=target_names, out_path=cm_out, normalize=args.cm_normalize)
        print(f"Saved confusion matrix figure to {cm_out}")

    if args.save_per_class_metrics:
        per_class_rows = _extract_per_class_metrics(cls_report_dict, class_keys=target_names)

        default_csv = metrics_path.parent / f"{model_path.stem}_per_class.csv"
        csv_out = (args.per_class_csv_output.expanduser() if args.per_class_csv_output else default_csv)
        _write_per_class_csv(model_tag=model_tag, rows=per_class_rows, out_path=csv_out)
        print(f"Saved per-class metrics CSV to {csv_out}")

        default_tex = metrics_path.parent / f"{model_path.stem}_per_class_rows.tex"
        tex_out = (args.per_class_latex_output.expanduser() if args.per_class_latex_output else default_tex)
        _write_per_class_latex_rows(model_tag=model_tag, rows=per_class_rows, out_path=tex_out)
        print(f"Saved per-class metrics LaTeX rows to {tex_out}")


if __name__ == "__main__":
    main()
