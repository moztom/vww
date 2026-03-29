import json
from pathlib import Path

import coremltools as ct
import coremltools.optimize as cto
import yaml

from src.evaluate_coreml_model import evaluate_coreml_model


def _load_config(path: Path) -> dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    if "coreml_quant" not in config:
        raise ValueError(f"Missing 'coreml_quant' section in {path}")

    return config


def _build_quant_config(quant_cfg: dict):
    mode = str(quant_cfg.get("mode", "linear_symmetric"))
    dtype = str(quant_cfg.get("dtype", "int8"))

    if mode != "linear_symmetric":
        raise ValueError(
            f"Unsupported Core ML quantization mode '{mode}'. "
            "This project uses weight-only linear_symmetric quantization."
        )
    if dtype != "int8":
        raise ValueError(
            f"Unsupported Core ML quantization dtype '{dtype}'. "
            "This project uses weight-only int8 quantization."
        )

    return cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpLinearQuantizerConfig(
            mode=mode,
            dtype=dtype,
        )
    )
def quantize_coreml_model(input_path: Path, output_path: Path, quant_cfg: dict) -> Path:
    input_path = input_path.expanduser()
    output_path = output_path.expanduser()

    if not input_path.exists():
        raise FileNotFoundError(f"Input Core ML model not found: {input_path}")

    if output_path.suffix != ".mlpackage":
        output_path = output_path.with_suffix(".mlpackage")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Core ML model from {input_path}")
    mlmodel = ct.models.MLModel(str(input_path))

    print("Applying 8-bit linear symmetric weight quantization...")
    quantized_model = cto.coreml.linear_quantize_weights(mlmodel, _build_quant_config(quant_cfg))
    quantized_model.save(output_path)
    print(f"Saved quantized model to {output_path}")

    return output_path


def _write_summary(summary_path: Path, payload: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as file:
        json.dump(payload, file, indent=2)


def run_coreml_quantization(config_path: Path) -> dict:
    config_path = config_path.expanduser()
    cfg = _load_config(config_path)

    quant_cfg = cfg["coreml_quant"]
    input_path = Path(quant_cfg["input_path"])
    output_path = Path(quant_cfg["output_path"])

    output_path = quantize_coreml_model(
        input_path=input_path,
        output_path=output_path,
        quant_cfg=quant_cfg,
    )

    summary = {
        "config_path": str(config_path),
        "input_path": str(input_path.expanduser()),
        "output_path": str(output_path),
        "mode": str(quant_cfg.get("mode", "linear_symmetric")),
        "dtype": str(quant_cfg.get("dtype", "int8")),
        "output_bytes": output_path.stat().st_size,
        "output_mb": output_path.stat().st_size / (1024 * 1024),
        "evaluation": {
            "enabled": False,
        },
    }

    eval_cfg = cfg.get("eval", {})
    if eval_cfg.get("enabled", False):
        data_path = eval_cfg.get("data_path")
        if not data_path:
            raise ValueError("Core ML quantization config has eval.enabled=true but no eval.data_path.")

        eval_result = evaluate_coreml_model(
            model_path=output_path,
            data_path=Path(data_path),
            batch_size=int(eval_cfg.get("batch_size", 1)),
            num_workers=int(eval_cfg.get("num_workers", 4)),
            save_cm_plot=bool(eval_cfg.get("save_cm_plot", False)),
            cm_normalize=bool(eval_cfg.get("cm_normalize", False)),
            save_per_class_metrics=bool(eval_cfg.get("save_per_class_metrics", False)),
        )

        summary["evaluation"] = {
            "enabled": True,
            "metrics_path": str(eval_result["metrics_path"]),
            "cm_path": str(eval_result["cm_path"]) if eval_result["cm_path"] else None,
            "per_class_csv_path": (
                str(eval_result["per_class_csv_path"])
                if eval_result["per_class_csv_path"]
                else None
            ),
            "val_acc": eval_result["metrics"]["val_acc"],
            "cpu_only": True,
        }

    summary_path = output_path.parent / f"{output_path.stem}_quant_summary.json"
    _write_summary(summary_path, summary)
    print(f"Saved quantization summary to {summary_path}")

    return summary
