"""
Export a trained PyTorch model to a Core ML .mlpackage file

Example usage:
# baseline
python -m src.export_to_coreml --config_path src/config/baseline.yaml --ckpt_path saved_runs/2025-11-04_17-28-09_baseline_mbv3s_vww96/model.pt --output_path coreml_models/baseline/baseline.mlpackage
# student
python -m src.export_to_coreml --config_path src/config/student.yaml --ckpt_path saved_runs/2025-10-31_21-20-46_student_mbv3s_vww96/model.pt --output_path coreml_models/student/student.mlpackage 
# pruned student
python -m src.export_to_coreml --config_path src/config/prune.yaml --ckpt_path saved_runs/2025-11-15_13-13-17_pruned_student_mbv3s_vww96/model_pruned_65_full.pt --output_path coreml_models/pruned/pruned.mlpackage
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import coremltools as ct
import torch
import yaml

from src.engine.models import build_model


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_config(config_path: Path) -> dict:
    with config_path.open("r") as handle:
        cfg = yaml.safe_load(handle) or {}
    return cfg


def _extract_model_from_checkpoint(ckpt_path: Path) -> Tuple[Optional[torch.nn.Module], Optional[dict]]:
    """Extracts a model or state dict from a checkpoint file"""
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    module = None
    state_dict = None

    if isinstance(obj, torch.nn.Module):
        module = obj
    elif isinstance(obj, dict):
        maybe_model = obj.get("model")
        maybe_state = obj.get("model_state")
        if isinstance(maybe_model, torch.nn.Module):
            module = maybe_model
        elif isinstance(maybe_model, dict):
            state_dict = maybe_model
        elif isinstance(maybe_state, dict):
            state_dict = maybe_state
        elif isinstance(maybe_state, torch.nn.Module):
            state_dict = maybe_state.state_dict()
        else:
            # Assume plain state dict
            state_dict = obj
    else:
        state_dict = obj

    if module is not None:
        module.to("cpu")
        module.eval()
    elif state_dict is not None and isinstance(state_dict, torch.nn.Module):
        state_dict = state_dict.state_dict()

    return module, state_dict


def _is_fx_graph_module(module: torch.nn.Module) -> bool:
    """Check if a module is a torch.fx GraphModule"""
    cls = module.__class__
    module_name = getattr(cls, "__module__", "")
    name = getattr(cls, "__name__", "")
    if "torch.fx" in module_name or name == "GraphModule":
        return True
    try:
        from torch.fx import GraphModule as FxGraphModule  # type: ignore
    except ImportError:
        FxGraphModule = None
    return FxGraphModule is not None and isinstance(module, FxGraphModule)


def _build_model_from_state(cfg: dict, state_dict: dict) -> torch.nn.Module:
    """Builds a model from config and loads the provided state dict"""
    model_cfg = cfg.get("model") or {}
    model_type = model_cfg.get("type")
    if not model_type:
        raise ValueError("Config missing model.type; cannot build model for state dict checkpoint.")
    pretrained = bool(model_cfg.get("pretrained", False))
    model = build_model(model_type, pretrained=pretrained)
    model.to("cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_model(config_path: Path, ckpt_path: Path) -> torch.nn.Module:
    """Loads a model from config and checkpoint paths"""
    cfg = _load_config(config_path)
    module, state_dict = _extract_model_from_checkpoint(ckpt_path)

    if module is not None:
        if _is_fx_graph_module(module):
            raise ValueError(
                "Checkpoint contains a PT2E/FX GraphModule, which cannot be exported to Core ML. "
            )
        return module

    if state_dict is None:
        raise ValueError(f"Unable to interpret checkpoint at {ckpt_path}")

    return _build_model_from_state(cfg, state_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, required=True, help="Path to the YAML config used to build the model architecture.")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Path to the trained checkpoint (.pt) file to export.")
    parser.add_argument("--output_path", type=Path, required=True, help="Destination path for the Core ML .mlpackage file.")
    parser.add_argument("--input_height", type=int, default=96, help="Input tensor height. Default: 96.")
    parser.add_argument("--input_width", type=int, default=96, help="Input tensor width. Default: 96.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch dimension for the exported model input. Default: 1.")
    args = parser.parse_args()



    config_path = args.config_path.expanduser()
    ckpt_path = args.ckpt_path.expanduser()
    output_path = args.output_path.expanduser()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = _load_model(config_path, ckpt_path)

    # Create example input
    height, width = args.input_height, args.input_width
    example_input = torch.randn(args.batch_size, 3, height, width, device="cpu")

    # Trace the model with example input
    with torch.inference_mode():
        traced = torch.jit.trace(model, example_input)
        traced.eval()

    # Convert to Core ML
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input",
                shape=example_input.shape,
            )
        ],
        outputs=[
            ct.TensorType(
                name="output",
            )
        ],
        compute_units=ct.ComputeUnit.ALL,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(output_path)
    print(f"Saved Core ML model to {output_path}")


if __name__ == "__main__":
    main()
