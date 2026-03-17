"""
Inspect checkpoints for non-float32 tensors.

Example usage:
python scripts/check_checkpoint_dtype.py saved_runs/2025-11-15_13-13-17_student_mbv3s_vww96_prune/model_pruned_65_full.pt
python scripts/check_checkpoint_dtype.py saved_runs/2025-11-15_13-13-17_student_mbv3s_vww96_prune --pattern 'model*.pt'
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

try:
    from src.engine.models import build_model
except Exception:
    build_model = None  # type: ignore


def _find_checkpoints(path: Path, pattern: str) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob(pattern))
    return []


def _maybe_build_model(model_type: Optional[str]) -> torch.nn.Module:
    if not model_type:
        raise TypeError("Checkpoint is a state dict; supply --model_type to instantiate the architecture.")
    if build_model is None:
        raise ImportError("Unable to import build_model from src.engine.models; run from repo root.")
    return build_model(model_type)


def _is_tensor_mapping(obj) -> bool:
    return isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values())


def _load_model_from_obj(obj, model_type: Optional[str]) -> torch.nn.Module:
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
        elif _is_tensor_mapping(obj):
            state_dict = obj
    elif isinstance(obj, torch.nn.modules.container.OrderedDict):
        state_dict = obj

    if module is None and state_dict is None:
        raise TypeError(f"Loaded object is not a module or state dict (type={type(obj)}).")

    if module is not None:
        module.eval()
        return module

    model = _maybe_build_model(model_type)
    model.load_state_dict(state_dict)  # type: ignore[arg-type]
    model.eval()
    return model


def _load_checkpoint(path: Path, model_type: Optional[str]):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if _is_tensor_mapping(obj) and model_type is None:
        return obj  # Return raw mapping for dtype inspection
    try:
        return _load_model_from_obj(obj, model_type)
    except Exception as exc:
        try:
            jit_mod = torch.jit.load(path, map_location="cpu")
            jit_mod.eval()
            return jit_mod
        except Exception:
            raise exc


def _summarize_tensor_mapping(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    counter: Counter = Counter(t.dtype for t in state.values())
    return {str(k): int(v) for k, v in counter.items()}


def _summarize_dtypes(model: torch.nn.Module) -> Tuple[Counter, Counter]:
    p_counter: Counter = Counter(p.dtype for _, p in model.named_parameters())
    b_counter: Counter = Counter(p.dtype for _, p in model.named_buffers())
    return p_counter, b_counter


def _list_non_f32(names_and_tensors: Iterable[Tuple[str, torch.Tensor]]) -> List[str]:
    return [name for name, tensor in names_and_tensors if tensor.dtype != torch.float32]


def inspect_checkpoint(path: Path, model_type: Optional[str]) -> Dict:
    obj = _load_checkpoint(path, model_type)

    if isinstance(obj, dict) and _is_tensor_mapping(obj):
        param_dtypes = _summarize_tensor_mapping(obj)
        non_f32_params = [k for k, v in obj.items() if v.dtype != torch.float32]
        return {
            "path": str(path),
            "param_dtypes": param_dtypes,
            "buffer_dtypes": {},
            "non_float32_params": non_f32_params,
            "non_float32_buffers": [],
        }

    model = obj  # type: ignore[assignment]
    p_counter, b_counter = _summarize_dtypes(model)
    param_names = list(model.named_parameters())
    buffer_names = list(model.named_buffers())

    non_f32_params = _list_non_f32(param_names)
    non_f32_buffers = _list_non_f32(buffer_names)

    return {
        "path": str(path),
        "param_dtypes": {str(k): int(v) for k, v in p_counter.items()},
        "buffer_dtypes": {str(k): int(v) for k, v in b_counter.items()},
        "non_float32_params": non_f32_params,
        "non_float32_buffers": non_f32_buffers,
    }


def main():
    parser = argparse.ArgumentParser(description="Report parameter/buffer dtypes in checkpoints.")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Checkpoint file(s) or directory/directories to scan.",
    )
    parser.add_argument(
        "--pattern",
        default="model*.pt",
        help="Glob pattern to use when scanning directories (default: model*.pt).",
    )
    parser.add_argument(
        "--model_type",
        choices=["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b2"],
        default=None,
        help="Model architecture to instantiate when checkpoints are state dicts.",
    )
    args = parser.parse_args()

    checkpoints: List[Path] = []
    for p in args.paths:
        found = _find_checkpoints(p, args.pattern)
        if not found:
            print(f"{p}: no checkpoints found.")
        checkpoints.extend(found)

    if not checkpoints:
        return

    for ckpt in checkpoints:
        try:
            summary = inspect_checkpoint(ckpt, args.model_type)
        except Exception as exc:
            size = ckpt.stat().st_size if ckpt.exists() else 0
            hint = f" (file is very small: {size} bytes; may be a failed save)" if size < 4096 else ""
            print(f"{ckpt}: error reading checkpoint ({exc}){hint}")
            continue

        non_f32_params = summary["non_float32_params"]
        non_f32_buffers = summary["non_float32_buffers"]
        p_dtypes = summary["param_dtypes"]
        b_dtypes = summary.get("buffer_dtypes", {})

        status = "all float32" if not non_f32_params and not non_f32_buffers else "mixed dtypes"
        print(f"{ckpt}: {status}")
        print(f"  param dtypes: {p_dtypes}")
        if b_dtypes:
            print(f"  buffer dtypes: {b_dtypes}")
        if non_f32_params:
            print(f"  non-float32 params (first 10): {non_f32_params[:10]}")
        if non_f32_buffers:
            print(f"  non-float32 buffers (first 10): {non_f32_buffers[:10]}")


if __name__ == "__main__":
    main()
