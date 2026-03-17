"""
Remove THOP profiling artifacts and force float32 in pruned checkpoints.

Example (in-place):
python scripts/clean_pruned_checkpoint.py saved_runs/2025-11-15_13-13-17_student_mbv3s_vww96_prune/model_pruned_65_full.pt --inplace

Example (write to new file):
python scripts/clean_pruned_checkpoint.py saved_runs/2025-11-15_13-13-17_student_mbv3s_vww96_prune/model_pruned_65_full.pt --output saved_runs/2025-11-15_13-13-17_student_mbv3s_vww96_prune/model_pruned_65_full_clean.pt
"""

import argparse
import copy
from pathlib import Path

import torch


def _strip_thop_stats(model: torch.nn.Module) -> None:
    for module in model.modules():
        for name in ("total_ops", "total_params"):
            if name in getattr(module, "_buffers", {}):
                module._buffers.pop(name, None)
            if hasattr(module, name):
                try:
                    delattr(module, name)
                except Exception:
                    pass


def clean_checkpoint(input_path: Path, output_path: Path) -> None:
    obj = torch.load(input_path, map_location="cpu", weights_only=False)
    model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected a torch.nn.Module in {input_path}, found {type(model)}.")

    cleaned = copy.deepcopy(model).float().cpu()
    _strip_thop_stats(cleaned)
    torch.save({"model": cleaned}, output_path)
    print(f"Saved cleaned checkpoint to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Strip THOP stats and force float32 in pruned checkpoints.")
    parser.add_argument("input", type=Path, help="Path to pruned checkpoint (full module, e.g., model_pruned_XX_full.pt).")
    parser.add_argument("--output", type=Path, default=None, help="Output path. If omitted and --inplace is not set, appends _clean.")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input file.")
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    if args.inplace:
        output_path = input_path
    else:
        output_path = args.output or input_path.with_name(input_path.stem + "_clean.pt")

    clean_checkpoint(input_path, output_path)


if __name__ == "__main__":
    main()
