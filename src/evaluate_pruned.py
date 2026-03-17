import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
from sklearn.metrics import confusion_matrix, classification_report

from src.engine.data import build_dataloaders
from src.engine.models import build_model
from src.engine.train_loops import evaluate
from src.engine.utils import compute_model_complexity, set_seed


def _ensure_float32(module: torch.nn.Module) -> torch.nn.Module:
    """Move parameters/buffers to float32 to avoid MPS/dtype issues from saved buffers."""
    return module.to(dtype=torch.float32)


def _select_device(preference: str) -> str:
    pref = preference.lower()
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and torch.backends.mps.is_available():
        return "mps"
    if pref == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _split_checkpoint(obj) -> Tuple[Optional[torch.nn.Module], Optional[dict]]:
    """Return a full module or a state dict from a loaded checkpoint-like object."""
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
            state_dict = obj
    else:
        state_dict = obj

    if module is not None:
        module.eval()
    elif state_dict is not None and isinstance(state_dict, torch.nn.Module):
        state_dict = state_dict.state_dict()

    return module, state_dict


def _load_pruned_model(model_path: Path, model_type: Optional[str]) -> torch.nn.Module:
    """
    Load a pruned model checkpoint.
    - Prefers full modules (e.g., model_pruned_65_full.pt saved as {'model': model})
    - Falls back to state dicts if provided, requiring --model_type.
    """
    obj = torch.load(model_path, map_location="cpu", weights_only=False)
    module, state_dict = _split_checkpoint(obj)

    if module is not None:
        return _ensure_float32(module)

    if state_dict is None:
        raise ValueError("Could not interpret checkpoint contents for pruned model.")

    if not model_type:
        raise ValueError("--model_type is required when loading a state-dict pruned checkpoint.")

    model = build_model(model_type)
    model.load_state_dict(state_dict)
    model.eval()
    return _ensure_float32(model)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a pruned model checkpoint on the VWW validation set.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to pruned checkpoint (e.g., model_pruned_65_full.pt).")
    parser.add_argument("--data_path", type=Path, required=True, help="Dataset root containing val/0 and val/1.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for evaluation.")
    parser.add_argument(
        "--model_type",
        choices=["mobilenet_v3_small", "mobilenet_v3_large"],
        required=False,
        help="Architecture to instantiate if checkpoint is a state dict instead of a full module.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device preference for evaluation.",
    )
    parser.add_argument("--teacher_224", action="store_true", help="Set if evaluating a 224 teacher model checkpoint.")
    args = parser.parse_args()

    set_seed(42)

    device = _select_device(args.device)
    model = _load_pruned_model(args.model_path, args.model_type)
    model.to(device)

    val_loader = build_dataloaders(args.data_path, args.batch_size, eval_only=True)
    loss, acc, preds, gts = evaluate(model, val_loader, device, metrics=True, teacher_224=args.teacher_224)

    labels = [0, 1]
    target_names = ["no_person(0)", "person(1)"]

    print(f"val loss: {loss}")
    print(f"val acc: {acc}")

    complexity = compute_model_complexity(model, loader=val_loader)
    if complexity:
        params = complexity["param_count"]
        macs = complexity["macs"]
        print(f"model params: {params:,} ({params/1e6:.2f}M)")
        print(f"model macs: {macs:,} ({macs/1e6:.2f}M)")

    print(confusion_matrix(gts, preds, labels=labels))
    print(classification_report(gts, preds, labels=labels, target_names=target_names))


if __name__ == "__main__":
    main()
