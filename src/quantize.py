import argparse
import json
import time
from pathlib import Path

from src.engine.setup import build_context
from src.engine.train_loops import evaluate
from src.engine.utils import compute_model_complexity
from src.engine.quantization import run_quantization


def _append_metrics(run_dir: Path, payload: dict) -> None:
    path = Path(run_dir) / "metrics.jsonl"
    with open(path, "a") as fh:
        fh.write(json.dumps(payload) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Quantize a trained student model (PTQ or QAT).")
    parser.add_argument("--config_path", type=Path, required=True, help="Path to quantization YAML config.")
    args = parser.parse_args()

    ctx = build_context(args.config_path, stage="quant")
    run_dir = ctx["run_dir"]

    '''
    print("Evaluating FP32 baseline before quantization...")
    baseline_start = time.perf_counter()
    base_loss, base_acc, *_ = evaluate(ctx["model"], ctx["val_loader"], ctx["device"])
    baseline_time = time.perf_counter() - baseline_start
    print(f"Baseline FP32 accuracy: {base_acc:.4f} (eval {baseline_time:.1f}s)")

    complexity = compute_model_complexity(ctx["model"], ctx["val_loader"])
    if complexity:
        params = complexity["param_count"]
        macs = complexity["macs"]
        print(
            f"Baseline complexity: params={params:,} ({params/1e6:.2f}M) | "
            f"MACs={macs:,} ({macs/1e6:.2f}M)"
        )
    else:
        params = macs = None
    
    '''

    summary = run_quantization(ctx)

    print(f"\nQuantization complete ({summary['mode']} on backend {summary['backend']}).")
    print(f"INT8 accuracy: {summary['val_acc']:.4f} | loss {summary['val_loss']:.4f}")
    print(f"Saved quantized weights to:\n  - {summary['state_dict_path']}\n  - {summary['full_model_path']}")

    '''
    _append_metrics(run_dir, {
        "tag": "baseline_fp32",
        "val_acc": base_acc,
        "val_loss": base_loss,
        "eval_seconds": baseline_time,
        "param_count": params,
        "macs": macs,
    })
    '''
    _append_metrics(run_dir, {
        "tag": f"quant_{summary['mode']}",
        **summary,
    })

    writer = ctx.get("writer")
    if writer:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
