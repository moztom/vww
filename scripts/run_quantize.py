#!/usr/bin/env python3
"""Launch quantization (PTQ or QAT) for the MobilenetV3-S student."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _invoke_quantize(config_path: Path):
    from src import quantize as quant_module  # Lazy import to avoid argparse conflicts.

    argv = ["quantize", "--config_path", str(config_path)]
    previous = sys.argv
    try:
        sys.argv = argv
        quant_module.main()
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser(description="Run PTQ/QAT quantization on the student model.")
    repo_root = REPO_ROOT
    default_config = repo_root / "src" / "config" / "student_mbv3s_vww96_quant.yaml"

    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to YAML config (default: {default_config.relative_to(repo_root)})",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    _invoke_quantize(config_path)


if __name__ == "__main__":
    main()
