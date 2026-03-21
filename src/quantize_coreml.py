"""
Weight-only post-training quantization for a Core ML model from a YAML config.

Default config: src/config/quantize_coreml_int8.yaml

Example usage:
python -m src.quantize_coreml
"""

import argparse
from pathlib import Path

from src.engine.quantization import run_coreml_quantization


def run_quantization(args: argparse.Namespace) -> None:
    run_coreml_quantization(args.config_path)


def main():
    parser = argparse.ArgumentParser(
        description="Apply weight-only Core ML quantization from a YAML config."
    )
    default_config = Path("src") / "config" / "quantize_coreml_int8.yaml"
    parser.add_argument(
        "--config_path",
        type=Path,
        default=default_config,
        help="Path to the Core ML quantization YAML config.",
    )
    args = parser.parse_args()

    run_quantization(args)


if __name__ == "__main__":
    main()
