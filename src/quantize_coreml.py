"""
int8 quantize a coreml model

Example usage:
python scripts/coreml/quantize_coreml.py --input_path coreml_models/pruned_fp32.mlpackage --output_path coreml_models/pruned_int8.mlpackage
"""

import argparse
from pathlib import Path

import coremltools as ct
import coremltools.optimize as cto


def quantize_to_int8(input_path: str, output_path: str):
    #input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Core ML model from {input_path}")
    mlmodel = ct.models.MLModel(input_path)

    # Config: 8-bit linear symmetric weight quantization (no activations).
    # This is weight-only PTQ; activations stay fp16/fp32.
    config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",  # or "linear" if you prefer
            dtype="int8",
        )
    )

    print("Applying 8-bit linear symmetric weight quantization...")
    quantized_model = cto.coreml.linear_quantize_weights(mlmodel, config)

    # For ML Program models, extension should be .mlpackage
    if output_path.suffix != ".mlpackage":
        print(
            f"Note: changing extension from {output_path.suffix} "
            f"to .mlpackage for ML Program model."
        )
        output_path = output_path.with_suffix(".mlpackage")

    quantized_model.save(output_path)
    print(f"Saved quantized model to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    quantize_to_int8(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
