# VWW

Main repository for the MSc dissertation project: "Building Lightweight Neural Networks".

A repeatable training, compression, and deployment pipeline for the Visual Wake Words (VWW) task.

## Setup

This repo uses Python `3.12.6`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Run all commands from the repository root.

## Dataset Preparation

The training pipeline expects image folders in this layout:

```text
data/vww96/{train,val}/{0,1}
```

Where `1` means `person` and `0` means `no_person`.

To build the dataset from MS COCO:

```bash
python scripts/prepare_vww.py --year 2017 --size 96
```

For the larger teacher experiment:

```bash
python scripts/prepare_vww.py --year 2017 --size 224
```

## Main Commands

Train the baseline MobileNetV3-S model:

```bash
python -m src.train --config_path src/config/baseline.yaml
```

Train the teacher models:

```bash
python -m src.train --config_path src/config/teacher_vww96.yaml
python -m src.train --config_path src/config/teacher_vww224.yaml
```

Train the distilled student:

```bash
python -m src.distill --config_path src/config/student.yaml
```

Run iterative structured pruning:

```bash
python -m src.prune --config_path src/config/prune.yaml
```

## Core ML Export And Evaluation

Export a trained checkpoint to Core ML:

```bash
python -m src.export_to_coreml \
  --config_path src/config/baseline.yaml \
  --ckpt_path saved_runs/2025-11-04_17-28-09_baseline_mbv3s_vww96/model.pt \
  --output_path coreml_models/baseline/baseline.mlpackage
```

Evaluate a Core ML model on the VWW validation set:

```bash
python -m src.evaluate_coreml_model --model_path coreml_models/baseline/baseline.mlpackage
```

Quantize a Core ML model using the YAML config in `src/config/quantize_coreml.yaml`:

```bash
python -m src.quantize_coreml
```

## Repository Layout

- `src/`: training, distillation, pruning, quantization, export, and evaluation code
- `src/config/`: configuration files for each model variant
- `scripts/`: dataset preparation helpers
- `saved_runs/`: saved checkpoints and logs, used to export to Core ML
- `coreml_models/`: exported and quantized Core ML artifacts

## Notes

- `runs/` is used for newly created training runs, and only created when the first run is started.
- `saved_runs/` contains the archived runs used for the dissertation experiments.
