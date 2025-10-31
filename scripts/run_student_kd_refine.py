#!/usr/bin/env python3
"""Launch a short KD fine-tune for the MobilenetV3-S student."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _invoke_kd(config_path: Path, debug: bool):
    """Call the shared KD entrypoint with a fixed config path."""
    from src import kd as kd_module  # Imported lazily to avoid argparse side-effects

    argv = ["kd_finetune", "--config_path", str(config_path)]
    if debug:
        argv.extend(["--debug", "True"])

    previous = sys.argv
    try:
        sys.argv = argv
        kd_module.main()
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser(description="Run the student KD refinement task.")
    repo_root = REPO_ROOT
    default_config = repo_root / "src" / "config" / "student_mbv3s_vww96_refine.yaml"

    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to YAML config (default: {default_config.relative_to(repo_root)})",
    )
    parser.add_argument("--debug", action="store_true", help="Print parsed config context.")
    args = parser.parse_args()

    config_path = args.config.expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    _invoke_kd(config_path, args.debug)


if __name__ == "__main__":
    main()
