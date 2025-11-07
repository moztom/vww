#!/usr/bin/env python3
"""Launch iterative pruning for the MobilenetV3-S student."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _invoke_prune(config_path: Path, targets):
    """Call the pruning entrypoint with optional target overrides."""
    from src import prune as prune_module  # Lazy import to avoid argparse conflicts.

    argv = ["prune", "--config_path", str(config_path)]
    if targets:
        argv.append("--targets")
        argv.extend(str(t) for t in targets)

    previous = sys.argv
    try:
        sys.argv = argv
        prune_module.main()
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser(description="Run iterative pruning on the student model.")
    repo_root = REPO_ROOT
    default_config = repo_root / "src" / "config" / "student_mbv3s_vww96_prune.yaml"

    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to YAML config (default: {default_config.relative_to(repo_root)})",
    )
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=None,
        help="Override prune targets (fractions, e.g. 0.1 0.2).",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    _invoke_prune(config_path, args.targets)


if __name__ == "__main__":
    main()
