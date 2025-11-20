"""
Summarise JPEG dimensions in the COCO train2017 and val2017 folders

Usage: python scripts/get_dataset_stats.py
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, List, Tuple


def jpeg_size(path: Path) -> Tuple[int, int]:
    """Read width/height from a JPEG without third-party libraries."""
    with path.open("rb") as handle:
        if handle.read(2) != b"\xFF\xD8":
            raise ValueError("Not a JPEG file.")
        while True:
            marker_start = handle.read(1)
            if marker_start == b"":
                break
            if marker_start != b"\xFF":
                continue
            marker = handle.read(1)
            while marker == b"\xFF":
                marker = handle.read(1)
            if marker in {b"\xD8", b"\xD9"}:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", length_bytes)[0]
            if marker in {b"\xC0", b"\xC1", b"\xC2", b"\xC3", b"\xC5", b"\xC6", b"\xC7", b"\xC9", b"\xCA", b"\xCB", b"\xCD", b"\xCE", b"\xCF"}:
                data = handle.read(5)
                if len(data) != 5:
                    break
                height, width = struct.unpack(">HH", data[1:5])
                return width, height
            handle.seek(segment_length - 2, 1)
    raise ValueError(f"Could not determine JPEG size for {path}.")


def iter_jpegs(paths: Iterable[Path]) -> Iterable[Path]:
    for directory in paths:
        if not directory.exists():
            print(f"Skipping {directory}: missing directory.")
            continue
        if not directory.is_dir():
            print(f"Skipping {directory}: not a directory.")
            continue
        yield from directory.rglob("*.jpg")
        yield from directory.rglob("*.jpeg")


def collect_dimensions(paths: Iterable[Path]) -> Tuple[List[int], List[int]]:
    widths, heights = [], []
    for image_path in paths:
        try:
            width, height = jpeg_size(image_path)
        except Exception as exc:
            print(f"Skipping {image_path}: {exc}")
            continue
        widths.append((width, image_path))
        heights.append((height, image_path))
    if not widths:
        raise ValueError("No valid JPEG images found.")
    return widths, heights


def summarise(values_with_paths: List[Tuple[int, Path]]) -> dict:
    values = [value for value, _ in values_with_paths]
    (min_value, min_path) = min(values_with_paths, key=lambda item: item[0])
    (max_value, max_path) = max(values_with_paths, key=lambda item: item[0])

    stats = {
        "min": min_value,
        "min_path": min_path,
        "max": max_value,
        "max_path": max_path,
        "mean": mean(values),
    }
    stats["std"] = pstdev(values)

    std = stats["std"]
    min_band = stats["min"] + std
    max_band = stats["max"] - std

    stats["near_min"] = sum(value <= min_band for value in values)
    stats["near_mean"] = sum(abs(value - stats["mean"]) <= std for value in values)
    stats["near_max"] = sum(value >= max_band for value in values)
    stats["count"] = len(values)
    return stats


def format_portion(count: int, total: int) -> str:
    return f"{count} image(s) ({(count / total) * 100:.2f}%)"


def print_summary(label: str, stats: dict) -> None:
    total = stats["count"]
    print(f"\n{label}:")
    print(f"  Min: {stats['min']}px ({stats['min_path']})")
    print(f"  Max: {stats['max']}px ({stats['max_path']})")
    print(f"  Mean: {stats['mean']:.2f}px")
    print(f"  Std dev: {stats['std']:.2f}px")
    print(f"  Near minimum (≤ min + std): {format_portion(stats['near_min'], total)}")
    print(f"  Near mean (within ±std): {format_portion(stats['near_mean'], total)}")
    print(f"  Near maximum (≥ max - std): {format_portion(stats['near_max'], total)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folders",
        nargs="*",
        type=Path,
        help="Optional list of folders to scan. Defaults to data/coco/train2017 and data/coco/val2017.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    default_dirs = [repo_root / "data" / "coco" / "train2017", repo_root / "data" / "coco" / "val2017"]
    target_dirs = args.folders if args.folders else default_dirs

    widths, heights = collect_dimensions(iter_jpegs(target_dirs))
    width_stats = summarise(widths)
    height_stats = summarise(heights)

    print(f"Processed {width_stats['count']} image(s).")
    print_summary("Width", width_stats)
    print_summary("Height", height_stats)


if __name__ == "__main__":
    main()
