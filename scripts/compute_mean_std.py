"""
Compute mean/std for a dataset split.

Example usage:
  python scripts/compute_mean_std.py
  python scripts/compute_mean_std.py --dataset-path data/vww224 --split train
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def compute_mean_std(
    split_dir: Path,
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[list[float], list[float], int]:
    """Compute RGB mean/std over all pixels in an ImageFolder split"""

    dataset = datasets.ImageFolder(split_dir, transform=transforms.ToTensor())
    if len(dataset) == 0:
        raise ValueError(f"No images found under: {split_dir}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    for images, _ in loader:
        images = images.to(torch.float64)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images * images).sum(dim=(0, 2, 3))
        total_pixels += images.size(0) * images.size(2) * images.size(3)

    mean = channel_sum / total_pixels
    variance = (channel_squared_sum / total_pixels) - (mean * mean)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    return mean.tolist(), std.tolist(), len(dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=repo_root / "data" / "vww96",
        help="Dataset root containing train/ and val/ folders",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Dataset split to scan",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="DataLoader batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    args = parser.parse_args()

    split_dir = args.dataset_root / args.split
    mean, std, image_count = compute_mean_std(
        split_dir=split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Dataset split: {split_dir}")
    print(f"Images: {image_count}")
    print("MEAN =", [round(x, 6) for x in mean])
    print("STD  =", [round(x, 6) for x in std])


if __name__ == "__main__":
    main()
