from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_dataloaders(
    data_path: Path,
    batch_size: int,
    num_workers: int = 4,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    rhf: float = 0.5,
    cj: List[float] = [0.2, 0.2, 0.2, 0.0],
    re: List[float] = [0.25, 0.02, 0.12, 0.3, 3.3],
    eval_only: bool = False, # used for evaluation
):
    """Create train and val DataLoaders for the dataset.
    Assumes data_path has 'train' and 'val' subfolders, each with '0' and '1' subfolders.
    """

    val_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_ds = datasets.ImageFolder(data_path / "val", transform=val_tf)

    # No shuffle as we want determinism for val
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if eval_only:
        return val_loader

    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=rhf),
            transforms.ColorJitter(brightness=cj[0], contrast=cj[1], saturation=cj[2], hue=cj[3]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=re[0], scale=(re[1], re[2]), ratio=(re[3], re[4])),
        ]
    )

    train_ds = datasets.ImageFolder(data_path / "train", transform=train_tf)
    
    # Confirm expected mapping: folder "0" -> class 0, "1" -> class 1
    expected_mapping = {
        "0": 0,
        "1": 1,
    }
    if train_ds.class_to_idx != expected_mapping:
        raise ValueError(
            f"Expected class mapping {expected_mapping} under {data_path / 'train'}, "
            f"but found {train_ds.class_to_idx}."
        )

    tr_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return tr_loader, val_loader
