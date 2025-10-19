from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_dataloaders(
    data_path: Path,
    batch_size: int,
    num_workers: int,
    mean: List[int],
    std: List[int],
    rhf: float,
    cj: List[float],
    re: List[float],
):
    """Create train and val DataLoaders for the dataset.
    Assumes data_path has 'train' and 'val' subfolders, each with '0' and '1' subfolders.
    """

    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=rhf),
            transforms.ColorJitter(brightness=cj[0], contrast=cj[1], saturation=cj[2], hue=cj[3]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=re[0], scale=(re[1], re[2]), ratio=(re[3], re[4])),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.ImageFolder(data_path / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_path / "val", transform=val_tf)

    # Confirm expected mapping: folder "0" -> class 0, "1" -> class 1
    assert train_ds.class_to_idx == {
        "0": 0,
        "1": 1,
    }, f"class mapping is {train_ds.class_to_idx}"

    tr_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Loss weighting alternative (if not using sampler)
    """
    labels = [y for _, y in train_ds.samples]
    counts = np.bincount(labels, minlength=2)
    total = float(sum(counts))
    # Balanced-ish: N/(2*n_c) so average weight ~1
    class_weight_tensor = torch.tensor([total/(2*counts[0]+1e-9), total/(2*counts[1]+1e-9)], dtype=torch.float32)
    """

    class_weight_tensor = None  # data is near balanced, so don’t weight

    # No shuffle as we want determininism for val
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return tr_loader, val_loader, class_weight_tensor
