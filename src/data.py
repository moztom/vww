from pathlib import Path
from typing import List, Optional
import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


# NOTE: torch.utils.data.DistributedSampler is used for KD to synchronize ordering
# between student/teacher streams even in single-process training. We explicitly
# set num_replicas=1 and rank=0 so no distributed backend is required.


def build_dataloaders(
    data_path: Path,
    batch_size: int,
    num_workers: int = 4,
    mean: List[int] = [0.485, 0.456, 0.406],
    std: List[int] = [0.229, 0.224, 0.225],
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

    # No shuffle as we want determininism for val
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
    assert train_ds.class_to_idx == {
        "0": 0,
        "1": 1,
    }, f"class mapping is {train_ds.class_to_idx}"

    """
    # Sampler to address slightly inbalanced batches
    targets = np.array(train_ds.targets)
    class_counts = np.bincount(targets)
    class_weights = (1.0 / class_counts)
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True
    )
    """

    tr_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        #sampler=sampler,
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

    return tr_loader, val_loader, class_weight_tensor


def build_kd_dataloaders(
    data_path: Path,
    batch_size: int,
    num_workers: int = 4,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    rhf: float = 0.5,
    cj: List[float] = [0.2, 0.2, 0.2, 0.0],
    re: List[float] = [0.25, 0.02, 0.12, 0.3, 3.3],
    teacher_view: str = "student",
):
    """Create KD DataLoaders (student + optional teacher stream + val).

    Args:
        teacher_view: "student" (teacher sees the same augmented tensors) or
                      "clean" (teacher sees non-augmented validation transforms).
    """

    teacher_view = teacher_view.lower()
    if teacher_view not in {"student", "clean"}:
        raise ValueError(f"Unsupported teacher_view '{teacher_view}', expected 'student' or 'clean'.")

    val_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=rhf),
            transforms.ColorJitter(brightness=cj[0], contrast=cj[1], saturation=cj[2], hue=cj[3]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=re[0], scale=(re[1], re[2]), ratio=(re[3], re[4])),
        ]
    )

    student_train_ds = datasets.ImageFolder(data_path / "train", transform=train_tf)
    assert student_train_ds.class_to_idx == {"0": 0, "1": 1}, f"class mapping is {student_train_ds.class_to_idx}"

    teacher_train_ds: Optional[datasets.ImageFolder] = None
    if teacher_view == "clean":
        teacher_train_ds = datasets.ImageFolder(data_path / "train", transform=val_tf)
        assert teacher_train_ds.class_to_idx == {"0": 0, "1": 1}, f"class mapping is {teacher_train_ds.class_to_idx}"
        assert len(student_train_ds) == len(teacher_train_ds)

    # Student loader -------------------------------------------------------
    if teacher_view == "clean":
        student_sampler: Optional[DistributedSampler] = DistributedSampler(
            student_train_ds,
            num_replicas=1,
            rank=0,
            shuffle=True,
            drop_last=False,
        )
        tr_loader_student = DataLoader(
            student_train_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=student_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        tr_loader_student = DataLoader(
            student_train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # Teacher loader (only needed for clean view) --------------------------
    if teacher_view == "clean" and teacher_train_ds is not None:
        teacher_sampler = DistributedSampler(
            teacher_train_ds,
            num_replicas=1,
            rank=0,
            shuffle=True,
            drop_last=False,
        )
        tr_loader_teacher: Optional[DataLoader] = DataLoader(
            teacher_train_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=teacher_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        tr_loader_teacher = None

    # Validation loader ----------------------------------------------------
    val_ds = datasets.ImageFolder(data_path / "val", transform=val_tf)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    class_weight_tensor = None

    return tr_loader_student, tr_loader_teacher, val_loader, class_weight_tensor
