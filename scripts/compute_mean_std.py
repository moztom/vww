# Computes the mean and std of the VWW96 training set.
#
# Run from inside scripts/ directory.
# Usage: python compute_mean_std.py

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from pathlib import Path

def compute_mean_std(data_root, batch=256, num_workers=4):
    # Use only ToTensor so you measure raw distribution (no flips/jitter)
    tmp_tf = transforms.ToTensor()
    tmp_ds = datasets.ImageFolder(Path(data_root) / "train", transform=tmp_tf)
    tmp_loader = DataLoader(tmp_ds, batch_size=batch, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    n = 0
    mean = torch.zeros(3)
    M2 = torch.zeros(3)  # sum of squared diffs for Welford’s
    for imgs, _ in tmp_loader:
        # imgs: [B, 3, H, W] in [0,1]
        b = imgs.size(0)
        imgs = imgs.view(b, 3, -1)  # flatten spatial
        batch_mean = imgs.mean(dim=(0,2))         # [3]
        batch_var  = imgs.var(dim=(0,2), unbiased=False)  # [3]
        if n == 0:
            mean = batch_mean
            M2 = batch_var
            n = b
        else:
            # combine means/vars across batches
            delta = batch_mean - mean
            tot = n + b
            mean = mean + delta * (b / tot)
            M2 = (n*M2 + b*batch_var) / tot + (n*b/(tot**2)) * (delta**2)
            n = tot
    std = torch.sqrt(M2)
    return mean.tolist(), std.tolist()

def main() -> None:
    mean, std = compute_mean_std("../data/vww96")
    print("MEAN =", mean, "STD =", std)

if __name__ == "__main__":
    main()
