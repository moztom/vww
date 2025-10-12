import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

# Computed mean and std of VWW96 training set
TRAIN_MEAN = (0.4699, 0.4469, 0.4077)
TRAIN_STD  = (0.2653, 0.2618, 0.2768)

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor(std).view(1,3,1,1))
    def forward(self, x):  # expects x in [0,1]
        return (x - self.mean) / self.std

class ModelWithNorm(nn.Module):
    def __init__(self, backbone: nn.Module, mean=None, std=None):
        super().__init__()
        self.norm = Normalize(mean, std) if (mean and std) else nn.Identity()
        self.backbone = backbone
    def forward(self, x):
        x = self.norm(x)
        return self.backbone(x)

def build_model(num_classes=2, mean=TRAIN_MEAN, std=TRAIN_STD):
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes) # 2 class final layer
    return ModelWithNorm(model, mean=mean, std=std)
