import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)


def build_model(model_type, pretrained=False):
    """Build a model based on the specified architecture"""

    if model_type == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model

    elif model_type == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model

    else:
        raise ValueError(f"Unsupported architecture: {model_type}")
