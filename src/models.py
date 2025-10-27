import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    mobilenet_v3_large,
    efficientnet_b2,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    EfficientNet_B2_Weights,
)


def build_model(model_type, pretrained=False):
    """Build a model based on the specified architecture"""

    if model_type == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model

    elif model_type == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model

    elif model_type == "efficientnet_b2":
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = efficientnet_b2(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model

    else:
        raise ValueError(f"Unsupported architecture: {model_type}")
