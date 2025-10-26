from typing import Optional

import torch
from torch.nn.modules.batchnorm import _BatchNorm


def set_backbone_trainable(model, requires_grad: bool):
    """Enable/disable gradients for the backbone (all but classifier)."""

    handled = False

    if hasattr(model, "features"):
        for param in model.features.parameters():
            param.requires_grad = requires_grad
        handled = True

    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = requires_grad
        handled = True

    if not handled:
        classifier_ids = set()
        if hasattr(model, "classifier"):
            classifier_ids = {id(p) for p in model.classifier.parameters()}
        for param in model.parameters():
            if classifier_ids and id(param) in classifier_ids:
                continue
            param.requires_grad = requires_grad

    # Always keep classifier/train head trainable
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True


@torch.no_grad()
def recalibrate_batch_norm(model, loader, device, max_batches: Optional[int] = None):
    """Refresh BatchNorm running stats by running data through the model."""

    was_training = model.training
    model.train()

    momenta = {}
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            momenta[module] = module.momentum
            module.reset_running_stats()
            module.momentum = None

    nonblock = torch.cuda.is_available()
    for idx, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=nonblock)
        model(imgs)
        if max_batches and (idx + 1) >= max_batches:
            break

    for module, momentum in momenta.items():
        module.momentum = momentum

    if not was_training:
        model.eval()
