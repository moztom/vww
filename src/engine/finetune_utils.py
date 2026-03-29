from typing import Optional

import torch
from torch.nn.modules.batchnorm import _BatchNorm


@torch.no_grad()
def recalibrate_batch_norm(model, loader, device, max_batches: Optional[int] = None):
    """Recal BatchNorm stats by running data through the model"""

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
