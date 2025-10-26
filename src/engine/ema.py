import copy
import torch


class ModelEMA:
    """Maintain an exponential moving average (EMA) copy of a model."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key]
            if not torch.is_floating_point(model_value):
                ema_value.copy_(model_value)
            else:
                ema_value.copy_(ema_value * self.decay + (1.0 - self.decay) * model_value)

    def to(self, device):
        self.ema_model.to(device)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)
