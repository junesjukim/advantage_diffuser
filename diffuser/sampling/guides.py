import torch
import torch.nn as nn


class ValueGuide(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad


class AdvantageGuide(ValueGuide):
    """Advantage diffusion에서 학습된 scalar advantage 값을 그대로 이용하는 가이드.
    ValueGuide 와 동일한 동작을 하지만, 코드 가독성을 위해 별도 클래스로 노출한다."""
    pass
