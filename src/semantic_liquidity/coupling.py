"""P9: differentiable coupled entropy/plasticity prototype routing.

This is a stateless-inference neural layer: 'plasticity' denotes a learned
update *gate*, not a mutation of parameters during forward passes.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CoupledSemanticLayer(nn.Module):
    def __init__(self, dim=64, prototypes=8, entropy_target=0.5,
                 coupling_strength=0.5, min_temperature=0.1, max_temperature=5.0):
        super().__init__()
        if dim < 1 or prototypes < 2 or not 0 <= entropy_target <= 1:
            raise ValueError("Invalid dimensions or entropy target")
        if not 0 <= coupling_strength <= 1 or not 0 < min_temperature < max_temperature:
            raise ValueError("Invalid coupling or temperature bounds")
        self.memory = nn.Parameter(torch.randn(prototypes, dim) * 0.02)
        self.temperature_net = nn.Linear(dim + 1, 1)
        self.plasticity_net = nn.Linear(dim + 2, 1)
        self.gate = nn.Linear(dim * 2 + 1, dim)
        self.project = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.entropy_target = entropy_target
        self.coupling_strength = coupling_strength
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

    def forward(self, x, return_diagnostics=False):
        if x.ndim != 3 or x.shape[-1] != self.memory.shape[-1]:
            raise ValueError("Expected [batch, tokens, dim]")
        logits = x @ self.memory.T / math.sqrt(x.shape[-1])
        initial_p = torch.softmax(logits, -1)
        initial_h = -(initial_p * initial_p.clamp_min(1e-9).log()).sum(-1, keepdim=True) / math.log(self.memory.shape[0])
        temp = self.min_temperature + (self.max_temperature - self.min_temperature) * torch.sigmoid(
            self.temperature_net(torch.cat([x, initial_h], -1)))
        p = torch.softmax(logits / temp, -1)
        entropy = -(p * p.clamp_min(1e-9).log()).sum(-1, keepdim=True) / math.log(self.memory.shape[0])
        context = p @ self.memory
        error = (x - context).square().mean(-1, keepdim=True)
        plasticity = torch.sigmoid(self.plasticity_net(torch.cat([x, entropy, error], -1)))
        stability = torch.exp(-(entropy - self.entropy_target).square())
        weight = (1 - self.coupling_strength) + self.coupling_strength * plasticity * stability
        gate = torch.sigmoid(self.gate(torch.cat([x, context, entropy], -1)))
        y = self.norm(x + gate * weight * self.project(context))
        if return_diagnostics:
            return y, {"probabilities": p, "entropy": entropy, "temperature": temp,
                       "plasticity": plasticity, "stability": stability,
                       "weight": weight}
        return y

    def regularization(self, diagnostics):
        """Optional penalty: target normalized entropy, not automatically added to training."""
        return (diagnostics["entropy"] - self.entropy_target).square().mean()
