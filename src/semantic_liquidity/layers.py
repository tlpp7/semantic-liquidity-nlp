"""30 differentiable semantic operators. Experimental mathematical proxies, not neuroscience models."""
import math
import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = {
    "activation": ("softmax_beta", "sigmoid_gate", "cosine_route", "topk_route", "adaptive_temperature"),
    "uncertainty": ("shannon_gate", "binary_entropy", "surprisal_gate", "confidence_gate", "entropy_target"),
    "plasticity": ("hebbian_proxy", "oja_proxy", "stdp_positive", "stdp_negative", "eligibility_trace"),
    "flexibility": ("quality_quadratic", "stability_gate", "ambiguity_penalty", "cost_gate", "adaptive_beta"),
    "consolidation": ("fast_saturation", "slow_saturation", "dual_timescale", "forgetting_decay", "reactivation"),
    "coupling": ("entropy_plasticity", "surprise_consolidation", "quality_router", "prediction_error", "coupled_dynamics"),
}
OPERATOR_NAMES = tuple(name for names in FAMILIES.values() for name in names)

class SemanticLiquidityLayer(nn.Module):
    """Contextual prototype routing with residual output, shape [batch, seq, dim]."""

    def __init__(self, dim=64, prototypes=8, operator="softmax_beta", beta=1.0):
        super().__init__()
        if operator not in OPERATOR_NAMES:
            raise ValueError(f"Unknown operator: {operator}")
        if dim < 1 or prototypes < 2:
            raise ValueError("dim must be positive and prototypes >= 2")
        self.operator = operator
        self.beta = beta
        self.memory = nn.Parameter(torch.randn(prototypes, dim) * 0.02)
        self.project = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 2 + 1, dim)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("steps", torch.zeros(()))

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.memory.shape[-1]:
            raise ValueError("Expected [batch, tokens, dim]")
        eps = 1e-8
        scores = x @ self.memory.T / math.sqrt(x.shape[-1])
        probs = torch.softmax(self.beta * scores, dim=-1)
        entropy = -(probs * probs.clamp_min(eps).log()).sum(-1, keepdim=True) / math.log(probs.shape[-1])
        surprise = -probs.max(-1, keepdim=True).values.clamp_min(eps).log()
        context = probs @ self.memory
        cosine = F.cosine_similarity(x, context, dim=-1).unsqueeze(-1)
        variance = probs.var(dim=-1, keepdim=True, unbiased=False)
        quality = (1 + entropy - entropy.square()).clamp(0, 1.25)
        n = self.operator
        weight = torch.ones_like(entropy)
        if n == "sigmoid_gate": weight = torch.sigmoid(cosine)
        elif n == "cosine_route": weight = (cosine + 1) / 2
        elif n == "topk_route":
            top = scores.topk(min(2, scores.shape[-1]), dim=-1).indices
            mask = torch.zeros_like(scores).scatter_(-1, top, 1.)
            p = torch.softmax(scores.masked_fill(mask == 0, -1e4), -1)
            context = p @ self.memory
        elif n == "adaptive_temperature": weight = torch.sigmoid(2 * (1 - entropy))
        elif n == "shannon_gate": weight = 1 - entropy
        elif n == "binary_entropy":
            p = probs.max(-1, keepdim=True).values.clamp(eps, 1-eps)
            weight = 1 + (p*p.log() + (1-p)*(1-p).log()) / math.log(2)
        elif n == "surprisal_gate": weight = torch.sigmoid(surprise)
        elif n == "confidence_gate": weight = probs.max(-1, keepdim=True).values
        elif n == "entropy_target": weight = torch.exp(-(entropy - 0.5).square())
        elif n == "hebbian_proxy": weight = torch.sigmoid((x * context).mean(-1, keepdim=True))
        elif n == "oja_proxy": weight = torch.sigmoid(cosine - context.square().mean(-1, keepdim=True))
        elif n == "stdp_positive": weight = torch.exp(-torch.abs(cosine) / 0.2)
        elif n == "stdp_negative": weight = -0.8 * torch.exp(-torch.abs(cosine) / 0.2)
        elif n == "eligibility_trace": weight = torch.sigmoid(cosine) * (1 - entropy)
        elif n == "quality_quadratic": weight = quality
        elif n == "stability_gate": weight = 1 - entropy
        elif n == "ambiguity_penalty": weight = 1 - entropy.square()
        elif n == "cost_gate": weight = 1 / (1 + variance)
        elif n == "adaptive_beta": weight = torch.sigmoid(self.beta * (1 - entropy))
        elif n == "fast_saturation": weight = 1 - torch.exp(-0.8 * (1 + surprise))
        elif n == "slow_saturation": weight = 1 - torch.exp(-0.2 * (1 + surprise))
        elif n == "dual_timescale": weight = 0.5 * (2 - torch.exp(-0.8*(1+surprise)) - torch.exp(-0.2*(1+surprise)))
        elif n == "forgetting_decay": weight = torch.exp(-0.2 * surprise)
        elif n == "reactivation": weight = torch.sigmoid(cosine + surprise)
        elif n == "entropy_plasticity": weight = (1-entropy) * torch.sigmoid(cosine)
        elif n == "surprise_consolidation": weight = torch.sigmoid(surprise) * (1-torch.exp(-0.2*(1+surprise)))
        elif n == "quality_router": weight = quality * probs.max(-1, keepdim=True).values
        elif n == "prediction_error": weight = torch.sigmoid((x-context).square().mean(-1, keepdim=True))
        elif n == "coupled_dynamics": weight = quality * torch.sigmoid(cosine+surprise) * (1-0.5*entropy)
        gate = torch.sigmoid(self.gate(torch.cat((x, context, entropy), -1)))
        return self.norm(x + gate * weight * self.project(context))
