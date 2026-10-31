"""P3 diagnostic views of prototype probabilities and entropy."""
import math
import torch
from torch.nn import functional as F

@torch.no_grad()
def inspect_layer(layer, embeddings):
    if embeddings.ndim != 3:
        raise ValueError("Expected [batch, tokens, dim]")
    scores = embeddings @ layer.memory.T / math.sqrt(embeddings.shape[-1])
    probabilities = torch.softmax(layer.beta * scores, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1) / math.log(probabilities.shape[-1])
    return {"probabilities": probabilities.cpu().tolist(),
            "entropy": entropy.cpu().tolist(),
            "prototype_norms": layer.memory.norm(dim=-1).cpu().tolist()}
