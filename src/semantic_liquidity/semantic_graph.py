"""P10: learnable semantic prototype graph with normalized message passing."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SemanticGraphLayer(nn.Module):
    def __init__(self, dim=64, prototypes=8, top_k=3, steps=2, self_loops=True):
        super().__init__()
        if dim < 1 or prototypes < 2 or not 1 <= top_k < prototypes or steps < 1:
            raise ValueError("Invalid graph configuration")
        self.memory = nn.Parameter(torch.randn(prototypes, dim) * 0.05)
        self.message = nn.Linear(dim, dim, bias=False)
        self.project = nn.Linear(dim, dim)
        self.gate = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.top_k = top_k
        self.steps = steps
        self.self_loops = self_loops

    def adjacency(self):
        normalized = F.normalize(self.memory, dim=-1, eps=1e-8)
        scores = normalized @ normalized.T
        n = scores.shape[0]
        scores = scores.masked_fill(torch.eye(n, dtype=torch.bool, device=scores.device), -1e4)
        idx = scores.topk(self.top_k, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)
        mask = mask | mask.T
        if self.self_loops:
            mask.fill_diagonal_(True)
            weights = (scores.masked_fill(~mask, -1e4)).clone()
            weights.diagonal().fill_(1.0)
        else:
            weights = scores.masked_fill(~mask, -1e4)
        return torch.softmax(weights, dim=-1)

    def forward(self, x, return_diagnostics=False):
        if x.ndim != 3 or x.shape[-1] != self.memory.shape[-1]:
            raise ValueError("Expected [batch, tokens, dim]")
        adj = self.adjacency()
        nodes = self.memory
        for _ in range(self.steps):
            nodes = F.layer_norm(nodes + F.gelu(self.message(adj @ nodes)), (nodes.shape[-1],))
        assignment = torch.softmax(x @ nodes.T / math.sqrt(x.shape[-1]), dim=-1)
        context = assignment @ nodes
        gate = torch.sigmoid(self.gate(torch.cat([x, context], dim=-1)))
        output = self.norm(x + gate * self.project(context))
        if return_diagnostics:
            return output, {"adjacency": adj, "assignments": assignment,
                            "nodes": nodes}
        return output


def graph_data(layer, threshold=0.0):
    """JSON-friendly graph for Plotly/Streamlit, with edge weights."""
    with torch.no_grad():
        adj = layer.adjacency().detach().cpu()
        n = adj.shape[0]
        nodes = [{"id": i, "norm": float(layer.memory[i].norm().item())} for i in range(n)]
        edges = [{"source": i, "target": j, "weight": float(adj[i, j])}
                 for i in range(n) for j in range(n) if i != j and float(adj[i, j]) > threshold]
    return {"nodes": nodes, "edges": edges, "adjacency": adj.tolist()}
