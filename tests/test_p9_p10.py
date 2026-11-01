import pytest
import torch
from semantic_liquidity.coupling import CoupledSemanticLayer
from semantic_liquidity.semantic_graph import SemanticGraphLayer, graph_data


@pytest.mark.parametrize("layer", [CoupledSemanticLayer(dim=12, prototypes=5),
                                    SemanticGraphLayer(dim=12, prototypes=5, top_k=2)])
def test_forward_backward(layer):
    x = torch.randn(2, 4, 12, requires_grad=True)
    output = layer(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert any(p.grad is not None for p in layer.parameters())


def test_coupling_bounds():
    layer = CoupledSemanticLayer(dim=8, prototypes=4)
    y, d = layer(torch.randn(2, 3, 8), return_diagnostics=True)
    assert torch.allclose(d["probabilities"].sum(-1), torch.ones(2, 3), atol=1e-5)
    assert d["entropy"].min() >= 0
    assert d["entropy"].max() <= 1.00001
    assert d["temperature"].min() >= layer.min_temperature
    assert d["temperature"].max() <= layer.max_temperature
    assert layer.regularization(d).item() >= 0


def test_graph_normalized_and_exportable():
    layer = SemanticGraphLayer(dim=8, prototypes=5, top_k=2)
    adj = layer.adjacency()
    assert torch.isfinite(adj).all()
    assert torch.allclose(adj.sum(-1), torch.ones(5), atol=1e-5)
    graph = graph_data(layer)
    assert len(graph["nodes"]) == 5
    assert all(e["source"] != e["target"] for e in graph["edges"])


def test_invalid_configs():
    with pytest.raises(ValueError):
        SemanticGraphLayer(prototypes=3, top_k=3)
    with pytest.raises(ValueError):
        CoupledSemanticLayer(entropy_target=2)
