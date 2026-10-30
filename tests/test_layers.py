import pytest
import torch
from semantic_liquidity import OPERATOR_NAMES, SemanticLiquidityLayer

@pytest.mark.parametrize("name", OPERATOR_NAMES)
def test_operator_forward_backward(name):
    torch.manual_seed(1)
    x = torch.randn(2, 4, 12, requires_grad=True)
    model = SemanticLiquidityLayer(dim=12, prototypes=5, operator=name)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_operator_count():
    assert len(OPERATOR_NAMES) == 30
    assert len(set(OPERATOR_NAMES)) == 30
