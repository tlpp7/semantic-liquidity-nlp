import json
import torch
from semantic_liquidity.layers import SemanticLiquidityLayer
from semantic_liquidity.semantic_analysis import inspect_layer
from semantic_liquidity.continual import make_tasks, run_continual, ewc_penalty
from semantic_liquidity.train import TextClassifier

def test_inspection_probabilities():
    layer = SemanticLiquidityLayer(dim=8, prototypes=4)
    out = inspect_layer(layer, torch.randn(2, 3, 8))
    assert len(out["probabilities"]) == 2
    assert abs(sum(out["probabilities"][0][0]) - 1) < 1e-5
    assert all(0 <= v <= 1.00001 for row in out["entropy"] for v in row)

def test_empty_ewc_penalty():
    model = TextClassifier(12, 2, dim=8)
    assert ewc_penalty(model, []).item() == 0

def test_sequential_smoke(tmp_path):
    records = [{"text": ("river water" if i%2 else "money bank") + f" {i}",
                "label":str(i%2)} for i in range(24)]
    result = run_continual(records, operator="baseline", n_tasks=2, epochs=1,
                           seed=3, output_dir=tmp_path)
    assert len(result["accuracy_matrix"]) == 2
    assert len(result["accuracy_matrix"][0]) == 2
    saved = tmp_path / "continual-baseline-3-0.0-replay0-0.json"
    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8")) == result
