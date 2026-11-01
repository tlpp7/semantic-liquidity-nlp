import json
import torch
from semantic_liquidity.observability import TelemetryWriter, gradient_metrics, read_events, resource_snapshot


def test_gradient_metrics():
    model = torch.nn.Linear(3, 2)
    model(torch.ones(4, 3)).square().mean().backward()
    stats = gradient_metrics(model)
    assert stats["grad_norm"] > 0
    assert stats["grad_elements"] == 8
    assert stats["grad_nonfinite"] == 0


def test_telemetry_jsonl(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path, "test-123")
    writer.log("epoch", epoch=1, train_loss=0.5)
    writer.log("complete", accuracy=0.75)
    rows = read_events(path)
    assert [row["event"] for row in rows] == ["epoch", "complete"]
    assert rows[0]["run_id"] == "test-123"
    assert rows[0]["elapsed_seconds"] >= 0


def test_resources_have_nullable_fields():
    stats = resource_snapshot()
    assert "cpu_rss_bytes" in stats
    assert "cuda_allocated_bytes" in stats
