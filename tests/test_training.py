import json
import pytest
from semantic_liquidity.train import build_vocab, encode, split_records, run_training

def test_no_validation_leakage():
    rows = [{"text": f"sample {i}", "label": str(i % 2)} for i in range(100)]
    tr, va, te = split_records(rows, seed=7)
    assert len(tr) == 80 and len(va) == 10 and len(te) == 10
    assert not (set(map(id, tr)) & set(map(id, va)))
    vocab = build_vocab(tr)
    assert "sample" in vocab
    assert encode(va, vocab, {"0": 0, "1": 1})[0].shape == (10, 64)

def test_training_checkpoint(tmp_path):
    path = tmp_path / "data.jsonl"
    with path.open("w") as f:
        for i in range(40):
            f.write(json.dumps({"text": ("river stream water " if i % 2 else "money finance loan ") + str(i), "label": str(i%2)})+"\n")
    result = run_training(str(path), "baseline", epochs=1, output_dir=str(tmp_path / "runs"))
    assert 0 <= result["test_accuracy"] <= 1
    assert result["test_size"] > 0
    assert (tmp_path / "runs" / (result["run_id"] + ".pt")).exists()
