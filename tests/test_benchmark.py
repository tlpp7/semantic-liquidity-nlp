import json
import pytest
from semantic_liquidity.benchmark import aggregate, dataset_sha256, run_benchmark

def test_aggregate_ranking():
    runs = [
        {"operator":"baseline","test_macro_f1":0.4,"test_accuracy":0.5,"elapsed_seconds":2,"parameter_count":10},
        {"operator":"sll","test_macro_f1":0.8,"test_accuracy":0.9,"elapsed_seconds":3,"parameter_count":20},
        {"operator":"sll","test_macro_f1":0.6,"test_accuracy":0.7,"elapsed_seconds":5,"parameter_count":20},
    ]
    summary = aggregate(runs)
    assert summary[0]["operator"] == "sll"
    assert summary[0]["n_seeds"] == 2
    assert summary[0]["macro_f1_mean"] == pytest.approx(0.7)

def test_dataset_hash(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"text":"hello","label":"a"}\n')
    assert len(dataset_sha256(p)) == 64

def test_reject_invalid_operator(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("x")
    with pytest.raises(ValueError):
        run_benchmark(dataset=p, operators=["not_an_operator"], seeds=[42], output_dir=tmp_path / "results")

def test_benchmark_smoke(tmp_path):
    p = tmp_path / "x.jsonl"
    with p.open("w") as f:
        for i in range(40):
            f.write(json.dumps({"text": ("river stream " if i%2 else "money finance ") + str(i), "label":str(i%2)})+"\n")
    result = run_benchmark(dataset=p, operators=["baseline", "softmax_beta"], seeds=[42],
                           epochs=1, batch_size=16, output_dir=tmp_path / "results")
    assert len(result["summary"]) == 2
    assert (tmp_path / "results" / result["manifest"]["benchmark_id"] / "summary.csv").exists()
