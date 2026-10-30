"""P2 benchmark suite: repeated-seed controlled comparisons and optional ablations."""
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
from .layers import OPERATOR_NAMES
from .train import run_training

def dataset_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def aggregate(runs):
    grouped = {}
    for run in runs:
        grouped.setdefault(run["operator"], []).append(run)
    results = []
    for operator, entries in grouped.items():
        f1 = [float(r["test_macro_f1"]) for r in entries]
        acc = [float(r["test_accuracy"]) for r in entries]
        elapsed = [float(r["elapsed_seconds"]) for r in entries]
        results.append({
            "operator": operator, "n_seeds": len(entries),
            "macro_f1_mean": statistics.mean(f1),
            "macro_f1_std": statistics.stdev(f1) if len(f1)>1 else 0.0,
            "accuracy_mean": statistics.mean(acc),
            "accuracy_std": statistics.stdev(acc) if len(acc)>1 else 0.0,
            "elapsed_mean_seconds": statistics.mean(elapsed),
            "parameter_count": entries[0]["parameter_count"],
        })
    return sorted(results, key=lambda r: (-r["macro_f1_mean"], r["operator"]))

def run_benchmark(dataset="storage/dataset.jsonl", operators=None, seeds=(42, 43, 44),
                  epochs=5, batch_size=32, output_dir="storage/benchmarks", max_length=64):
    operators = list(operators) if operators is not None else ["baseline", *OPERATOR_NAMES]
    if not operators or len(set(operators)) != len(operators):
        raise ValueError("Choose distinct operators")
    if any(op != "baseline" and op not in OPERATOR_NAMES for op in operators):
        raise ValueError("Unknown operator")
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct seeds")
    if epochs < 1 or batch_size < 1:
        raise ValueError("Invalid training budget")
    digest = dataset_sha256(dataset)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    benchmark_id = f"benchmark-{int(time.time()*1000)}"
    run_dir = root / benchmark_id
    run_dir.mkdir(parents=True)
    runs = []
    manifest = {
        "benchmark_id": benchmark_id, "dataset_sha256": digest, "operators": operators,
        "seeds": list(seeds), "epochs": epochs, "batch_size": batch_size,
        "max_length": max_length,
        "note": "Identical epoch budgets, not equal parameter counts or FLOPs. Test scores are descriptive; choose models on validation in future preregistered evaluation.",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for operator in operators:
        for seed in seeds:
            start = time.perf_counter()
            result = run_training(
                dataset=dataset, operator=operator, epochs=epochs, seed=seed,
                batch_size=batch_size, max_length=max_length,
                output_dir=str(run_dir / "checkpoints"))
            result["elapsed_seconds"] = time.perf_counter() - start
            result["dataset_sha256"] = digest
            # The train function's metadata contains model configuration; compute count from checkpoint.
            import torch
            from .train import TextClassifier
            checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
            model = TextClassifier(len(checkpoint["vocab"]), len(checkpoint["labels"]), operator=operator)
            result["parameter_count"] = sum(p.numel() for p in model.parameters())
            runs.append(result)
            (run_dir / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    summary = aggregate(runs)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary:
        with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    return {"directory": str(run_dir), "manifest": manifest, "summary": summary, "runs": runs}
