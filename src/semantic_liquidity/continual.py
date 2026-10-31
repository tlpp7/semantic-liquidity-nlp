"""P3: sequential-task evaluation with optional EWC stability penalty."""
import copy
import json
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from .train import TextClassifier, build_vocab, encode, scores

def make_tasks(records, n_tasks=3, seed=42):
    if n_tasks < 2:
        raise ValueError("n_tasks must be >= 2")
    rows = list(records)
    random.Random(seed).shuffle(rows)
    tasks = [rows[i::n_tasks] for i in range(n_tasks)]
    if any(not task for task in tasks):
        raise ValueError("Not enough records")
    return tasks

def estimate_fisher(model, x, y, batch_size=32, max_batches=8):
    """Diagonal empirical Fisher approximation, computed on labeled training data."""
    model.eval()
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    count = 0
    for xs, ys in zip(x.split(batch_size), y.split(batch_size)):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xs), ys)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach().square()
        count += 1
        if count >= max_batches:
            break
    return {name: value / max(count, 1) for name, value in fisher.items()}

def ewc_penalty(model, snapshots):
    if not snapshots:
        return next(model.parameters()).new_zeros(())
    penalty = next(model.parameters()).new_zeros(())
    for old_params, fisher in snapshots:
        for name, p in model.named_parameters():
            penalty = penalty + (fisher[name] * (p - old_params[name]).square()).sum()
    return penalty

def run_continual(records, operator="coupled_dynamics", n_tasks=3, epochs=2,
                  ewc_lambda=0.0, seed=42, output_dir="storage/continual"):
    if epochs < 1 or ewc_lambda < 0:
        raise ValueError("epochs must be positive and lambda nonnegative")
    random.seed(seed)
    torch.manual_seed(seed)
    tasks = make_tasks(records, n_tasks, seed)
    labels = {name: i for i, name in enumerate(sorted({str(r["label"]) for r in records}))}
    if len(labels) < 2:
        raise ValueError("At least two classes required")
    vocab = build_vocab(records)
    model = TextClassifier(len(vocab), len(labels), operator=operator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    snapshots = []
    matrix = []
    for task in tasks:
        x, y = encode(task, vocab, labels)
        for _ in range(epochs):
            model.train()
            for ix in torch.randperm(len(x)).split(32):
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x[ix]), y[ix])
                if snapshots and ewc_lambda:
                    loss = loss + 0.5 * ewc_lambda * ewc_penalty(model, snapshots)
                loss.backward()
                optimizer.step()
        model.eval()
        row = []
        with torch.no_grad():
            for past in tasks:
                px, py = encode(past, vocab, labels)
                row.append(scores(model(px), py, len(labels))[0])
        matrix.append(row)
        snapshots.append((
            {name: p.detach().clone() for name, p in model.named_parameters()},
            estimate_fisher(model, x, y),
        ))
    # Backward transfer / forgetting uses best previous accuracy on each earlier task.
    forgetting = []
    for task_id in range(n_tasks - 1):
        best_previous = max(matrix[t][task_id] for t in range(task_id, n_tasks - 1))
        forgetting.append(best_previous - matrix[-1][task_id])
    result = {"operator": operator, "seed": seed, "ewc_lambda": ewc_lambda,
              "accuracy_matrix": matrix, "mean_forgetting": sum(forgetting)/len(forgetting) if forgetting else 0.0,
              "note": "Exploratory sequential batches; evaluation reuses training records. Not a held-out continual-learning benchmark."}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"continual-{operator}-{seed}-{ewc_lambda}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
