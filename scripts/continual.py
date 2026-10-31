import argparse
from semantic_liquidity.train import load_records
from semantic_liquidity.continual import run_continual

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="storage/dataset.jsonl")
p.add_argument("--operator", default="coupled_dynamics")
p.add_argument("--tasks", type=int, default=3)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--ewc-lambda", type=float, default=0.0)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
result = run_continual(load_records(args.dataset), args.operator, args.tasks, args.epochs, args.ewc_lambda, args.seed)
print("Mean forgetting:", result["mean_forgetting"])
