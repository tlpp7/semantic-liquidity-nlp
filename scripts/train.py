import argparse
from semantic_liquidity.train import run_training

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="storage/dataset.jsonl")
parser.add_argument("--operator", default="softmax_beta")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()
result = run_training(dataset=args.dataset, operator=args.operator, epochs=args.epochs, seed=args.seed, batch_size=args.batch_size)
print(f"Run: {result['run_id']} | Test accuracy: {result['test_accuracy']:.3f} | Macro-F1: {result['test_macro_f1']:.3f}")
