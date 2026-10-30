import argparse
from semantic_liquidity.benchmark import run_benchmark

def main():
    parser = argparse.ArgumentParser(description="Compare NLP operators")
    parser.add_argument("--dataset", default="storage/dataset.jsonl")
    parser.add_argument("--operators", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = run_benchmark(dataset=args.dataset, operators=args.operators, seeds=args.seeds, epochs=args.epochs, batch_size=args.batch_size)
    print("Results:", result["directory"])
    for row in result["summary"]:
        print(row["operator"], round(row["macro_f1_mean"], 4))

if __name__ == "__main__":
    main()
