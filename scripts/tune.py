import argparse
from semantic_liquidity.tuning import run_study

p=argparse.ArgumentParser(description="P7 validation-only hyperparameter tuning")
p.add_argument("--dataset",default="storage/dataset.jsonl")
p.add_argument("--operators",nargs="+",default=["baseline","softmax_beta"])
p.add_argument("--trials",type=int,default=12)
p.add_argument("--epochs",type=int,default=8)
p.add_argument("--seed",type=int,default=42)
p.add_argument("--timeout",type=int,default=None)
a=p.parse_args()
result=run_study(a.dataset,a.operators,a.trials,a.epochs,a.seed,timeout=a.timeout)
print("Best validation F1:",result["best_validation_macro_f1"])
print("Best hyperparameters:",result["best_params"])
