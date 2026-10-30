"""Usage: python scripts/kaggle_download.py owner/dataset-name --out data/raw
Requires Kaggle credentials and acceptance of any dataset/competition terms."""
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Kaggle dataset slug owner/name")
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    if "/" not in args.dataset or args.dataset.startswith("/"):
        parser.error("Expected owner/dataset-name")
    subprocess.run(["kaggle", "datasets", "download", "-d", args.dataset, "-p", args.out, "--unzip"], check=True)

if __name__ == "__main__":
    main()
