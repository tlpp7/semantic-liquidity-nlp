# Semantic Liquidity NLP

Experimental NLP engine with 30 differentiable PyTorch semantic operators inspired by semantic activation, entropy, surprisal, plasticity proxies, flexibility and consolidation.

## P0 + P1
- 30 operator variants in six mathematical families
- CSV ingestion into local JSONL + SQLite registry
- Kaggle dataset CLI (requires credentials and compliance with dataset terms)
- **P1:** reproducible supervised text classifier (embedding + bidirectional GRU + optional SLL)
- **P1:** train/validation/test splits (80/10/10), train-only vocabulary, fixed random seed
- **P1:** validation loss, accuracy, macro-F1, best-validation checkpoint, held-out test metrics
- **P1:** run metadata and metrics in SQLite and JSON; Streamlit training/experiment dashboard
- GitHub Actions unit tests

## Installation
```bash
python -m pip install -e ".[dev,lab]"
pytest -q
streamlit run dashboard/app.py --server.address 0.0.0.0
```

## Dataset
Use the dashboard to upload a labeled CSV with `text` and `target` columns, or:
```bash
python scripts/kaggle_download.py owner/dataset-name
```
Import the resulting CSV via the dashboard. Kaggle dataset licensing and authentication are your responsibility.

## Train
```bash
python scripts/train.py --dataset storage/dataset.jsonl --operator coupled_dynamics --epochs 5 --seed 42
python scripts/train.py --dataset storage/dataset.jsonl --operator baseline --epochs 5 --seed 42
```
Each run saves a local checkpoint in `storage/runs/`, a JSON metrics history and an SQLite experiment row.

## Experimental limitations
These 30 operators are mathematical proxies, not biologically validated STDP or memory consolidation simulations. P1 is a functional training pipeline, **not evidence of improved NLP performance**. Data splitting is randomized but not stratified; tiny or imbalanced datasets can fail if a class is absent from the training set. For serious benchmarks use multiple seeds, stratified splits and identical compute budgets.

## Roadmap
- P0: project foundation, differentiable operators, dataset ingestion, dashboard and CI
- **P1: supervised training, checkpoints, run tracking and visualization**
- P2: systematic benchmarks across operators, controlled baselines and ablation studies
- P3: continual learning, semantic dynamics visualization, scalable artifact storage

Never commit Kaggle tokens, raw datasets, private records or model checkpoints. Local `storage/` is gitignored.

MIT license.
