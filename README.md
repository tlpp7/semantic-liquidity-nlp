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

## P2 — controlled benchmarks
Run all 30 operators and the baseline over three seeds:

```bash
python scripts/benchmark.py --dataset storage/dataset.jsonl --epochs 5 --seeds 42 43 44
```

For a fast smoke comparison:

```bash
python scripts/benchmark.py --operators baseline softmax_beta coupled_dynamics --epochs 1 --seeds 42
```

P2 writes JSON and CSV summaries, individual run histories, checkpoints and dataset SHA-256 fingerprints into `storage/benchmarks/`. The **Benchmarks** dashboard tab compares Macro-F1, accuracy, elapsed time, parameter counts and learning curves. Runs use the same split seed and epoch count; parameter counts and FLOPs can differ. Do not interpret the leaderboard as proof of statistical superiority. For unbiased model selection, choose configurations using validation metrics before a final held-out evaluation.

## P3 — continual learning, diagnostics, and optional object storage

P3 introduces sequential-task training with optional diagonal empirical Fisher / EWC penalty, accuracy matrices, and a descriptive forgetting metric. The Streamlit **Continual Learning** tab visualizes sequential results.

```bash
python scripts/continual.py --dataset storage/dataset.jsonl --tasks 3 --epochs 2 --ewc-lambda 1.0
streamlit run dashboard/app.py
```

Prototype diagnostics are available through `semantic_liquidity.semantic_analysis.inspect_layer(layer, embeddings)`, returning per-token prototype probabilities, normalized entropy, and prototype norms.

Optional S3-compatible artifact upload (AWS S3 or MinIO) is implemented in `semantic_liquidity.artifacts.upload_artifact`. Supply credentials through environment variables or an IAM role, never through Git-tracked files. Upload is **explicit**, not automatic.

**Important experimental limitations:** tasks are randomly partitioned batches of one dataset, not independently defined domains; the current continual-learning accuracy matrix evaluates on training records, not held-out data. The Fisher approximation is diagonal and simplified. P3 is a research scaffold, not evidence of resistance to catastrophic forgetting. Future work should use domain-shift benchmarks, stratified per-task held-out evaluation, and repeated seeds.

## P4 — validation and uncertainty

Training now uses class-stratified train/validation/test partitions and rejects exact normalized-text duplicates or overlaps. Each class needs at least ten examples for the default 80/10/10 split. Use the P5 catalog to normalize and deduplicate data first.

`semantic_liquidity.validation` provides paired bootstrap confidence intervals, leakage reports and duplicate/conflicting-label diagnostics. The bootstrap is a descriptive paired estimator, not proof of statistical superiority; for model comparisons, ensure aligned observations and avoid repeated test-set selection.

## P5 — dataset catalog and provenance

```bash
python scripts/catalog.py csv my_dataset.csv --text-col text --label-col target --license CC0
python scripts/catalog.py list
# optional, requires Hugging Face datasets and license review:
python scripts/catalog.py hf imdb --split train --text-col text --label-col label --revision YOUR_PINNED_REVISION
```

Each imported dataset is normalized, deduplicated, content-addressed by SHA-256 and recorded in SQLite with class counts, source and declared license. The Data Catalog tab supports CSV ingestion. Source/license fields are declarations, not automatically verified facts. Kaggle datasets can be downloaded with the existing CLI and then imported into this catalog.

P4/P5 are functional foundations; complete nested cross-validation, dataset cards, license verification and statistical significance reporting in benchmark dashboards remain future enhancements.
