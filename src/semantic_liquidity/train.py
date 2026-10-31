"""Reproducible supervised text classification using a lightweight GRU + SLL."""
import json
import random
import sqlite3
import time
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .layers import SemanticLiquidityLayer, OPERATOR_NAMES
from .validation import stratified_split, leakage_report, detect_exact_duplicates

PAD, UNK = "<pad>", "<unk>"

def load_records(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if str(r.get("text", "")).strip() and str(r.get("label", "")).strip()]
    if len(rows) < 10 or len(set(str(r["label"]) for r in rows)) < 2:
        raise ValueError("Need >=10 nonempty examples and >=2 labels")
    return rows

def split_records(rows, seed=42, train_fraction=0.8, val_fraction=0.1):
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("Invalid split fractions")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    n = len(rows)
    a, b = int(n*train_fraction), int(n*(train_fraction+val_fraction))
    if not a or b <= a or b >= n:
        raise ValueError("Dataset too small for nonempty train/val/test splits")
    return [[rows[i] for i in part] for part in (indices[:a], indices[a:b], indices[b:])]

def build_vocab(rows, max_vocab=10000):
    counts = Counter(word for r in rows for word in str(r["text"]).lower().split())
    return {word: i for i, word in enumerate([PAD, UNK] + [w for w, _ in counts.most_common(max_vocab-2)])}

def encode(rows, vocab, labels, max_length=64):
    xs, ys = [], []
    for row in rows:
        tokens = str(row["text"]).lower().split()[:max_length]
        xs.append([vocab.get(w, 1) for w in tokens] + [0] * (max_length-len(tokens)))
        ys.append(labels[str(row["label"])])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, classes, dim=64, operator="softmax_beta"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.encoder = nn.GRU(dim, dim//2, batch_first=True, bidirectional=True)
        self.liquidity = SemanticLiquidityLayer(dim=dim, operator=operator) if operator != "baseline" else nn.Identity()
        self.classifier = nn.Linear(dim, classes)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        encoded, _ = self.encoder(embedded)
        enriched = self.liquidity(encoded)
        mask = (tokens != 0).unsqueeze(-1)
        pooled = (enriched * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(pooled)

def scores(logits, labels, classes):
    preds = logits.argmax(-1)
    accuracy = (preds == labels).float().mean().item()
    f1s = []
    for c in range(classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        f1s.append(2*tp / max(1, 2*tp+fp+fn))
    return accuracy, sum(f1s)/classes

def run_training(dataset="storage/dataset.jsonl", operator="softmax_beta", epochs=5, seed=42,
                 batch_size=32, lr=1e-3, output_dir="storage/runs", max_length=64):
    if operator not in OPERATOR_NAMES and operator != "baseline":
        raise ValueError("Unknown operator")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    random.seed(seed)
    torch.manual_seed(seed)
    rows = load_records(dataset)
    quality = detect_exact_duplicates(rows)
    if quality['duplicates']:
        raise ValueError('Duplicate normalized texts detected; import through P5 catalog with deduplication first')
    train, val, test = stratified_split(rows, seed)
    leakage = leakage_report(train, val, test)
    if any(leakage.values()):
        raise ValueError(f'Text leakage between partitions: {leakage}')
    vocab = build_vocab(train)
    labels = {label: i for i, label in enumerate(sorted({str(r["label"]) for r in train}))}
    if len(labels) < 2 or any(str(r["label"]) not in labels for r in val + test):
        raise ValueError("All classes must be present in training split; use more data")
    tr_x, tr_y = encode(train, vocab, labels, max_length)
    va_x, va_y = encode(val, vocab, labels, max_length)
    te_x, te_y = encode(test, vocab, labels, max_length)
    model = TextClassifier(len(vocab), len(labels), operator=operator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"{int(time.time()*1000)}-{seed}-{operator}"
    history = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        permutation = torch.randperm(len(tr_x))
        total = 0.0
        for ids in permutation.split(batch_size):
            optimizer.zero_grad()
            logits = model(tr_x[ids])
            loss = F.cross_entropy(logits, tr_y[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(ids)
        model.eval()
        with torch.no_grad():
            val_logits = model(va_x)
            val_loss = F.cross_entropy(val_logits, va_y).item()
            acc, f1 = scores(val_logits, va_y, len(labels))
        history.append({"epoch": epoch, "train_loss": total/len(tr_x), "val_loss": val_loss, "val_accuracy": acc, "val_macro_f1": f1})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(te_x)
        test_loss = F.cross_entropy(test_logits, te_y).item()
        test_accuracy, test_f1 = scores(test_logits, te_y, len(labels))
    checkpoint = root / f"{run_id}.pt"
    torch.save({"model": best_state, "vocab": vocab, "labels": labels, "operator": operator, "max_length": max_length}, checkpoint)
    result = {"run_id": run_id, "operator": operator, "seed": seed, "epochs": epochs, "train_size": len(train),
              "val_size": len(val), "test_size": len(test), "test_loss": test_loss,
              "test_accuracy": test_accuracy, "test_macro_f1": test_f1, "checkpoint": str(checkpoint), "history": history, "split_method": "stratified", "leakage": leakage}
    (root / f"{run_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with sqlite3.connect(root / "experiments.sqlite") as con:
        con.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, operator TEXT, seed INTEGER, test_accuracy REAL, test_macro_f1 REAL, checkpoint TEXT)")
        con.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)", (run_id, operator, seed, test_accuracy, test_f1, str(checkpoint)))
    return result
