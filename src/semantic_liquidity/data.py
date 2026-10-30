"""Local dataset registry and CSV normalization. No credentials or raw data committed."""
import csv
import json
import sqlite3
from pathlib import Path

def ingest_csv(csv_path, text_col="text", label_col="target", root="storage"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    db = root / "registry.sqlite"
    output = root / "dataset.jsonl"
    count = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as src, output.open("w", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        if not {text_col, label_col}.issubset(reader.fieldnames or []):
            raise ValueError("Required columns missing")
        for row in reader:
            if not row[text_col] or not row[label_col]:
                continue
            dst.write(json.dumps({"text": row[text_col], "label": row[label_col]}, ensure_ascii=False) + "\n")
            count += 1
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE IF NOT EXISTS datasets (id INTEGER PRIMARY KEY, path TEXT, rows INTEGER, imported_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        con.execute("INSERT INTO datasets(path, rows) VALUES (?, ?)", (str(output), count))
    return {"rows": count, "path": str(output), "registry": str(db)}
