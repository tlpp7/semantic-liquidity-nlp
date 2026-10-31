"""P5: versioned, provenance-aware dataset catalog. Local-only artifacts."""
import csv
import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .validation import detect_exact_duplicates

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def normalize_rows(rows, text_col="text", label_col="target", deduplicate=True):
    cleaned, seen, skipped = [], set(), 0
    for row in rows:
        text = " ".join(str(row.get(text_col) or "").split())
        label = str(row.get(label_col) or "").strip()
        if not text or not label:
            skipped += 1
            continue
        key = (text.casefold(), label)
        if deduplicate and key in seen:
            skipped += 1
            continue
        seen.add(key)
        cleaned.append({"text":text,"label":label})
    return cleaned, skipped

def import_csv(path, root="storage/catalog", text_col="text", label_col="target",
               source="local", license_name="unknown", deduplicate=True):
    """Import CSV as content-addressed JSONL; reject conflicting duplicate texts."""
    path, root = Path(path), Path(root)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if text_col not in (reader.fieldnames or []) or label_col not in (reader.fieldnames or []):
            raise ValueError("Missing text/label columns")
        rows, skipped = normalize_rows(reader, text_col, label_col, deduplicate)
    if not rows:
        raise ValueError("No valid records")
    quality = detect_exact_duplicates(rows)
    if quality["conflicting_labels"]:
        raise ValueError("Conflicting labels for identical normalized text")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, suffix=".tmp", delete=False) as temp:
        temporary = Path(temp.name)
        for row in rows:
            temp.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n")
    digest = sha256(temporary)
    target = root / f"{digest}.jsonl"
    if target.exists():
        temporary.unlink()
    else:
        temporary.replace(target)
    metadata = {"id":digest, "path":str(target), "source":source, "license":license_name,
                "rows":len(rows), "skipped":skipped, "class_counts":dict(Counter(r["label"] for r in rows)),
                "sha256":digest, "imported_at":datetime.now(timezone.utc).isoformat(),
                "quality":quality}
    with sqlite3.connect(root/"catalog.sqlite") as con:
        con.execute("CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, path TEXT, source TEXT, license TEXT, rows INTEGER, sha256 TEXT, metadata_json TEXT)")
        con.execute("INSERT OR IGNORE INTO datasets VALUES (?,?,?,?,?,?,?)",
                    (digest,str(target),source,license_name,len(rows),digest,json.dumps(metadata)))
    (root/f"{digest}.metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return metadata

def list_datasets(root="storage/catalog"):
    db = Path(root)/"catalog.sqlite"
    if not db.exists():
        return []
    with sqlite3.connect(db) as con:
        return [json.loads(r[0]) for r in con.execute("SELECT metadata_json FROM datasets ORDER BY rowid DESC")]

def import_huggingface(dataset_id, split, text_col, label_col, root="storage/catalog",
                       license_name="unknown", revision=None, max_rows=100000):
    """Optional Hugging Face connector. Pin revision for reproducibility."""
    from datasets import load_dataset
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    ds = load_dataset(dataset_id, split=split, revision=revision, trust_remote_code=False)
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=root_path, suffix=".csv", delete=False) as f:
        temporary = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=["text","target"])
        writer.writeheader()
        for i, row in enumerate(ds):
            if i >= max_rows:
                break
            writer.writerow({"text":row.get(text_col), "target":row.get(label_col)})
    try:
        return import_csv(temporary,root=root,source=f"hf:{dataset_id}@{revision or 'unPinned'}:{split}",license_name=license_name)
    finally:
        temporary.unlink(missing_ok=True)
