import csv
from semantic_liquidity.data import ingest_csv

def test_ingest(tmp_path):
    p = tmp_path / "train.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "target"])
        w.writeheader()
        w.writerow({"text": "river bank", "target": "river"})
    result = ingest_csv(p, root=tmp_path / "storage")
    assert result["rows"] == 1
