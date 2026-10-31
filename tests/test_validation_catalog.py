import csv
import pytest
from semantic_liquidity.validation import stratified_split, paired_bootstrap, detect_exact_duplicates, leakage_report
from semantic_liquidity.catalog import import_csv, list_datasets

def test_stratified_no_overlap():
    rows=[{"text":f"unique {i}","label":str(i%2)} for i in range(100)]
    a,b,c=stratified_split(rows,seed=4)
    assert len(a)+len(b)+len(c)==100
    assert set(r["text"] for r in a).isdisjoint(r["text"] for r in b)
    assert all({r["label"] for r in part}=={"0","1"} for part in (a,b,c))
    assert not any(leakage_report(a,b,c).values())

def test_bootstrap():
    r=paired_bootstrap([0.8]*8,[0.5]*8,repeats=100,seed=1)
    assert r["mean_difference"]==pytest.approx(0.3)
    assert r["ci_low"]==pytest.approx(0.3)

def test_duplicate_conflict():
    assert detect_exact_duplicates([{"text":"A  B","label":"0"},{"text":"a b","label":"1"}])["conflicting_labels"]==1

def test_catalog_versioned(tmp_path):
    csv_path=tmp_path/"samples.csv"
    with csv_path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["text","target"])
        w.writeheader()
        w.writerow({"text":"hello world","target":"a"})
        w.writerow({"text":"hello world","target":"a"})
        w.writerow({"text":"another sample","target":"b"})
    root=tmp_path/"catalog"
    a=import_csv(csv_path,root=root,license_name="CC0")
    b=import_csv(csv_path,root=root,license_name="CC0")
    assert a["sha256"]==b["sha256"]
    assert a["rows"]==2
    assert len(list_datasets(root))==1
