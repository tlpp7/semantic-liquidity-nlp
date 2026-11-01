import pytest
from semantic_liquidity.memory import ReplayBuffer
from semantic_liquidity.tuning import run_study

def test_replay_reservoir_reproducible():
    rows=[{"text":str(i),"label":str(i%2)} for i in range(100)]
    a,b=ReplayBuffer(10,7),ReplayBuffer(10,7)
    a.add(rows)
    b.add(rows)
    assert a.rows==b.rows
    assert a.stats()["seen"]==100
    assert len(a.sample(100))==10

def test_replay_rejects_negative():
    with pytest.raises(ValueError):
        ReplayBuffer(0)

def test_tuning_rejects_bad_operator(tmp_path):
    pytest.importorskip("optuna")
    with pytest.raises(ValueError):
        run_study("missing.jsonl",operators=["invalid"],trials=1,output_dir=tmp_path)
