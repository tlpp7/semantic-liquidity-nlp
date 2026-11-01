"""Protocol and Python-to-Zig circuit tests; Zig tests are skipped if not built."""
import ctypes
import importlib.util
import os
import struct
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

slp = module("slp_protocol", "scripts/slp_protocol.py")
bridge = module("zig_bridge", "scripts/zig_bridge.py")

@pytest.mark.parametrize("shape", [(1, 2), (2, 3), (32, 64), (64, 128)])
def test_roundtrip(shape):
    x = np.random.default_rng(42).normal(size=shape)
    frame = slp.encode_frame(x, request_id=987654321)
    y = slp.decode_frame(frame)
    assert len(frame) == 32 + x.size * 8
    assert y["request_id"] == 987654321
    np.testing.assert_array_equal(y["matrix"], x)

@pytest.mark.parametrize("mutation", [
    lambda b: b[:3] + b"X" + b[4:],
    lambda b: b[:4] + b"\x02" + b[5:],
    lambda b: b[:-1],
    lambda b: b + b"\x00",
    lambda b: b[:16] + b"\x00" * 8 + b[24:],
])
def test_reject_corrupt_frames(mutation):
    frame = slp.encode_frame([[1., 2., 3.]])
    with pytest.raises(ValueError):
        slp.decode_frame(mutation(frame))

def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        slp.encode_frame([[1., np.nan]])
    frame = bytearray(slp.encode_frame([[1., 2.]]))
    frame[32:40] = struct.pack("<d", float("inf"))
    with pytest.raises(ValueError):
        slp.decode_frame(bytes(frame))

def test_payload_is_row_major():
    x = np.array([[1., 2., 3.], [4., 5., 6.]])
    frame = slp.encode_frame(x)
    np.testing.assert_array_equal(np.frombuffer(frame[32:], dtype="<f8"), [1, 2, 3, 4, 5, 6])

@pytest.fixture(scope="module")
def core():
    path = os.environ.get("SLP_ZIG_LIBRARY")
    if not path or not Path(path).exists():
        pytest.skip("Zig native library not built (SLP_ZIG_LIBRARY missing)")
    return bridge.ZigCore(path)

def test_native_softmax_entropy(core):
    x = np.array([[1., 2., 3.], [4., 5., 6.]])
    transported = slp.decode_frame(slp.encode_frame(x))["matrix"]
    p = core.softmax(transported)
    ref = np.exp(x - x.max(axis=1, keepdims=True))
    ref /= ref.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(p, ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
    h = core.entropy(p)
    np.testing.assert_allclose(h, -(ref * np.log(ref)).sum(axis=1) / np.log(3), rtol=1e-12)
    assert (h >= 0).all() and (h <= 1).all()

def test_native_bad_probabilities(core):
    with pytest.raises(ValueError):
        core.entropy([[0.1, 0.1]])
