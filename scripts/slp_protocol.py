"""SLP v1: local binary framing shared by R, Python and Zig.

32-byte header: magic(4), version(1), op(1), dtype(1), flags(1),
rows(u32), cols(u32), payload_bytes(u64), request_id(u64).
All integers little-endian; payload contiguous row-major float64.
"""
import struct
import numpy as np

HEADER = struct.Struct("<4sBBBBIIQQ")
OPS = {"softmax": 1, "entropy": 2}
MAX_PAYLOAD = 128 * 1024 * 1024

def encode_frame(values, operation="softmax", request_id=0):
    if operation not in OPS or not 0 <= request_id < 2**64:
        raise ValueError("Invalid operation or request ID")
    x = np.asarray(values, dtype="<f8", order="C")
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite 2D float64 matrix with >=2 columns")
    payload = x.tobytes(order="C")
    if len(payload) > MAX_PAYLOAD or max(x.shape) > 1_000_000:
        raise ValueError("Payload exceeds protocol limits")
    return HEADER.pack(b"SLP1", 1, OPS[operation], 1, 0, *x.shape,
                       len(payload), request_id) + payload

def decode_frame(frame):
    if len(frame) < HEADER.size:
        raise ValueError("Truncated header")
    magic, version, op, dtype, flags, rows, cols, length, request_id = HEADER.unpack_from(frame)
    if (magic, version, dtype, flags) != (b"SLP1", 1, 1, 0) or op not in OPS.values():
        raise ValueError("Invalid SLP header")
    if not (1 <= rows <= 1_000_000 and 2 <= cols <= 1_000_000):
        raise ValueError("Invalid dimensions")
    if length != rows * cols * 8 or length > MAX_PAYLOAD or len(frame) != HEADER.size + length:
        raise ValueError("Invalid payload length")
    matrix = np.frombuffer(frame, dtype="<f8", offset=HEADER.size).reshape(rows, cols).copy()
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite payload")
    return {"operation": next(k for k, v in OPS.items() if v == op),
            "request_id": request_id, "matrix": matrix}
