"""High-level Python -> Zig C ABI bridge, with strict shape validation."""
import ctypes
from pathlib import Path
import numpy as np

class ZigCore:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        ptr = ctypes.POINTER(ctypes.c_double)
        self.lib.sl_softmax_f64.argtypes = [ptr, ptr, ctypes.c_size_t, ctypes.c_size_t]
        self.lib.sl_entropy_f64.argtypes = [ptr, ptr, ctypes.c_size_t, ctypes.c_size_t]
        self.lib.sl_softmax_f64.restype = ctypes.c_int
        self.lib.sl_entropy_f64.restype = ctypes.c_int

    @staticmethod
    def _matrix(values):
        a = np.asarray(values, dtype=np.float64, order="C")
        if a.ndim != 2 or not a.size or not np.isfinite(a).all():
            raise ValueError("Expected nonempty finite 2D matrix")
        return a

    def softmax(self, values):
        x = self._matrix(values)
        out = np.empty_like(x)
        ptr = ctypes.POINTER(ctypes.c_double)
        status = self.lib.sl_softmax_f64(x.ctypes.data_as(ptr), out.ctypes.data_as(ptr), *x.shape)
        if status:
            raise ValueError(f"Zig softmax failed with status {status}")
        return out

    def entropy(self, probabilities):
        p = self._matrix(probabilities)
        if p.shape[1] < 2:
            raise ValueError("At least two prototypes required")
        out = np.empty(p.shape[0], dtype=np.float64)
        ptr = ctypes.POINTER(ctypes.c_double)
        status = self.lib.sl_entropy_f64(p.ctypes.data_as(ptr), out.ctypes.data_as(ptr), *p.shape)
        if status:
            raise ValueError(f"Zig entropy failed with status {status}")
        return out
