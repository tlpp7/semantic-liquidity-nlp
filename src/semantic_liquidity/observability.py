"""P6: lightweight per-epoch JSONL telemetry and hardware/gradient diagnostics."""
import json
import math
import os
import time
from pathlib import Path

import torch


def gradient_metrics(model):
    squared = 0.0
    maximum = 0.0
    count = 0
    nonfinite = 0
    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        values = grad.detach()
        finite = torch.isfinite(values)
        nonfinite += int((~finite).sum().item())
        safe = torch.where(finite, values, torch.zeros_like(values))
        squared += float(safe.double().square().sum().item())
        maximum = max(maximum, float(safe.abs().max().item()))
        count += values.numel()
    return {"grad_norm": math.sqrt(squared), "grad_abs_max": maximum,
            "grad_elements": count, "grad_nonfinite": nonfinite}


def resource_snapshot():
    snapshot = {"cpu_rss_bytes": None, "cuda_allocated_bytes": None,
                "cuda_reserved_bytes": None, "cuda_peak_allocated_bytes": None}
    try:
        import psutil
        snapshot["cpu_rss_bytes"] = psutil.Process(os.getpid()).memory_info().rss
        snapshot["system_cpu_percent"] = psutil.cpu_percent(interval=None)
    except ImportError:
        snapshot["system_cpu_percent"] = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        snapshot.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                        cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    return snapshot


class TelemetryWriter:
    def __init__(self, path, run_id):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.started = time.perf_counter()

    def log(self, event, **fields):
        payload = {"run_id": self.run_id, "event": event,
                   "elapsed_seconds": round(time.perf_counter()-self.started, 6),
                   "timestamp_unix": time.time(), **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, allow_nan=False) + "\n")
        return payload


def read_events(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
