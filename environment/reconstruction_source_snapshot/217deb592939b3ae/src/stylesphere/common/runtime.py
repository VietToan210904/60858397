import importlib.metadata
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .io import utc_now, write_json


def require_cuda():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in this interpreter") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(f"Local CUDA inference unavailable: torch={torch.__version__}, compiled_cuda={torch.version.cuda}. Use an isolated CUDA environment; CPU fallback is prohibited.")
    return torch


def seed_everything(seed):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


class Measurement:
    """Synchronized wall-clock seconds; allocator MiB, not whole GPU usage."""
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.elapsed_s = None
        self.peak_allocated_mb = None
        self.peak_reserved_mb = None

    def __enter__(self):
        self.torch = require_cuda() if self.cuda else None
        if self.torch:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.torch:
            try:
                self.torch.cuda.synchronize()
            finally:
                self.elapsed_s = time.perf_counter() - self.start
                self.peak_allocated_mb = self.torch.cuda.max_memory_allocated() / 2**20
                self.peak_reserved_mb = self.torch.cuda.max_memory_reserved() / 2**20
        else:
            self.elapsed_s = time.perf_counter() - self.start

    def as_dict(self):
        return {"elapsed_s": self.elapsed_s, "peak_allocated_mb": self.peak_allocated_mb, "peak_reserved_mb": self.peak_reserved_mb}


def doctor(root, output=None):
    root = Path(root)
    packages = {}
    for name in ("torch", "torchvision", "transformers", "diffusers", "open_clip_torch", "numpy", "scipy", "PyYAML", "pytest", "matplotlib", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"timestamp_utc": utc_now(), "platform": platform.platform(), "python": sys.version, "executable": sys.executable, "packages": packages, "disk_bytes": shutil.disk_usage(root)._asdict(), "tools": {k: shutil.which(k) for k in ("python", "uv", "git", "nvidia-smi", "nvcc", "cl", "wsl")}, "virtual_env": os.environ.get("VIRTUAL_ENV"), "project_environments": [x.name for x in root.glob(".venv*") if x.is_dir()]}
    try:
        import torch
        report["torch"] = {"version": torch.__version__, "compiled_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "devices": []}
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            report["torch"]["devices"].append({"name": prop.name, "total_bytes": total, "free_bytes": free, "compute_capability": list(torch.cuda.get_device_capability(index))})
    except Exception as exc:
        report["torch_error"] = str(exc)
    if shutil.which("nvidia-smi"):
        completed = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        report["nvidia_smi"] = {"returncode": completed.returncode, "output": completed.stdout.strip(), "error": completed.stderr.strip()}
    write_json(output or root / "environment" / "doctor.json", report)
    return report
