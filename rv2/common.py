"""Shared helpers: config, paths, hashing, seeding, JSON IO."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import yaml

RERUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(RERUN_DIR, "configs", "default.yaml")


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = os.path.expanduser(cfg["project_root"])
    cfg["project_root"] = root if os.path.isabs(root) else os.path.normpath(os.path.join(RERUN_DIR, root))
    cfg["_config_path"] = os.path.abspath(path or DEFAULT_CONFIG)
    for k, v in (overrides or {}).items():          # dotted keys: "baseline.epochs"
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    if cfg.get("frozen_override"):          # smoke tests: an alternative frozen-data directory
        cfg["_frozen_override"] = os.path.expanduser(cfg["frozen_override"])
    return cfg


def P(cfg: dict, rel: str) -> str:
    """Resolve a project-relative path."""
    rel = os.path.expanduser(rel)
    return rel if os.path.isabs(rel) else os.path.join(cfg["project_root"], rel)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def dump_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{platform.node()}.{os.getpid()}"      # PID-unique: concurrent writers never share a tmp
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Re-seed Python, NumPy, Torch (CPU+CUDA) at the start of EVERY run."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "executable": sys.executable,
            "host": platform.node(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        import torch, transformers, sklearn
        info.update(torch=torch.__version__, transformers=transformers.__version__,
                    sklearn=sklearn.__version__, cuda=torch.cuda.is_available(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:  # pragma: no cover
        pass
    return info


def code_hash(files=None) -> str:
    """SHA-256 over rv2 sources (all rv2/*.py, or only the listed module files)."""
    h = hashlib.sha256()
    d = os.path.join(RERUN_DIR, "rv2")
    for fn in sorted(files or [f for f in os.listdir(d) if f.endswith(".py")]):
        with open(os.path.join(d, fn), "rb") as f:
            h.update(fn.encode() + b"\0" + f.read())
    return h.hexdigest()


# Job-scoped code groups: a job's fingerprint covers only the modules on its code path,
# so an edit to the report code does not invalidate the GPU runs, and vice versa.
_EXEC = ["jobs.py", "launcher.py", "run.py"]      # executing/orchestration modules: in every group
CODE_GROUPS = {
    "train": sorted(_EXEC + ["common.py", "manifest.py", "metrics.py", "models.py", "select.py", "textnorm.py", "train.py",
                             "xai.py"]),
    "select": sorted(_EXEC + ["common.py", "metrics.py", "select.py"]),
    "report": sorted(_EXEC + ["common.py", "evaluate.py", "manifest.py", "metrics.py", "select.py"]),
}
_CODE_CACHE: dict = {}


def group_code_hash(group: str) -> str:
    if group not in _CODE_CACHE:
        _CODE_CACHE[group] = code_hash(CODE_GROUPS[group])
    return _CODE_CACHE[group]


_DIR_CACHE: dict = {}


def dir_hash(path: str) -> str:
    """Deterministic sha256 over every file (relative name + content) below a local model directory."""
    path = os.path.abspath(path)
    if path not in _DIR_CACHE:
        h = hashlib.sha256()
        for root, dirs, files in os.walk(path):
            dirs.sort()
            for fn in sorted(files):
                fp = os.path.join(root, fn)
                h.update(os.path.relpath(fp, path).encode() + b"\0" + sha256_file(fp).encode() + b"\n")
        _DIR_CACHE[path] = "sha256:" + h.hexdigest()
    return _DIR_CACHE[path]
