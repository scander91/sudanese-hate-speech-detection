"""Frozen manifest IO + leakage assertions.  EVERY trainer loads data only through here."""
from __future__ import annotations

import os
from itertools import combinations

import pandas as pd

from .common import P, load_json, sha256_file
from .textnorm import preprocess, sha256

STRICT_MODES = ("none", "minimal", "alg3")   # identical model input must never cross partitions
REPORT_MODES = ("mhamed",)                    # lossy stop-word representation: collisions reported


def write_manifest(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, sep="\t", index=False, encoding="utf-8")
    os.replace(tmp, path)


def read_manifest(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")


def write_checksums(d: str) -> None:
    lines = []
    for fn in sorted(os.listdir(d)):
        if fn == "CHECKSUMS.sha256" or fn.endswith(".tmp"):
            continue
        lines.append(f"{sha256_file(os.path.join(d, fn))}  {fn}")
    with open(os.path.join(d, "CHECKSUMS.sha256"), "w") as f:
        f.write("\n".join(lines) + "\n")


def verify_checksums(d: str, files=None) -> None:
    want = {}
    with open(os.path.join(d, "CHECKSUMS.sha256")) as f:
        for line in f:
            h, fn = line.strip().split("  ", 1)
            want[fn] = h
    for fn in files or want:
        got = sha256_file(os.path.join(d, fn))
        if got != want[fn]:
            raise RuntimeError(f"manifest checksum mismatch: {fn} ({got} != {want[fn]})")


def assert_disjoint(df: pd.DataFrame, col: str, parts=("train", "val", "test"), fatal_all: bool = False) -> dict:
    """Pairwise-empty intersections of uid, raw-text hash and every model-input hash.
    fatal_all=True (fail-closed datasets, data.fail_closed_all_representations) also makes the lossy CNN
    representation (mhamed) fatal instead of reported."""
    out = {}
    views = {"uid": df.uid, "raw_sha256": df.raw_sha256}
    for m in STRICT_MODES + REPORT_MODES:
        views[m] = df.text.map(lambda t, m=m: sha256(preprocess(t, m)))
    for name, s in views.items():
        sets = {p: set(s[df[col] == p]) for p in parts}
        for a, b in combinations(parts, 2):
            n = len(sets[a] & sets[b])
            out[f"{name}:{a}&{b}"] = n
            if (fatal_all or name not in REPORT_MODES) and n:
                raise AssertionError(f"LEAKAGE: {n} {name} hashes shared by {a} and {b}")
    return out


# ------------------------------------------------------------------ loaders
def _frozen(cfg):
    return P(cfg, cfg["frozen_dir"])


def load_hs(cfg: dict, task: str, frozen_dir: str | None = None) -> dict:
    """task in {'binary','3class'} -> {'label_names', 'train'|'val'|'test': dict(uid,text,y)}"""
    d = frozen_dir or _frozen(cfg)
    verify_checksums(d, ["hs_manifest.tsv"])
    df = read_manifest(os.path.join(d, "hs_manifest.tsv"))
    assert df.uid.is_unique
    assert_disjoint(df, "split")
    names = cfg["data"]["hs_labels_3class"] if task == "3class" else cfg["data"]["hs_labels_binary"]
    col = "label3" if task == "3class" else "label_bin"
    l2i = {n: i for i, n in enumerate(names)}
    out = {"label_names": names, "manifest": "hs_manifest.tsv"}
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        out[s] = {"uid": sub.uid.tolist(), "text": sub.text.tolist(), "y": [l2i[v] for v in sub[col]]}
    return out


def load_telecom(cfg: dict, frozen_dir: str | None = None, for_mtl_aux: bool = False) -> dict:
    d = frozen_dir or _frozen(cfg)
    verify_checksums(d, ["telecom_manifest.tsv"])
    df = read_manifest(os.path.join(d, "telecom_manifest.tsv"))
    assert df.uid.is_unique
    assert_disjoint(df, "split")
    names = cfg["data"]["sent_labels"]
    l2i = {n: i for i, n in enumerate(names)}
    out = {"label_names": names, "manifest": "telecom_manifest.tsv"}
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        if for_mtl_aux and s == "train":
            sub = sub[sub.mtl_exclude == "0"]
        out[s] = {"uid": sub.uid.tolist(), "text": sub.text.tolist(), "y": [l2i[v] for v in sub.label]}
    return out


FOLD_TASKS = ("sudsenti3", "sudsenti2")


def sent_labels(cfg: dict, name: str) -> list:
    return cfg["data"].get(f"{name}_labels") or cfg["data"]["sent_labels"]


def load_fold_task(cfg: dict, name: str, fold: int, frozen_dir: str | None = None) -> dict:
    """10-fold sentiment datasets (SudSenti3, SudSenti2): frozen fold lists, all-representation disjointness."""
    d = frozen_dir or _frozen(cfg)
    verify_checksums(d, [f"{name}_manifest.tsv", f"{name}_folds.json"])
    df = read_manifest(os.path.join(d, f"{name}_manifest.tsv")).set_index("uid", drop=False)
    folds = load_json(os.path.join(d, f"{name}_folds.json"))[str(fold)]
    names = sent_labels(cfg, name)
    l2i = {n: i for i, n in enumerate(names)}
    tagged = pd.concat([df.loc[folds[s]].assign(part=s) for s in ("train", "val", "test")])
    assert_disjoint(tagged, "part", fatal_all=name in (cfg["data"].get("fail_closed_all_representations") or []))
    out = {"label_names": names, "manifest": f"{name}_manifest.tsv", "fold": fold}
    for s in ("train", "val", "test"):
        sub = df.loc[folds[s]]
        out[s] = {"uid": sub.uid.tolist(), "text": sub.text.tolist(), "y": [l2i[v] for v in sub.label]}
    return out


def load_sudsenti3_fold(cfg: dict, fold: int, frozen_dir: str | None = None) -> dict:
    return load_fold_task(cfg, "sudsenti3", fold, frozen_dir)


def load_task(cfg: dict, task: str, fold: int | None = None, frozen_dir: str | None = None) -> dict:
    if task in ("binary", "3class"):
        return load_hs(cfg, task, frozen_dir)
    if task == "telecom":
        return load_telecom(cfg, frozen_dir)
    if task in FOLD_TASKS:
        return load_fold_task(cfg, task, int(fold), frozen_dir)
    raise ValueError(task)
