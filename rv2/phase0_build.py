"""Phase 0 -- build the frozen, de-duplicated data manifests.

    python -m rv2.phase0_build [--config configs/default.yaml] [--limit N]

Outputs (config.frozen_dir):
    hs_manifest.tsv            one row per de-duplicated HS sentence, split, labels
    hs_lf_matrix_40k.npy       40 LF votes for the 40,000 original rows (LF module, read-only import)
    telecom_manifest.tsv       de-duplicated Telecom with train/val/test split
    sudsenti3_manifest.tsv     de-duplicated SudSenti3 with fold ids
    sudsenti3_folds.json       per-fold train/val/test uid lists
    sudsenti2_manifest.tsv     de-duplicated SudSenti2 with fold ids (also addable later: --addendum sudsenti2)
    sudsenti2_folds.json       per-fold train/val/test uid lists
    phase0_report.json         all counts, checksums, LF weights, label flips
    CHECKSUMS.sha256           sha256 of every frozen file

Label rules reproduced from the data (verified 40,000/40,000 in the report):
    final_3class = majority(ws, gpt, llama) if some label has >=2 votes else gpt
    binary       = HARMFUL if final_3class in {HATE, OFFENSIVE} else NEUTRAL
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from .common import P, dump_json, load_config, load_json, sha256_file, env_info, code_hash
from .textnorm import CANON_VERSION, ALG3_VERSION, canonical_key, preprocess, sha256
from . import manifest as M

HATE, OFFENSIVE, NEUTRAL, ABSTAIN = 0, 1, 2, -1
NAME = {HATE: "HATE", OFFENSIVE: "OFFENSIVE", NEUTRAL: "NEUTRAL"}


# ------------------------------------------------------------------ helpers
def read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                       keep_default_na=False, encoding="utf-8")


def resolve(ws: str, gpt: str, llama: str) -> tuple[str, str]:
    c = Counter([ws, gpt, llama]).most_common()
    if c[0][1] == 3:
        return c[0][0], "full_agreement"
    if c[0][1] == 2:
        return c[0][0], "majority_vote"
    return gpt, "gpt_tiebreaker"


def to_binary(lbl: str) -> str:
    return "HARMFUL" if lbl in ("HATE", "OFFENSIVE") else "NEUTRAL"


def import_old_lfs(path: str):
    spec = importlib.util.spec_from_file_location("snorkel_pipeline_v3_readonly", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)     # module has an if __name__ guard; import has no side effects
    assert (mod.HATE, mod.OFFENSIVE, mod.NEUTRAL, mod.ABSTAIN) == (HATE, OFFENSIVE, NEUTRAL, ABSTAIN)
    return mod


def lf_weights(L: np.ndarray) -> np.ndarray:
    """Exactly snorkel_pipeline_v3.weighted_vote's weights, fitted on the rows of L."""
    n, m = L.shape
    w = np.array([1.0 / (np.sum(L[:, j] != ABSTAIN) / n + 0.005) for j in range(m)])
    return w / w.sum() * m


def weighted_vote_with(L: np.ndarray, w: np.ndarray):
    """snorkel_pipeline_v3.weighted_vote with externally supplied (frozen) weights.
    Tie policy and no-vote default are unchanged: max over dict in order
    HATE, OFFENSIVE, NEUTRAL (first wins); no votes -> NEUTRAL, confidence 0."""
    n, m = L.shape
    labels = np.full(n, NEUTRAL, dtype=int)
    confs = np.zeros(n)
    novote = np.zeros(n, dtype=bool)
    for i in range(n):
        sc = {HATE: 0.0, OFFENSIVE: 0.0, NEUTRAL: 0.0}
        tw = 0.0
        for j in range(m):
            if L[i, j] != ABSTAIN:
                sc[L[i, j]] += w[j]
                tw += w[j]
        if tw == 0:
            labels[i], confs[i], novote[i] = NEUTRAL, 0.0, True
        else:
            best = max(sc, key=sc.get)
            labels[i], confs[i] = best, sc[best] / tw
    return labels, confs, novote


def dedup(df: pd.DataFrame, label_col: str, id_col: str, report: dict, tag: str) -> pd.DataFrame:
    """Canonicalise, group by key, detect label conflicts, keep one row per clean group."""
    df = df.copy()
    df["canon_key"] = [canonical_key(t) for t in df["text"]]
    df["uid"] = [sha256(k) for k in df["canon_key"]]
    df["raw_sha256"] = [sha256(t) for t in df["text"]]
    n0 = len(df)
    empty = df["canon_key"].str.len() == 0
    raw_dup_rows = int(df.duplicated("raw_sha256", keep="first").sum())
    raw_groups = df.groupby("raw_sha256").size()
    g = df[~empty].groupby("uid")
    sizes = g.size()
    nlab = g[label_col].nunique()
    conflict_uids = set(nlab[nlab > 1].index)
    raw_conf = df.groupby("raw_sha256")[label_col].nunique()
    r = {
        "rows_in": n0,
        "empty_after_canonicalisation": int(empty.sum()),
        "exact_raw_duplicate_rows_beyond_first": raw_dup_rows,
        "exact_raw_duplicate_groups": int((raw_groups > 1).sum()),
        "exact_raw_groups_with_label_conflict": int((raw_conf > 1).sum()),
        "canonical_groups": int(len(sizes)),
        "canonical_groups_size_gt1": int((sizes > 1).sum()),
        "rows_in_canonical_groups_size_gt1": int(sizes[sizes > 1].sum()),
        "canonical_groups_with_label_conflict": len(conflict_uids),
        "rows_in_conflicting_groups": int(sizes[list(conflict_uids)].sum()) if conflict_uids else 0,
    }
    keep = df[~empty & ~df["uid"].isin(conflict_uids)]
    keep = keep.sort_values(id_col, key=lambda s: s.astype(int)).drop_duplicates("uid", keep="first")
    r["rows_out"] = int(len(keep))
    r["rows_removed_total"] = n0 - int(len(keep))
    report[tag] = r
    return keep.reset_index(drop=True)


# ------------------------------------------------------------------ HS
def build_hs(cfg, out, report, limit=None):
    d = cfg["data"]
    res = read_tsv(P(cfg, d["hs_resolved"]))
    d3 = read_tsv(P(cfg, d["hs_3class_tsv"]))
    db = read_tsv(P(cfg, d["hs_binary_tsv"]))
    ws = read_tsv(P(cfg, d["ws_tsv"]))
    corpus = load_json(P(cfg, d["hs_corpus_json"]))
    chk = {}
    assert len(res) == len(d3) == len(db) == len(ws) == len(corpus) == 40000
    chk["resolved_text_eq_dataset_3class"] = bool((res.text == d3.text).all())
    chk["resolved_final_eq_dataset_3class"] = bool((res.final_label == d3.label).all())
    chk["binary_text_eq_3class"] = bool((db.text == d3.text).all())
    chk["binary_eq_rule(3class)"] = bool((d3.label.map(to_binary) == db.label).all())
    chk["ws_tsv_eq_resolved_ws"] = bool((ws.label_name == res.ws_label).all())
    rr = [resolve(a, b, c) for a, b, c in zip(res.ws_label, res.gpt_label, res.llama_label)]
    chk["final_eq_rule(ws,gpt,llama)"] = int(sum(x[0] == y for x, y in zip(rr, res.final_label)))
    chk["method_eq_rule"] = int(sum(x[1] == y for x, y in zip(rr, res.method)))
    tsv_json_text_eq = sum(a == b["text"].replace("\t", " ").replace("\n", " ") for a, b in zip(res.text, corpus))
    chk["resolved_text_eq_corpus_json_text(tab/newline->space)"] = int(tsv_json_text_eq)
    chk["human_verified_nonempty"] = int((res.human_verified.str.len() > 0).sum())
    assert all(v is True or v == 40000 or k == "human_verified_nonempty" for k, v in chk.items()), chk

    # --- 40 LFs on the ORIGINAL json texts (what the weak-supervision pipeline labelled)
    mod = import_old_lfs(P(cfg, d["snorkel_module"]))
    lf_names = [n for n, _ in mod.ALL_LFS]
    texts_json = [c["text"] for c in corpus]
    Lpath = os.path.join(out, "hs_lf_matrix_40k.npy")
    if os.path.exists(Lpath) and limit is None:
        L = np.load(Lpath)
    else:
        idx = range(len(texts_json)) if limit is None else range(limit)
        L = np.full((len(texts_json), len(lf_names)), ABSTAIN, dtype=np.int8)
        for i in idx:
            t = texts_json[i]
            for j, (_, f) in enumerate(mod.ALL_LFS):
                L[i, j] = f(t)
        if limit is None:
            np.save(Lpath, L)
    w_full = lf_weights(L)
    lab_full, _, _ = weighted_vote_with(L, w_full)
    ws_old = res.ws_label.map({"HATE": 0, "OFFENSIVE": 1, "NEUTRAL": 2}).values
    chk["old_weighted_vote_reproduced"] = int((lab_full == ws_old).sum())
    if limit is None:
        assert chk["old_weighted_vote_reproduced"] == 40000, chk
    report["hs_input_checks"] = chk
    report["lf"] = {"n_lfs": len(lf_names), "names": lf_names,
                    "by_prefix_note": "counts per target class are in lf_analysis.json of the weak-supervision pipeline"}

    # --- dedup on the v1 (paper) 3-class label
    df = res.rename(columns={"final_label": "label3_v1"}).copy()
    df["orig_row"] = np.arange(len(df))
    kept = dedup(df, "label3_v1", "id", report, "hs_dedup")
    # binary-conflict view (for the report only; binary is derived from 3-class)
    df["labelb_v1"] = df.label3_v1.map(to_binary)
    tmp = {}
    dedup(df, "labelb_v1", "id", tmp, "hs_dedup_binary_view")
    report["hs_dedup"]["canonical_groups_with_binary_label_conflict"] = tmp["hs_dedup_binary_view"]["canonical_groups_with_label_conflict"]

    # --- split (stratified on v1 3-class label, two-stage)
    seed = cfg["data"]["split_seed"]
    idx = np.arange(len(kept))
    y = kept.label3_v1.values
    tr, tmp_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    va, te = train_test_split(tmp_idx, test_size=0.5, random_state=seed, stratify=y[tmp_idx])
    split = np.empty(len(kept), dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    kept["split"] = split

    # --- LF weights: train-only (config) vs full-corpus, same formula
    Lk = L[kept.orig_row.values]
    w_train = lf_weights(Lk[kept.split.values == "train"])
    lab_tr, conf_tr, nov = weighted_vote_with(Lk, w_train)
    lab_fu, _, _ = weighted_vote_with(Lk, w_full)
    fit = cfg["data"]["lf_weight_fit"]
    ws_new = lab_tr if fit == "train_only" else lab_fu
    kept["ws_v1"] = kept.ws_label
    kept["ws_v2"] = [NAME[int(v)] for v in ws_new]
    kept["ws_v2_conf"] = np.round(conf_tr if fit == "train_only" else 0.0, 4)
    kept["ws_novote"] = nov.astype(int)
    rr = [resolve(a, b, c) for a, b, c in zip(kept.ws_v2, kept.gpt_label, kept.llama_label)]
    kept["label3"] = [x[0] for x in rr]
    kept["method_v2"] = [x[1] for x in rr]
    kept["label_bin"] = kept.label3.map(to_binary)
    kept["labelb_v1"] = kept.label3_v1.map(to_binary)
    flips = {}
    for s in ("train", "val", "test", "all"):
        m = np.ones(len(kept), bool) if s == "all" else (kept.split.values == s)
        flips[s] = {"n": int(m.sum()),
                    "ws_label_changed": int((kept.ws_v1.values[m] != kept.ws_v2.values[m]).sum()),
                    "final3_changed": int((kept.label3_v1.values[m] != kept.label3.values[m]).sum()),
                    "binary_changed": int((kept.labelb_v1.values[m] != kept.label_bin.values[m]).sum())}
    report["lf_weights"] = {
        "fit": fit, "formula": "w_j = 1/(coverage_j + 0.005), normalised to sum m (snorkel_pipeline_v3.py weighted_vote)",
        "w_full_corpus_40k": dict(zip(lf_names, np.round(w_full, 6))),
        "w_train_only": dict(zip(lf_names, np.round(w_train, 6))),
        "max_abs_rel_diff": float(np.max(np.abs(w_train - w_full) / w_full)),
        "label_changes_train_only_vs_old": flips,
        "no_vote_rows_default_NEUTRAL": int(nov.sum()),
    }
    # --- distributions
    dist = {}
    for s in ("train", "val", "test"):
        sub = kept[kept.split == s]
        dist[s] = {"n": int(len(sub)), "label3": dict(Counter(sub.label3)),
                   "label_bin": dict(Counter(sub.label_bin)), "label3_v1": dict(Counter(sub.label3_v1))}
    report["hs_split"] = dist

    cols = ["uid", "id", "orig_row", "split", "label3", "label_bin", "label3_v1", "labelb_v1",
            "ws_v2", "ws_v2_conf", "ws_novote", "ws_v1", "gpt_label", "llama_label", "method_v2",
            "method", "source", "keyword_cat", "raw_sha256", "text"]
    man = kept[cols].rename(columns={"id": "orig_id", "method": "method_v1"})
    M.write_manifest(man, os.path.join(out, "hs_manifest.tsv"))
    return man


# ------------------------------------------------------------------ sentiment
def _pool_json(cfg, paths):
    rows = []
    for p in paths:
        for r in load_json(P(cfg, p)):
            rows.append({"text": r["text"], "label": r["label"], "src_file": os.path.basename(p)})
    df = pd.DataFrame(rows)
    df["id"] = (np.arange(len(df)) + 1).astype(str)
    return df


def build_telecom(cfg, out, report, hs_man):
    d = cfg["data"]
    df = _pool_json(cfg, d["telecom_json"])
    kept = dedup(df, "label", "id", report, "telecom_dedup")
    seed = d["split_seed"]
    idx = np.arange(len(kept)); y = kept.label.values
    trv, te = train_test_split(idx, test_size=d["telecom_test_frac"], random_state=seed, stratify=y)
    tr, va = train_test_split(trv, test_size=d["telecom_val_frac_of_train"], random_state=seed, stratify=y[trv])
    s = np.empty(len(kept), dtype=object); s[tr], s[va], s[te] = "train", "val", "test"
    kept["split"] = s
    hs_eval = set(hs_man.uid[hs_man.split.isin(["val", "test"])])
    hs_all = set(hs_man.uid)
    kept["collides_hs_any"] = kept.uid.isin(hs_all).astype(int)
    kept["mtl_exclude"] = kept.uid.isin(hs_eval).astype(int)
    report["telecom_split"] = {x: {"n": int((kept.split == x).sum()), "label": dict(Counter(kept.label[kept.split == x]))}
                               for x in ("train", "val", "test")}
    report["telecom_split"]["uids_also_in_hs_manifest"] = int(kept.collides_hs_any.sum())
    report["telecom_split"]["train_rows_excluded_from_mtl_aux(hs_val_test_collision)"] = int(
        ((kept.split == "train") & (kept.mtl_exclude == 1)).sum())
    man = kept[["uid", "id", "split", "label", "src_file", "collides_hs_any", "mtl_exclude", "raw_sha256", "text"]]
    man = man.rename(columns={"id": "pool_id"})
    M.write_manifest(man, os.path.join(out, "telecom_manifest.tsv"))
    return man


def build_foldset(cfg, out, report, name):
    """10-fold CV datasets (SudSenti3, SudSenti2): StratifiedKFold(10, shuffle,
    seed 42) on the pooled de-duplicated rows; inside each fold 1/9 of train+val is held out for validation."""
    d = cfg["data"]
    df = _pool_json(cfg, d[f"{name}_json"])
    kept = dedup(df, "label", "id", report, f"{name}_dedup")
    seed = d["split_seed"]; y = kept.label.values
    grouped = name in (d.get("group_by_mhamed") or [])
    if grouped:
        # connected groups: rows are unique per canonical key after dedup, so the components are the
        # classes of identical CNN input (mhamed); a whole group always lands in one partition.
        mh = kept.text.map(lambda t: sha256(preprocess(t, "mhamed")))
        gid = mh.map({h: i for i, h in enumerate(pd.unique(mh))}).values
        sizes = pd.Series(gid).value_counts()
        glab = pd.Series(y).groupby(gid).nunique()
        report[f"{name}_mhamed_groups"] = {
            "groups": int(len(sizes)), "groups_size_gt1": int((sizes > 1).sum()),
            "rows_in_groups_size_gt1": int(sizes[sizes > 1].sum()), "largest_group": int(sizes.max()),
            "groups_with_mixed_labels_kept_together": int((glab > 1).sum()),
            "rows_with_empty_mhamed_input": int((kept.text.map(lambda t: preprocess(t, "mhamed").strip()) == "").sum()),
            "fold_method": "StratifiedGroupKFold(10, shuffle, seed) + inner StratifiedGroupKFold(round(1/val_frac)) first split as val"}
        skf = StratifiedGroupKFold(n_splits=d[f"{name}_folds"], shuffle=True, random_state=seed)
        outer = skf.split(np.zeros(len(y)), y, gid)
    else:
        skf = StratifiedKFold(n_splits=d[f"{name}_folds"], shuffle=True, random_state=seed)
        outer = skf.split(np.zeros(len(y)), y)
    folds = {}
    fold_of = np.full(len(kept), -1)
    for f, (trv, te) in enumerate(outer):
        if grouped:
            k = int(round(1 / d[f"{name}_val_frac_of_trainval"]))
            inner = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            a_, b_ = next(inner.split(np.zeros(len(trv)), y[trv], gid[trv]))
            tr, va = trv[a_], trv[b_]
        else:
            tr, va = train_test_split(trv, test_size=d[f"{name}_val_frac_of_trainval"], random_state=seed, stratify=y[trv])
        fold_of[te] = f
        folds[str(f)] = {"train": kept.uid.values[tr].tolist(), "val": kept.uid.values[va].tolist(),
                         "test": kept.uid.values[te].tolist()}
        a, b, c = map(set, (folds[str(f)]["train"], folds[str(f)]["val"], folds[str(f)]["test"]))
        assert not (a & b) and not (a & c) and not (b & c)
    kept["fold"] = fold_of
    report[f"{name}_folds"] = {f: {k: len(v) for k, v in folds[f].items()} for f in folds}
    report[f"{name}_label_dist"] = dict(Counter(kept.label))
    man = kept[["uid", "id", "fold", "label", "src_file", "raw_sha256", "text"]].rename(columns={"id": "pool_id"})
    M.write_manifest(man, os.path.join(out, f"{name}_manifest.tsv"))
    dump_json(folds, os.path.join(out, f"{name}_folds.json"))
    return man


def build_sudsenti3(cfg, out, report):
    return build_foldset(cfg, out, report, "sudsenti3")


def fold_assertions(cfg, out, name):
    """All folds, every representation (strict modes raise; mhamed collisions are reported)."""
    df = M.read_manifest(os.path.join(out, f"{name}_manifest.tsv")).set_index("uid", drop=False)
    folds = load_json(os.path.join(out, f"{name}_folds.json"))
    res = {}
    for f, parts in folds.items():
        tagged = pd.concat([df.loc[parts[s]].assign(part=s) for s in ("train", "val", "test")])
        res[f] = M.assert_disjoint(tagged, "part",
                                   fatal_all=name in (cfg["data"].get("fail_closed_all_representations") or []))
    oof = sorted(u for p in folds.values() for u in p["test"])
    assert oof == sorted(df.uid) and len(set(oof)) == len(oof), "out-of-fold test sets must partition the dataset"
    return res


def addendum(cfg, out, name):
    """Add one fold dataset to an already frozen Phase-0 directory without touching any existing file:
    verify every existing checksum, build, assert all folds, append to the report, rewrite CHECKSUMS."""
    M.verify_checksums(out)                               # existing frozen files are byte-identical
    if os.path.exists(os.path.join(out, f"{name}_manifest.tsv")):
        sys.exit(f"{name} already frozen in {out}")
    before = open(os.path.join(out, "CHECKSUMS.sha256")).read()
    rep = load_json(os.path.join(out, "phase0_report.json"))
    part = {}
    build_foldset(cfg, out, part, name)
    part["assertions"] = {name: fold_assertions(cfg, out, name)}
    inputs = {p: sha256_file(P(cfg, p)) for p in cfg["data"][f"{name}_json"]}
    for k, v in part.items():
        if k == "assertions":
            rep.setdefault("assertions", {}).update(v)
        else:
            rep[k] = v
    rep["input_sha256"].update(inputs)
    rep.setdefault("addenda", []).append({
        "dataset": name, "env": env_info(), "code_sha256": code_hash(), "config": cfg["_config_path"],
        "input_sha256": inputs, "previous_CHECKSUMS": before.splitlines(),
        "note": "additive: previously frozen files verified unchanged before and after"})
    dump_json(rep, os.path.join(out, "phase0_report.json"))
    M.write_checksums(out)
    old = dict(l.split("  ")[::-1] for l in before.splitlines() if l.strip())
    new = dict(l.split("  ")[::-1] for l in open(os.path.join(out, "CHECKSUMS.sha256")).read().splitlines() if l.strip())
    changed = [f for f in old if f != "phase0_report.json" and new.get(f) != old[f]]
    assert not changed, f"addendum changed frozen files: {changed}"
    print(f"{name} added to {out}:", rep[f"{name}_dedup"], rep[f"{name}_label_dist"])
    return rep


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="override frozen_dir (smoke tests)")
    ap.add_argument("--limit", type=int, default=None, help="apply LFs to first N rows only (smoke; skips 40k assert)")
    ap.add_argument("--addendum", choices=["sudsenti2"], help="add one fold dataset to the frozen dir (additive)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = a.out or P(cfg, cfg["frozen_dir"])
    if a.addendum:
        return addendum(cfg, out, a.addendum)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "CHECKSUMS.sha256")) and a.out is None:
        sys.exit(f"{out} is frozen (CHECKSUMS.sha256 exists); delete the directory deliberately to rebuild.")
    d = cfg["data"]
    inputs = [d["hs_resolved"], d["hs_corpus_json"], d["hs_3class_tsv"], d["hs_binary_tsv"], d["ws_tsv"],
              d["snorkel_module"], *d["telecom_json"], *d["sudsenti3_json"], *d.get("sudsenti2_json", [])]
    report = {"canon_version": CANON_VERSION, "alg3_version": ALG3_VERSION, "env": env_info(),
              "code_sha256": code_hash(), "config": cfg["_config_path"],
              "input_sha256": {p: sha256_file(P(cfg, p)) for p in inputs}}
    hs = build_hs(cfg, out, report, a.limit)
    tel = build_telecom(cfg, out, report, hs)
    ss3 = build_sudsenti3(cfg, out, report)
    if d.get("sudsenti2_json"):
        build_foldset(cfg, out, report, "sudsenti2")
    # ---- cross-partition assertions under every model-input representation
    report["assertions"] = {
        "hs": M.assert_disjoint(hs, "split", ("train", "val", "test")),
        "telecom": M.assert_disjoint(tel, "split", ("train", "val", "test")),
        **{ds: fold_assertions(cfg, out, ds) for ds in ("sudsenti3", "sudsenti2") if d.get(f"{ds}_json")},
    }
    dump_json(report, os.path.join(out, "phase0_report.json"))
    M.write_checksums(out)
    print(f"Phase 0 done -> {out}")
    for k in ("hs_dedup", "hs_split", "telecom_dedup", "sudsenti3_dedup"):
        print(k, report[k])
    print("lf label changes:", report["lf_weights"]["label_changes_train_only_vs_old"])
    return report


if __name__ == "__main__":
    main()
