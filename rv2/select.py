"""Validation-only selection (teacher, ensemble members + rule, MTL alpha, best single model).

Every selector reads ONLY metrics_val.json / preds_val.npz of training-run directories (a guard
refuses any test path).  The selection is written to runs/_selection/<name>.json under a
per-selection lock, write-once: a later call must reproduce the same choice AND the same
candidate-input hash (DONE fingerprints + sha256 of every validation file read).  Test
predictions of the candidates are materialised only by `test__*` jobs, which depend on these
selection jobs (chronological firewall), and ensembles are materialised by post__enstest__*.
"""
from __future__ import annotations

import itertools
import os
import shutil
import socket
import time

import numpy as np

from .common import dump_json, load_json, sha256_file, sha256_str
from .metrics import compute_metrics, macro_f1


def sel_dir(runs_root):
    d = os.path.join(runs_root, "_selection")
    os.makedirs(d, exist_ok=True)
    return d


def _sel_lock(runs_root, name, timeout=600):
    """Atomic mkdir lock with an owner token.  Never broken automatically: a waiter gives up after
    `timeout` s with an error naming the owner; a lock left by a dead writer is cleared by an operator."""
    lk = os.path.join(sel_dir(runs_root), f".lock_{name}")
    tok = f"{socket.gethostname()}.{os.getpid()}.{time.time_ns()}"
    t0 = time.time()
    while True:
        try:
            os.mkdir(lk)
            dump_json({"host": socket.gethostname(), "pid": os.getpid(), "time": time.time(), "token": tok},
                      os.path.join(lk, "owner.json"))
            return lk, tok
        except FileExistsError:
            if time.time() - t0 > timeout:
                try:
                    o = load_json(os.path.join(lk, "owner.json"))
                except Exception:
                    o = {}
                raise RuntimeError(f"selection lock {lk} held for >{timeout}s by {o.get('host')}:{o.get('pid')}; "
                                   "verify that process is dead, then remove the lock directory by hand")
            time.sleep(0.5)


def inputs_hash(runs_root, run_ids, files=("metrics_val.json",)):
    """Hash of every candidate input a selector reads: DONE fingerprint + sha256 of each validation file."""
    parts = []
    for r in sorted(run_ids):
        d = os.path.join(runs_root, r)
        parts.append(r + ":" + str(load_json(os.path.join(d, "DONE")).get("fingerprint")))
        for fn in files:
            parts.append(fn + ":" + sha256_file(os.path.join(d, fn)))
    return sha256_str("\n".join(parts))


def write_selection(runs_root, name, sel):
    """Atomic, locked, write-once.  A second call must agree on `chosen` and on `inputs_sha256`."""
    p = os.path.join(sel_dir(runs_root), f"{name}.json")
    assert "inputs_sha256" in sel, f"selection {name} lacks a candidate-input hash"
    lk, tok = _sel_lock(runs_root, name)
    try:
        if os.path.exists(p):
            old = load_json(p)
            if old.get("inputs_sha256") != sel["inputs_sha256"]:
                raise RuntimeError(f"selection {name}: candidate inputs changed since it was frozen "
                                   f"({old.get('inputs_sha256')} -> {sel['inputs_sha256']})")
            if old.get("chosen") != sel.get("chosen"):
                raise RuntimeError(f"selection {name} changed: {old.get('chosen')} -> {sel.get('chosen')}")
            return old
        sel = dict(sel, test_data_used=False, frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   frozen_by=f"{socket.gethostname()}:{os.getpid()}")
        dump_json(sel, p)
        return sel
    finally:
        _release_sel_lock(lk, tok)


def _release_sel_lock(lk, tok):
    """Delete only our own lock (token compared); rename first, then delete the private copy (a plain rmtree
    could leave an empty ownerless dir behind when a waiter holds owner.json open on NFS)."""
    try:
        if load_json(os.path.join(lk, "owner.json")).get("token") != tok:
            return
    except Exception:
        return
    priv = f"{lk}.released.{tok}"
    try:
        os.rename(lk, priv)
    except OSError:
        return
    shutil.rmtree(priv, ignore_errors=True)


def read_selection(runs_root, name):
    p = os.path.join(runs_root, "_selection", f"{name}.json")
    if not os.path.exists(p):
        raise RuntimeError(f"selection {name} is not frozen yet ({p})")
    return load_json(p)


def _done(runs_root, rid):
    return os.path.exists(os.path.join(runs_root, rid, "DONE"))


def mean_val_f1(runs_root, run_ids):
    v = []
    for r in run_ids:
        if not _done(runs_root, r):
            raise RuntimeError(f"selection needs finished run {r}")
        v.append(load_json(os.path.join(runs_root, r, "metrics_val.json"))["macro_f1"])
    return float(np.mean(v))


def _load(runs_root, rid, split):
    """Selectors (split='val') read the training run; only the ensemble materialiser reads 'test',
    and then from the test__ job directory (which exists only after the selections are frozen)."""
    d = os.path.join(runs_root, ("test__" + rid) if split == "test" else rid)
    z = np.load(os.path.join(d, f"preds_{split}.npz"))
    info = load_json(os.path.join(runs_root, rid, "info.json"))
    return z, info["label_names"]


def combine(probs_list, rule, weights=None):
    P = np.stack(probs_list)                        # (K, N, C)
    if rule == "uniform_prob_mean":
        return P.mean(0)
    if rule == "valf1_weighted_prob_mean":
        w = np.asarray(weights, float); w = w / w.sum()
        return np.tensordot(w, P, axes=1)
    if rule == "hard_majority":                     # ties -> uniform probability mean
        K, N, C = P.shape
        votes = np.zeros((N, C))
        for k in range(K):
            votes[np.arange(N), P[k].argmax(1)] += 1
        tie = (votes == votes.max(1, keepdims=True)).sum(1) > 1
        out = votes / K
        out[tie] = P.mean(0)[tie]
        return out
    raise ValueError(rule)


def _members_preds(runs_root, members, task, seed, split, allow_test=False):
    assert split != "test" or allow_test, "test predictions requested inside a selector"
    probs, uids, y, names = [], None, None, None
    for m in members:
        z, ln = _load(runs_root, _rid(m, task, seed), split)
        if names is None:
            names, uids, y = ln, list(z["uids"]), z["y"]
        assert ln == names, f"label order mismatch for {m}"
        assert list(z["uids"]) == uids, f"uid order mismatch for {m}"
        assert np.array_equal(z["y"], y), f"label mismatch for {m}"
        probs.append(z["probs"])
    return probs, uids, y, names


def _rid(model, task, seed):
    return f"{'sent' if task == 'telecom' else 'base'}__{model}__{task}__s{seed}"


def ensemble_select(cfg, runs_root, task, log=print):
    seeds = cfg["seeds"]
    if task == "telecom":
        cands, k = cfg["sentiment"]["models"], cfg["sentiment"]["telecom_ensemble"]["size"]
        rules, how = [cfg["sentiment"]["telecom_ensemble"]["rule"]], "top_k_val"
    else:
        cands, k = cfg["ensemble"]["candidates"], cfg["ensemble"]["size"]
        rules, how = cfg["ensemble"]["rules"], cfg["ensemble"]["member_selection"]
    single = {m: mean_val_f1(runs_root, [_rid(m, task, s) for s in seeds]) for m in cands}
    ranked = sorted(cands, key=lambda m: -single[m])
    subsets = [tuple(ranked[:k])] if how == "top_k_val" else list(itertools.combinations(cands, k))
    table = []
    for sub in subsets:
        for rule in rules:
            f1s = []
            for s in seeds:
                probs, _, y, names = _members_preds(runs_root, sub, task, s, "val")
                f1s.append(macro_f1(y, combine(probs, rule, [single[m] for m in sub]).argmax(1), len(names)))
            table.append({"members": list(sub), "rule": rule, "mean_val_macro_f1": float(np.mean(f1s))})
    best = max(table, key=lambda r: r["mean_val_macro_f1"])      # first maximum wins (deterministic order)
    ih = inputs_hash(runs_root, [_rid(m, task, s) for m in cands for s in seeds], ("metrics_val.json", "preds_val.npz"))
    sel = {"what": f"ensemble ({task})", "criterion": "mean validation macro-F1 over seeds",
           "single_model_mean_val_f1": single, "member_selection": how, "candidates_scored": table,
           "chosen": {"members": best["members"], "rule": best["rule"]},
           "member_weights_if_weighted": {m: single[m] for m in best["members"]}, "inputs_sha256": ih}
    sel = write_selection(runs_root, f"ensemble_{task}", sel)
    write_selection(runs_root, f"best_single_{task}", {
        "what": f"best single model ({task})", "criterion": "mean validation macro-F1 over seeds",
        "scores": single, "chosen": ranked[0], "inputs_sha256": ih})
    log(f"ensemble({task}) = {sel['chosen']}")
    return sel


def ensemble_materialise(cfg, runs_root, task, members, rule, name, weights=None):
    """Write per-seed pseudo-runs ens*/ with val+test predictions and metrics (test opened only here)."""
    for s in cfg["seeds"]:
        final = os.path.join(runs_root, f"{name}__{task}__s{s}")
        if os.path.exists(os.path.join(final, "DONE")):
            continue                                   # published by an earlier attempt (deterministic content)
        rd = os.path.join(runs_root, f".stage.{os.getpid()}.{time.time_ns()}.{name}__{task}__s{s}")
        os.makedirs(rd)
        for split in ("val", "test"):
            probs, uids, y, names = _members_preds(runs_root, members, task, s, split, allow_test=True)
            p = combine(probs, rule, weights)
            np.savez_compressed(os.path.join(rd, f"preds_{split}.npz"), uids=np.array(uids), y=y, probs=p.astype(np.float32),
                                logits=np.log(np.clip(p, 1e-12, 1)).astype(np.float32))
            dump_json(compute_metrics(y, p.argmax(1), names), os.path.join(rd, f"metrics_{split}.json"))
        dump_json({"label_names": names, "members": members, "rule": rule, "seed": s}, os.path.join(rd, "info.json"))
        dump_json({"members": members, "rule": rule}, os.path.join(rd, "DONE"))
        try:
            os.rename(rd, final)                        # atomic publish; loses cleanly to a concurrent publisher
        except OSError:
            shutil.rmtree(rd, ignore_errors=True)
            if not os.path.exists(os.path.join(final, "DONE")):
                raise


def run_post_ens(cfg, runs_root, task, log=print):
    """Selection only (validation); the test materialisation is post__enstest__<task>."""
    return ensemble_select(cfg, runs_root, task, log)


def run_post_enstest(cfg, runs_root, task, log=print):
    assert_frozen(runs_root)
    sel = read_selection(runs_root, f"ensemble_{task}")
    ch = sel["chosen"]
    w = [sel["member_weights_if_weighted"][m] for m in ch["members"]]
    ensemble_materialise(cfg, runs_root, task, ch["members"], ch["rule"], "ens", w)
    fixed = cfg["ensemble"].get("also_report_fixed_paper_members", {}).get(task)
    if fixed:   # predeclared (manuscript) members, uniform soft vote; secondary row, no selection
        ensemble_materialise(cfg, runs_root, task, fixed, "uniform_prob_mean", "ensfixed")
    log(f"ensemble({task}) test materialised: {ch}")


def run_post_mtlsel(cfg, runs_root, task, log=print):
    out = {}
    for e in cfg["mtl"]["encoders"]:
        rids = {a: [f"mtl__{e}__{task}__a{a}__s{s}" for s in cfg["seeds"]] for a in cfg["mtl"]["alphas"]}
        sc = {str(a): mean_val_f1(runs_root, rids[a]) for a in cfg["mtl"]["alphas"]}
        best = max(cfg["mtl"]["alphas"], key=lambda a: sc[str(a)])
        out[e] = write_selection(runs_root, f"mtl_alpha_{e}_{task}", {
            "what": f"MTL alpha ({e}, {task})", "criterion": "mean validation HS macro-F1 over seeds",
            "scores": sc, "chosen": best,
            "inputs_sha256": inputs_hash(runs_root, [r for a in rids for r in rids[a]])})["chosen"]
    log(f"mtl alpha ({task}) = {out}")
    return out


def _teacher_pick(cfg, runs_root, task, cands):
    scores = {m: mean_val_f1(runs_root, [f"base__{m}__{task}__s{s}" for s in cfg["seeds"]]) for m in cands}
    return max(cands, key=lambda m: scores[m]), scores          # ties -> first listed


def run_post_teacher(cfg, runs_root, task, log=print):
    """Distillation teacher, validation only (mean validation macro-F1 over ALL seeds).
    teacher_<task>  : PRIMARY, over all distill.teacher_candidates;
    teacher4_<task> : reproduction sensitivity, over the historical distill.repro_teacher_candidates."""
    out = {}
    for name, key, role in (("teacher", "teacher_candidates", "primary"),
                            ("teacher4", "repro_teacher_candidates", "reproduction sensitivity")):
        cands = cfg["distill"].get(key)
        if not cands:
            continue
        best, scores = _teacher_pick(cfg, runs_root, task, cands)
        rids = [f"base__{m}__{task}__s{s}" for m in cands for s in cfg["seeds"]]
        out[name] = write_selection(runs_root, f"{name}_{task}", {
            "what": f"distillation teacher ({task}, {role})", "criterion": "mean validation macro-F1 over seeds",
            "candidates": list(cands), "scores": scores, "chosen": best,
            "inputs_sha256": inputs_hash(runs_root, rids)})["chosen"]
    log(f"teacher({task}) = {out}")
    return out


FROZEN = "FROZEN.json"


def selection_digest(runs_root):
    """sha256 of every selection file (name + content) currently in _selection/."""
    d = sel_dir(runs_root)
    files = sorted(f for f in os.listdir(d) if f.endswith(".json") and f != FROZEN)
    return {f: sha256_file(os.path.join(d, f)) for f in files}


def run_post_freeze(cfg, runs_root, expected, log=print):
    """Global barrier: write the immutable FROZEN.json listing every selection and its sha256 (O_EXCL, write-once).
    `expected` = selection names that must exist.  A second run must find an identical record."""
    missing = [n for n in expected if not os.path.exists(os.path.join(sel_dir(runs_root), f"{n}.json"))]
    if missing:
        raise RuntimeError(f"freeze: selections missing {missing}")
    rec = {"selections": selection_digest(runs_root), "expected": sorted(expected)}
    p = os.path.join(sel_dir(runs_root), FROZEN)
    if os.path.exists(p):
        old = load_json(p)
        if old["selections"] != rec["selections"]:
            raise RuntimeError("freeze: selections differ from the existing FROZEN record")
        return old
    rec.update(frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"), frozen_by=f"{socket.gethostname()}:{os.getpid()}")
    tmp = f"{p}.tmp.{socket.gethostname()}.{os.getpid()}"
    import json
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    try:
        os.link(tmp, p)                                  # atomic create-if-absent on NFS
    finally:
        os.remove(tmp)
    os.chmod(p, 0o444)
    log(f"FROZEN {len(rec['selections'])} selections")
    return rec


def assert_frozen(runs_root):
    """Called by every test__ job before any test file is opened: FROZEN exists and every selection is unchanged."""
    p = os.path.join(runs_root, "_selection", FROZEN)
    if not os.path.exists(p):
        raise RuntimeError(f"test firewall: {p} missing -- selections are not globally frozen")
    rec = load_json(p)
    now = selection_digest(runs_root)
    if any(now.get(k) != v for k, v in rec["selections"].items()):
        raise RuntimeError("test firewall: a selection file changed after the global freeze")
    return rec


def expected_selections(cfg):
    hs = cfg.get("hs_tasks", ["binary", "3class"])
    out = []
    for t in hs:
        out += [f"teacher_{t}", f"ensemble_{t}", f"best_single_{t}"]
        if cfg["distill"].get("repro_teacher_candidates"):
            out.append(f"teacher4_{t}")
        out += [f"mtl_alpha_{e}_{t}" for e in cfg["mtl"]["encoders"]]
    if "telecom" in cfg["sentiment"]["datasets"]:
        out += ["ensemble_telecom", "best_single_telecom"]
    return out
