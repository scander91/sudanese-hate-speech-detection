"""Job registry: every run of the experiment, its dependencies and a GPU-time estimate.

Job ids are stable strings, e.g.  base__marbertv2__binary__s42,
sent__qarib__sudsenti3__f3__s42, distill__3class__s44, mtl__marbert__binary__a0.7__s42,
cnn__scm_mma__hs_3class__s42, post__teacher__binary, post__ens__binary, test__base__marbertv2__binary__s42,
test__mtl__marbert__binary__s42 (selected alpha), post__enstest__binary, post__report.
"""
from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict

from .common import P


def _j(id_, kind, gpu=True, deps=(), **kw):
    d = OrderedDict(id=id_, kind=kind, gpu=gpu, deps=list(deps))
    d.update(kw)
    return d


def _folds(cfg, ds):
    sub = cfg["data"].get("fold_subset")
    return [f for f in range(cfg["data"][f"{ds}_folds"]) if sub is None or f in sub]


TRAIN_KINDS = ("base", "sent", "bilstm", "distill", "distill4", "mtl", "cnn")


def test_id(train_id: str) -> str:
    """Test-materialisation job of a training job (MTL: alpha-free, the selected alpha is read at run time)."""
    if train_id.startswith("mtl__"):
        p = train_id.split("__")                       # mtl, enc, task, aX, sY
        return "__".join(["test", p[0], p[1], p[2], p[4]])
    return "test__" + train_id


def all_jobs(cfg: dict) -> list[dict]:
    """Chronological test firewall: training jobs write train/val predictions and a best-state
    checkpoint only.  Selections are frozen by CPU post jobs.  `test__*` jobs, which depend on every
    selection their run is a candidate in, are the only jobs that ever predict the test partition."""
    J = []
    seeds = cfg["seeds"]
    models = list(cfg["models"])
    hs_tasks = cfg.get("hs_tasks", ["binary", "3class"])
    # Phase 1a: transformer baselines on hate speech
    for t in hs_tasks:
        for m in models:
            for s in seeds:
                J.append(_j(f"base__{m}__{t}__s{s}", "base", model=m, task=t, seed=s))
    # Phase 1b: BiLSTM hybrid
    for t in hs_tasks:
        for s in seeds:
            J.append(_j(f"bilstm__{t}__s{s}", "bilstm", task=t, seed=s, model=cfg["bilstm"]["encoder"]))
    # Phase 1c: teacher selection (CPU, validation only) then distillation.
    #   distill  = primary: teacher chosen from all candidate baselines;
    #   distill4 = reproduction sensitivity: teacher chosen from the historical 4-candidate set
    #              (an alias of `distill` when both sets choose the same teacher).
    for t in hs_tasks:
        cands = sorted(set(cfg["distill"]["teacher_candidates"]) | set(cfg["distill"].get("repro_teacher_candidates") or []))
        J.append(_j(f"post__teacher__{t}", "post_teacher", gpu=False, task=t,
                    deps=[f"base__{m}__{t}__s{s}" for m in cands for s in seeds]))
        for s in seeds:
            J.append(_j(f"distill__{t}__s{s}", "distill", task=t, seed=s, model=cfg["distill"]["student"],
                        deps=[f"post__teacher__{t}"]))
            if cfg["distill"].get("repro_teacher_candidates"):
                J.append(_j(f"distill4__{t}__s{s}", "distill4", task=t, seed=s, model=cfg["distill"]["student"],
                            deps=[f"post__teacher__{t}"]))
    # Phase 1d: MTL grid (alpha chosen on validation afterwards; only the chosen alpha is tested)
    for t in hs_tasks:
        for e in cfg["mtl"]["encoders"]:
            for a in cfg["mtl"]["alphas"]:
                for s in seeds:
                    J.append(_j(f"mtl__{e}__{t}__a{a}__s{s}", "mtl", model=e, task=t, alpha=a, seed=s))
    # Phase 1e: sentiment target-task runs
    for m in cfg["sentiment"]["models"]:
        for s in seeds:
            if "telecom" in cfg["sentiment"]["datasets"]:
                J.append(_j(f"sent__{m}__telecom__s{s}", "sent", model=m, task="telecom", seed=s))
            for ds in ("sudsenti3", "sudsenti2"):
                if ds in cfg["sentiment"]["datasets"]:
                    for f in _folds(cfg, ds):
                        J.append(_j(f"sent__{m}__{ds}__f{f}__s{s}", "sent", model=m, task=ds, fold=f, seed=s))
    # Phase 1f: CNN baselines
    for cm in cfg["cnn"]["models"]:
        for ds in cfg["cnn"]["datasets"]:
            for s in seeds:
                if ds in ("sudsenti3", "sudsenti2"):
                    for f in _folds(cfg, ds):
                        J.append(_j(f"cnn__{cm}__{ds}__f{f}__s{s}", "cnn", model=cm, task=ds, fold=f, seed=s))
                else:
                    J.append(_j(f"cnn__{cm}__{ds}__s{s}", "cnn", model=cm, task=ds, seed=s))
    # Post-hoc validation-only selections (CPU) -- frozen before any candidate's test prediction exists
    for t in hs_tasks:
        J.append(_j(f"post__ens__{t}", "post_ens", gpu=False, task=t,
                    deps=[f"base__{m}__{t}__s{s}" for m in cfg["ensemble"]["candidates"] for s in seeds]))
        J.append(_j(f"post__mtlsel__{t}", "post_mtlsel", gpu=False, task=t,
                    deps=[j["id"] for j in J if j["kind"] == "mtl" and j["task"] == t]))
    if "telecom" in cfg["sentiment"]["datasets"]:
        J.append(_j("post__ens__telecom", "post_ens", gpu=False, task="telecom",
                    deps=[j["id"] for j in J if j["kind"] == "sent" and j["task"] == "telecom"]))
    # Global freeze barrier: one CPU job depending on EVERY selection writes
    # the immutable _selection/FROZEN.json; every test__ job depends on it and asserts it before any test IO.
    sel_jobs = [j["id"] for j in J if j["kind"] in ("post_teacher", "post_ens", "post_mtlsel")]
    J.append(_j("post__freeze", "post_freeze", gpu=False, deps=sel_jobs))
    # Test materialisation (GPU): one per training run; MTL only for the selected alpha
    tests = {}
    for j in [j for j in J if j["kind"] in TRAIN_KINDS]:
        tid = test_id(j["id"])
        if j["kind"] == "base":
            deps = [j["id"], f"post__teacher__{j['task']}", f"post__ens__{j['task']}"]
        elif j["kind"] == "sent" and j["task"] == "telecom":
            deps = [j["id"], "post__ens__telecom"]
        elif j["kind"] == "mtl":
            deps = [f"post__mtlsel__{j['task']}"]
        elif j["kind"] == "distill4":
            deps = [j["id"], test_id(f"distill__{j['task']}__s{j['seed']}")]   # alias case copies the primary test
        else:
            deps = [j["id"]]
        deps = ["post__freeze"] + deps
        if tid in tests:
            continue
        kw = {k: v for k, v in j.items() if k not in ("id", "kind", "gpu", "deps", "alpha")}
        tests[tid] = _j(tid, "test", deps=deps, train_kind=j["kind"],
                        train_id=None if j["kind"] == "mtl" else j["id"], **kw)
    J.extend(tests.values())
    # Post-freeze interpretability (GPU): SHAP+LIME of the selected best baseline; LIME MTL vs baseline
    if cfg.get("xai"):
        s0 = seeds[0]
        enc = cfg["xai"]["compare_encoder"]
        for t in hs_tasks:
            J.append(_j(f"xai__{t}", "xai", task=t, seed=s0,
                        deps=["post__freeze"] + [f"test__base__{m}__{t}__s{s0}" for m in models]))
            if enc in cfg["mtl"]["encoders"] and enc in models:
                J.append(_j(f"xai_mtlcmp__{t}", "xai_mtlcmp", task=t, seed=s0, model=enc,
                            deps=["post__freeze", f"test__base__{enc}__{t}__s{s0}", f"test__mtl__{enc}__{t}__s{s0}"]))
    for t in hs_tasks + (["telecom"] if "telecom" in cfg["sentiment"]["datasets"] else []):
        if t == "telecom":
            mem = [f"test__sent__{m}__telecom__s{s}" for m in cfg["sentiment"]["models"] for s in seeds]
        else:
            ms = set(cfg["ensemble"]["candidates"]) | set(cfg["ensemble"].get("also_report_fixed_paper_members", {}).get(t, []))
            mem = [f"test__base__{m}__{t}__s{s}" for m in sorted(ms) for s in seeds]
        J.append(_j(f"post__enstest__{t}", "post_enstest", gpu=False, task=t, deps=[f"post__ens__{t}"] + mem))
    J.append(_j("post__report", "post_report", gpu=False, deps=[j["id"] for j in J if j["kind"] != "post_report"]))
    ids = [j["id"] for j in J]
    assert len(ids) == len(set(ids))
    assert all("post__freeze" in j["deps"] for j in J if j["kind"] == "test"), "test job without the freeze barrier"
    assert set(sel_jobs) <= set(next(j for j in J if j["id"] == "post__freeze")["deps"])
    known = set(ids)
    assert all(d in known for j in J for d in j["deps"]), "dangling dependency"
    return J


# ------------------------------------------------------------------ estimates
def est_minutes(cfg: dict) -> dict:
    """Per-run wall-minute estimates read from timing files under project_root/results/ (scheduling only;
    missing files fall back to the defaults in job_minutes)."""
    root = cfg["project_root"]
    est = {}

    def rd(p, keys):
        try:
            r = json.load(open(p))
        except Exception:
            return None
        for k in keys:
            if k in r:
                return float(r[k]) / 60.0
        return None
    for m in cfg["models"]:
        v = [rd(os.path.join(root, f"results/hate_speech_models/{m}_{t}/results.json"), ["train_time_seconds"])
             for t in ("binary", "3class")]
        v = [x for x in v if x]
        est[f"base:{m}"] = sum(v) / len(v) if v else None
        tel = rd(os.path.join(root, f"results/sentiment_evaluation/{m}_telecom/results.json"), ["training_time"])
        est[f"sent_telecom:{m}"] = tel
        for ds in ("sudsenti3", "sudsenti2"):
            v = rd(os.path.join(root, f"results/sentiment_evaluation/{m}_{ds}/results.json"), ["training_time"])
            est[f"sent_{ds}_fold:{m}"] = v / cfg["data"][f"{ds}_folds"] if v else None
    b = [rd(os.path.join(root, f"results/hate_speech_hybrid/bilstm_sudabert_{t}/results.json"), ["train_time_seconds"]) for t in ("binary", "3class")]
    d = [rd(os.path.join(root, f"results/hate_speech_hybrid/distill_sudabert_{t}/results.json"), ["train_time_seconds"]) for t in ("binary", "3class")]
    b, d = [x for x in b if x], [x for x in d if x]
    est["bilstm"] = sum(b) / len(b) if b else None
    est["distill"] = sum(d) / len(d) if d else None
    mt = [rd(p, ["training_time_seconds"]) for p in glob.glob(os.path.join(root, "results/mtl_hate_sentiment/mtl_*/results.json"))]
    mt = [x for x in mt if x]
    est["mtl"] = sum(mt) / len(mt) if mt else None
    try:   # CNN timings; fold datasets are a 10-fold total
        for x in json.load(open(os.path.join(root, "results/cnn_mhamed_comparison/all_results_summary.json"))):
            n = len(x.get("fold_results") or []) or 1
            est[f"cnn:{x['model']}:{x['dataset']}"] = float(x["training_time"]) / 60.0 / n
    except Exception:
        pass
    return est


def job_minutes(job: dict, est: dict) -> float:
    k = job["kind"]
    if k == "base":
        return est.get(f"base:{job['model']}") or 10.0
    if k == "sent":
        key = f"sent_{job['task']}{'_fold' if job['task'] in ('sudsenti3', 'sudsenti2') else ''}:{job['model']}"
        return est.get(key) or 3.0
    if k in ("bilstm", "distill", "mtl"):
        return est.get(k) or 10.0
    if k == "distill4":
        return est.get("distill") or 10.0
    if k == "cnn":
        return est.get(f"cnn:{job['model']}:{job['task']}") or 2.0
    if k == "test":
        return 1.0
    if k in ("xai", "xai_mtlcmp"):
        return 15.0
    return 0.0
