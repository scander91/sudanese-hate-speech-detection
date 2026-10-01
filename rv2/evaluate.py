"""Evaluation over seeds: mean +- SD, 95% CIs, per-class P/R/F1, Nadeau-Bengio corrected
resampled t-test (+ Holm), McNemar on frozen paired test predictions.

    python -m rv2.evaluate [--runs DIR]      (also run as job post__report)

Statistics
  * mean_sd / t_ci   : across run seeds (t distribution, df = r-1).
  * boot_ci          : percentile bootstrap over TEST ITEMS of the seed-averaged macro-F1
                       (items resampled jointly for all seeds), B from config.
  * nb_ttest         : Nadeau & Bengio (2003) corrected resampled t:  t = mean(d) /
                       sqrt((1/r + n_test/n_train) * var(d)), df = r-1, on paired per-seed
                       (or per fold x seed) differences.  For a fixed split with varying seeds only
                       the correction is conservative.
  * holm             : Holm (1979) step-down adjustment within each task's comparison family.
  * mcnemar          : per seed on the paired test predictions; exact binomial if b+c < 25, else
                       chi-square with continuity correction.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from scipy import stats

from .common import dump_json, load_config, load_json, P


# ------------------------------------------------------------------ primitives
def fast_macro_f1(y, p, C):
    cm = np.bincount(y * C + p, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(C), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(C), where=cm.sum(1) > 0)
    f = np.divide(2 * prec * rec, prec + rec, out=np.zeros(C), where=(prec + rec) > 0)
    return f.mean()


def mean_sd(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def t_ci(v, level=0.95):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [float(v.mean())] * 2
    h = stats.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean() - h), float(v.mean() + h)]


def boot_ci(y, preds, C, B=2000, level=0.95, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        vals[b] = np.mean([fast_macro_f1(y[i], p[i], C) for p in preds])
    a = (1 - level) / 2
    return [float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))]


def nb_ttest(a, b, n_train, n_test):
    d = np.asarray(a, float) - np.asarray(b, float)
    r = len(d)
    if r < 2:
        return {"r": r, "mean_diff": float(d.mean()), "t": None, "p": None}
    var = d.var(ddof=1)
    denom = np.sqrt((1.0 / r + n_test / n_train) * var)
    t = float(d.mean() / denom) if denom > 0 else (np.inf if d.mean() != 0 else 0.0)
    p = float(2 * stats.t.sf(abs(t), r - 1)) if np.isfinite(t) else 0.0
    return {"r": r, "mean_diff": float(d.mean()), "sd_diff": float(np.sqrt(var)), "t": t, "df": r - 1, "p": p,
            "n_train": n_train, "n_test": n_test}


def holm(pvals):
    idx = [i for i, p in enumerate(pvals) if p is not None]
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    adj = [None] * len(pvals)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * pvals[i]))
        adj[i] = run
    return adj


def mcnemar(y, pa, pb):
    ca, cb = pa == y, pb == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    if b + c == 0:
        return {"b": b, "c": c, "p": 1.0, "method": "none"}
    if b + c < 25:
        p = float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)))
        return {"b": b, "c": c, "p": p, "method": "exact_binomial"}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2_cc": float(chi2), "p": float(stats.chi2.sf(chi2, 1)), "method": "chi2_cc"}


# ------------------------------------------------------------------ collection
def _runs(rr, pattern):
    return sorted(d for d in glob.glob(os.path.join(rr, pattern)) if os.path.exists(os.path.join(d, "DONE")))


def _sel(rr, name):
    p = os.path.join(rr, "_selection", f"{name}.json")
    return load_json(p)["chosen"] if os.path.exists(p) else None


def systems_for(cfg, rr, task):
    """System -> glob of the run directories holding its TEST predictions (test__* jobs, ensemble pseudo-runs)."""
    S = {}
    if task in ("binary", "3class"):
        for m in cfg["models"]:
            S[f"base:{m}"] = f"test__base__{m}__{task}__s*"
        S["bilstm"] = f"test__bilstm__{task}__s*"
        S["distill"] = f"test__distill__{task}__s*"
        t4 = _sel(rr, f"teacher4_{task}")
        if t4 is not None:
            same = t4 == _sel(rr, f"teacher_{task}")
            S[f"distill(4-cand repro, teacher={t4}{', = primary' if same else ''})"] = f"test__distill4__{task}__s*"
        for e in cfg["mtl"]["encoders"]:
            a = _sel(rr, f"mtl_alpha_{e}_{task}")
            if a is not None:
                S[f"mtl:{e}(a={a})"] = f"test__mtl__{e}__{task}__s*"
        S["ensemble(val-selected)"] = f"ens__{task}__s*"
        S["ensemble(paper members)"] = f"ensfixed__{task}__s*"
        for cm in cfg["cnn"]["models"]:
            S[f"cnn:{cm}"] = f"test__cnn__{cm}__hs_{task}__s*"
    elif task == "telecom":
        for m in cfg["sentiment"]["models"]:
            S[f"sent:{m}"] = f"test__sent__{m}__telecom__s*"
        S["ensemble(val-selected)"] = f"ens__telecom__s*"
    elif task in ("sudsenti3", "sudsenti2"):
        for m in cfg["sentiment"]["models"]:
            S[f"sent:{m}"] = f"test__sent__{m}__{task}__f*__s*"
        for cm in cfg["cnn"]["models"]:
            S[f"cnn:{cm}"] = f"test__cnn__{cm}__{task}__f*__s*"
    return S


def _load_system(rr, pattern):
    out = []
    for d in _runs(rr, pattern):
        mt = load_json(os.path.join(d, "metrics_test.json"))
        z = np.load(os.path.join(d, "preds_test.npz"))
        out.append({"run": os.path.basename(d), "metrics": mt, "y": z["y"], "pred": z["probs"].argmax(1),
                    "uids": list(z["uids"])})
    return out


def summarise(runs, cfg):
    if not runs:
        return None
    names = runs[0]["metrics"]["label_names"]
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    s = {"n_runs": len(runs), "runs": [r["run"] for r in runs]}
    for k in keys:
        v = [r["metrics"][k] for r in runs]
        m, sd = mean_sd(v)
        s[k] = {"mean": m, "sd": sd, "t_ci95": t_ci(v, cfg["evaluation"]["ci_level"])}
    s["per_class"] = {n: {k: mean_sd([r["metrics"]["per_class"][n][k] for r in runs])
                          for k in ("precision", "recall", "f1")} for n in names}
    same_items = all(r["uids"] == runs[0]["uids"] for r in runs)
    if same_items:
        s["macro_f1"]["boot_ci95_items"] = boot_ci(runs[0]["y"], [r["pred"] for r in runs], len(names),
                                                   cfg["evaluation"]["bootstrap_B"], cfg["evaluation"]["ci_level"])
    return s


def _key(run_name):
    """pairing key: seed (and fold) suffix of a run id."""
    parts = run_name.split("__")
    return "__".join(p for p in parts if p.startswith("s") and p[1:].isdigit() or p.startswith("f") and p[1:].isdigit())


def compare(cfg, A, B, n_train, n_test):
    ka = {_key(r["run"]): r for r in A}
    kb = {_key(r["run"]): r for r in B}
    common = sorted(set(ka) & set(kb))
    res = nb_ttest([ka[k]["metrics"]["macro_f1"] for k in common], [kb[k]["metrics"]["macro_f1"] for k in common],
                   n_train, n_test)
    mc = []
    for k in common:
        if ka[k]["uids"] == kb[k]["uids"]:
            mc.append(dict(pair=k, **mcnemar(ka[k]["y"], ka[k]["pred"], kb[k]["pred"])))
    res["mcnemar_per_pair"] = mc
    res["paired_keys"] = common
    return res


def comparisons_for(cfg, rr, task):
    C = []
    best = _sel(rr, f"best_single_{task}")
    if task in ("binary", "3class"):
        if best:
            C.append(("ensemble(val-selected)", f"base:{best}"))
            for cm in cfg["cnn"]["models"]:
                C.append((f"base:{best}", f"cnn:{cm}"))
        C.append(("distill", f"base:{cfg['distill']['student']}"))
        C.append(("bilstm", f"base:{cfg['bilstm']['encoder']}"))
        for e in cfg["mtl"]["encoders"]:
            a = _sel(rr, f"mtl_alpha_{e}_{task}")
            if a is not None:
                C.append((f"mtl:{e}(a={a})", f"base:{e}"))
    elif task == "telecom" and best:
        C.append(("ensemble(val-selected)", f"sent:{best}"))
    return C


def build_report(cfg, rr, log=print):
    from .manifest import read_manifest
    frozen = cfg.get("_frozen_override") or P(cfg, cfg["frozen_dir"])
    hs = read_manifest(os.path.join(frozen, "hs_manifest.tsv"))
    tel = read_manifest(os.path.join(frozen, "telecom_manifest.tsv"))
    sizes = {"binary": ((hs.split == "train").sum(), (hs.split == "test").sum()),
             "3class": ((hs.split == "train").sum(), (hs.split == "test").sum()),
             "telecom": ((tel.split == "train").sum(), (tel.split == "test").sum())}
    for ds in ("sudsenti3", "sudsenti2"):
        fp = os.path.join(frozen, f"{ds}_folds.json")
        if os.path.exists(fp):
            f0 = load_json(fp)["0"]
            sizes[ds] = (len(f0["train"]), len(f0["test"]))
    report = {"selections": {os.path.basename(p)[:-5]: load_json(p)
                             for p in glob.glob(os.path.join(rr, "_selection", "*.json"))}}
    md = ["# Results (test partition; mean ± SD over seeds)\n"]
    for task in [t for t in ("binary", "3class", "telecom", "sudsenti3", "sudsenti2") if t in sizes]:
        S = systems_for(cfg, rr, task)
        loaded = {k: _load_system(rr, v) for k, v in S.items()}
        summ = {k: summarise(v, cfg) for k, v in loaded.items() if v}
        comps = []
        for a, b in comparisons_for(cfg, rr, task):
            if loaded.get(a) and loaded.get(b):
                comps.append(dict(a=a, b=b, **compare(cfg, loaded[a], loaded[b], int(sizes[task][0]), int(sizes[task][1]))))
        adj = holm([c["p"] for c in comps])
        for c, p in zip(comps, adj):
            c["p_holm"] = p
        report[task] = {"systems": summ, "comparisons": comps, "n_train": int(sizes[task][0]),
                        "n_test": int(sizes[task][1])}
        if summ:
            md.append(f"\n## {task}\n\n| system | runs | Acc | P | R | macro-F1 | 95% CI (seeds) |\n|---|---|---|---|---|---|---|")
            for k, s in sorted(summ.items(), key=lambda kv: -kv[1]["macro_f1"]["mean"]):
                f = lambda x: f"{100 * s[x]['mean']:.2f} ± {100 * s[x]['sd']:.2f}"
                ci = s["macro_f1"]["t_ci95"]
                md.append(f"| {k} | {s['n_runs']} | {f('accuracy')} | {f('macro_precision')} | {f('macro_recall')} | "
                          f"{f('macro_f1')} | [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}] |")
            for c in comps:
                md.append(f"\n- {c['a']} vs {c['b']}: Δ={100 * c['mean_diff']:.2f} pp, NB-corrected t p={c['p']}, "
                          f"Holm p={c['p_holm']}; McNemar p per seed = {[round(m['p'], 4) for m in c['mcnemar_per_pair']]}")
    out = os.path.join(rr, "_report")
    dump_json(report, os.path.join(out, "summary.json"))
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    log(f"report -> {out}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--runs")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    rr = a.runs or P(cfg, cfg["out_root"])
    build_report(cfg, rr)


if __name__ == "__main__":
    main()
