"""Post-freeze interpretability (SHAP + LIME) on the SELECTED frozen checkpoints.

Jobs (GPU, depend on post__freeze and on the test jobs of the explained runs):
  xai__<task>          SHAP + LIME of the validation-selected best single baseline (best_single_<task>),
                       run seed = cfg.seeds[0] (fixed rule).
  xai_mtlcmp__<task>   LIME of MTL(<compare_encoder>, validation-selected alpha) vs the single-task baseline
                       of the same encoder on the SAME instances.

Settings:
  SHAP  shap.maskers.Text(r"\\s+") + shap.Explainer(predict, masker, output_names=labels);
        n = 200 examples, global / per-class token aggregation, top-100 global and top-30 per class
        with >= 3 occurrences.
  LIME  LimeTextExplainer(class_names, split_expression=r"\\s+", random_state=seed), num_features=15,
        num_samples=500, all labels.
Sampling (outcome-free):
  stratified by TRUE label only -- per class floor(n / C) test items drawn with
  numpy.random.RandomState(xai.sample_seed) from the frozen test partition in manifest order.  The rule
  never sees a prediction; predictions are only recorded afterwards (correct / misclassified tags).
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np
import torch

from .common import dump_json, load_json


# ------------------------------------------------------------------ sampling (outcome-free)
def sample_indices(y, n_total, n_classes, seed):
    """Per class floor(n_total / C) indices, RandomState(seed), classes in label order; returns sorted indices."""
    rs = np.random.RandomState(seed)
    y = np.asarray(y)
    per = max(1, n_total // n_classes)
    out = []
    for c in range(n_classes):
        idx = np.flatnonzero(y == c)
        k = min(per, len(idx))
        out.extend(rs.choice(idx, size=k, replace=False).tolist())
    return sorted(out)


# ------------------------------------------------------------------ models
class Predictor:
    """texts -> class probabilities, batched, eval mode, same tokenisation / fp16 as the run."""

    def __init__(self, tok, forward, max_len, fp16, bs=16):
        from .train import DEV
        self.tok, self.fwd, self.max_len, self.fp16, self.bs, self.dev = tok, forward, max_len, fp16, bs, DEV

    @torch.no_grad()
    def __call__(self, texts):
        texts = [str(t) for t in texts]
        out = []
        for i in range(0, len(texts), self.bs):
            enc = self.tok(texts[i:i + self.bs], truncation=True, padding="max_length", max_length=self.max_len,
                           return_tensors="pt")
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16 and self.dev.type == "cuda"):
                lo = self.fwd(enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev))
            out.append(torch.softmax(lo.float(), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))


def _hp(cfg, key, smoke):
    hp = dict(cfg[key])
    hp.update((smoke or {}).get("hp", {}))
    return hp


def load_base(cfg, runs_root, model, task, seed, smoke):
    """Frozen baseline checkpoint: HF directory written by its test job, or state.pt (smoke)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .train import DEV, load_pretrained, load_state, load_tokenizer
    rid = f"base__{model}__{task}__s{seed}"
    src = os.path.join(runs_root, rid)
    names = load_json(os.path.join(src, "info.json"))["label_names"]
    hp = _hp(cfg, "baseline", smoke)
    ck = os.path.join(runs_root, "test__" + rid, "ckpt")      # HF copy written by the (fenced) test job
    if os.path.exists(os.path.join(ck, "config.json")):
        m = AutoModelForSequenceClassification.from_pretrained(ck).to(DEV).eval()
        tok = AutoTokenizer.from_pretrained(ck)
    else:
        m, _ = load_pretrained(AutoModelForSequenceClassification, cfg, model, num_labels=len(names),
                               id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)},
                               ignore_mismatched_sizes=True)
        load_state(m, src)
        tok = load_tokenizer(cfg, model)
    return Predictor(tok, lambda i, a: m(input_ids=i, attention_mask=a).logits, hp["max_len"], hp["fp16"]), rid, names


def load_mtl(cfg, runs_root, enc, task, alpha, seed, smoke):
    from transformers import AutoModel
    from .models import MTLModel
    from .train import DEV, load_pretrained, load_state, load_tokenizer
    rid = f"mtl__{enc}__{task}__a{alpha}__s{seed}"
    src = os.path.join(runs_root, rid)
    info = load_json(os.path.join(src, "info.json"))
    hp = _hp(cfg, "mtl", smoke)
    em, _ = load_pretrained(AutoModel, cfg, enc)
    m = MTLModel(em, em.config.hidden_size, len(info["label_names"]), len(cfg["data"]["sent_labels"]), hp["dropout"]).to(DEV)
    load_state(m, src)
    tok = load_tokenizer(cfg, enc)
    return Predictor(tok, lambda i, a: m(i, a, task="hs"), hp["max_len"], hp["fp16"]), rid, info["label_names"]


def _test_items(cfg, task, smoke, runs_root, rid):
    """Frozen test partition (model input text) + the run's own test predictions for consistency checks."""
    from .train import _prep, get_data
    data = get_data(cfg, task, None, smoke and smoke["n"])
    _prep(cfg, data, task)
    z = np.load(os.path.join(runs_root, ("test__" + rid) if not rid.startswith("mtl__") else
                             "test__" + "__".join(rid.split("__")[:3] + [rid.split("__")[4]]), "preds_test.npz"))
    assert list(z["uids"]) == data["test"]["uid"], "test uids differ from the frozen manifest"
    return data["test"], z["probs"]


def _check_reload(pred, texts, stored_probs, idx, tag):
    p = pred([texts[i] for i in idx])
    agree = float((p.argmax(1) == stored_probs[idx].argmax(1)).mean())
    if agree < 0.99:
        raise RuntimeError(f"{tag}: reloaded checkpoint disagrees with its test job ({agree:.3f})")
    return {"argmax_agreement_with_test_job": agree, "max_abs_prob_diff": float(np.abs(p - stored_probs[idx]).max())}


# ------------------------------------------------------------------ SHAP
def run_shap(pred, texts, y, names, idx, xc, out, log):
    import shap
    masker = shap.maskers.Text(xc["shap"]["masker_regex"])
    expl = shap.Explainer(lambda t: pred(list(t)), masker, output_names=list(names))
    sel = [texts[i] for i in idx]
    sv = expl(sel, max_evals=xc["shap"]["max_evals"], batch_size=xc["shap"]["batch_size"])
    C = len(names)
    tot, cnt, cls = defaultdict(float), defaultdict(int), defaultdict(lambda: np.zeros(C))
    per_ex = []
    for k in range(len(sel)):
        toks = [str(t).strip() for t in sv[k].data]
        vals = np.asarray(sv[k].values).reshape(len(toks), C)
        per_ex.append({"test_index": int(idx[k]), "true": names[y[idx[k]]], "tokens": toks,
                       "values": np.round(vals, 6).tolist()})
        for t, v in zip(toks, vals):
            if not t:
                continue
            tot[t] += float(np.abs(v).sum()); cnt[t] += 1; cls[t] += v
    glob = sorted(((t, tot[t] / cnt[t], cnt[t]) for t in tot), key=lambda r: -r[1])[: xc["topk_global"]]
    with open(os.path.join(out, "shap_global_topk.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["token", "mean_abs_shap_sum_over_classes", "count"]); w.writerows(glob)
    rows = []
    for c, n in enumerate(names):
        cand = [(t, cls[t][c] / cnt[t], cnt[t]) for t in cls if cnt[t] >= xc["min_count"]]
        for t, v, n_ in sorted(cand, key=lambda r: -r[1])[: xc["topk_per_class"]]:
            rows.append([n, t, v, n_])
    with open(os.path.join(out, "shap_per_class_topk.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["class", "token", "mean_shap", "count"]); w.writerows(rows)
    dump_json(per_ex, os.path.join(out, "shap_values.json"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = glob[:30][::-1]
        plt.figure(figsize=(7, 8))
        plt.barh([t for t, _, _ in top], [v for _, v, _ in top])
        plt.xlabel("mean |SHAP| (summed over classes)")
        plt.tight_layout(); plt.savefig(os.path.join(out, "shap_summary_bar.png"), dpi=150); plt.close()
    except Exception as e:                                    # plotting is cosmetic
        log(f"  shap plot skipped: {e}")
    log(f"  SHAP: {len(sel)} examples, {len(tot)} distinct tokens")
    return {"n_examples": len(sel), "n_tokens": len(tot)}


# ------------------------------------------------------------------ LIME
def lime_explain(pred, texts, names, idx, xc, seed):
    from lime.lime_text import LimeTextExplainer
    lc = xc["lime"]
    ex = LimeTextExplainer(class_names=list(names), split_expression=lc["split_expression"], random_state=seed)
    out = []
    for i in idx:
        e = ex.explain_instance(texts[i], lambda t: pred(list(t)), num_features=lc["num_features"],
                                num_samples=lc["num_samples"], labels=list(range(len(names))))
        out.append({"test_index": int(i), "text": texts[i],
                    "weights": {names[c]: [[w, float(v)] for w, v in e.as_list(label=c)] for c in range(len(names))},
                    "probs": np.asarray(e.predict_proba).round(6).tolist()})
    return out


def _tag(r, y, names):
    p = int(np.argmax(r["probs"]))
    r.update(true=names[y[r["test_index"]]], pred=names[p], correct=bool(p == y[r["test_index"]]))
    return r


# ------------------------------------------------------------------ jobs
def run_xai(cfg, job, run_dir, log, smoke=None):
    from .select import assert_frozen, read_selection
    rr = os.path.dirname(run_dir)
    assert_frozen(rr)                                     # test sentences only after the global freeze
    xc = cfg["xai"]
    task, seed = job["task"], cfg["seeds"][0]
    best = read_selection(rr, f"best_single_{task}")["chosen"]
    pred, rid, names = load_base(cfg, rr, best, task, seed, smoke)
    test, stored = _test_items(cfg, task, smoke, rr, rid)
    texts, y = test["text"], test["y"]
    n_shap, n_lime = xc["shap"]["n_samples"], xc["lime"]["n_examples"]
    i_shap = sample_indices(y, n_shap, len(names), xc["sample_seed"])
    i_lime = sample_indices(y, n_lime, len(names), xc["sample_seed"])
    chk = _check_reload(pred, texts, stored, i_shap, rid)
    shp = run_shap(pred, texts, y, names, i_shap, xc, run_dir, log)
    lime = [_tag(r, y, names) for r in lime_explain(pred, texts, names, i_lime, xc, xc["sample_seed"])]
    dump_json(lime, os.path.join(run_dir, "lime_examples.json"))
    dump_json({"model_run": rid, "model": best, "selection": f"best_single_{task}", "seed_rule": "cfg.seeds[0]",
               "sample_rule": "stratified by true label, RandomState(xai.sample_seed); no prediction used",
               "shap_uids": [test["uid"][i] for i in i_shap], "lime_uids": [test["uid"][i] for i in i_lime]},
              os.path.join(run_dir, "examples_used.json"))
    log(f"  xai({task}) on {rid}: shap={shp} lime={len(lime)} reload={chk}")
    return {"kind": "xai", "model_run": rid, "reload_check": chk, "shap": shp, "lime_n": len(lime),
            "label_names": names}


def _top(ws, k):
    return [w for w, _ in sorted(ws, key=lambda r: -abs(r[1]))[:k]]


def run_xai_mtlcmp(cfg, job, run_dir, log, smoke=None):
    """LIME of MTL(enc, selected alpha) vs single-task baseline(enc), same seed, same instances."""
    from scipy.stats import spearmanr
    from .select import assert_frozen, read_selection
    rr = os.path.dirname(run_dir)
    assert_frozen(rr)
    xc = cfg["xai"]
    task, seed, enc = job["task"], cfg["seeds"][0], xc["compare_encoder"]
    alpha = read_selection(rr, f"mtl_alpha_{enc}_{task}")["chosen"]
    pb, rid_b, names = load_base(cfg, rr, enc, task, seed, smoke)
    pm, rid_m, names_m = load_mtl(cfg, rr, enc, task, alpha, seed, smoke)
    assert names == names_m
    test, sb = _test_items(cfg, task, smoke, rr, rid_b)
    _, sm = _test_items(cfg, task, smoke, rr, rid_m)
    texts, y = test["text"], test["y"]
    idx = sample_indices(y, xc["lime"]["n_examples"], len(names), xc["sample_seed"])
    chk = {"base": _check_reload(pb, texts, sb, idx, rid_b), "mtl": _check_reload(pm, texts, sm, idx, rid_m)}
    lb = [_tag(r, y, names) for r in lime_explain(pb, texts, names, idx, xc, xc["sample_seed"])]
    lm = [_tag(r, y, names) for r in lime_explain(pm, texts, names, idx, xc, xc["sample_seed"])]
    k = min(5, xc["lime"]["num_features"])
    rows, ov, rho = [], [], []
    for a, b in zip(lb, lm):
        lab = a["true"]
        wa, wb = dict(map(tuple, a["weights"][lab])), dict(map(tuple, b["weights"][lab]))
        ta, tb = set(_top(a["weights"][lab], k)), set(_top(b["weights"][lab], k))
        j = len(ta & tb) / max(1, len(ta | tb))
        shared = sorted(set(wa) & set(wb))
        r = float(spearmanr([wa[t] for t in shared], [wb[t] for t in shared])[0]) if len(shared) >= 3 else None
        ov.append(j)
        if r is not None and np.isfinite(r):
            rho.append(r)
        rows.append({"test_index": a["test_index"], "text": a["text"], "true": lab,
                     "pred_base": a["pred"], "pred_mtl": b["pred"], "correct_base": a["correct"], "correct_mtl": b["correct"],
                     f"jaccard_top{k}": j, "spearman_shared": r,
                     "tokens": [{"token": t, "w_base": wa.get(t), "w_mtl": wb.get(t)} for t in sorted(set(wa) | set(wb))]})
    dump_json({"base": lb, "mtl": lm}, os.path.join(run_dir, "lime_both.json"))
    dump_json(rows, os.path.join(run_dir, "lime_mtl_vs_base.json"))
    with open(os.path.join(run_dir, "lime_mtl_vs_base_tokens.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["test_index", "true", "token", "w_base", "w_mtl", "delta"])
        for r in rows:
            for t in r["tokens"]:
                d = None if t["w_base"] is None or t["w_mtl"] is None else t["w_mtl"] - t["w_base"]
                w.writerow([r["test_index"], r["true"], t["token"], t["w_base"], t["w_mtl"], d])
    summ = {"base_run": rid_b, "mtl_run": rid_m, "alpha": alpha, "n_instances": len(rows),
            f"mean_jaccard_top{k}": float(np.mean(ov)) if ov else None,
            "mean_spearman_shared": float(np.mean(rho)) if rho else None, "reload_check": chk,
            "sample_rule": "stratified by true label, RandomState(xai.sample_seed); no prediction used",
            "uids": [test["uid"][i] for i in idx]}
    dump_json(summ, os.path.join(run_dir, "summary.json"))
    log(f"  xai_mtlcmp({task}): {summ[f'mean_jaccard_top{k}']} {summ['mean_spearman_shared']}")
    return {"kind": "xai_mtlcmp", **{k_: v for k_, v in summ.items() if k_ != "uids"}}
