"""Training for every GPU job kind: base, sent, bilstm, distill, mtl, cnn.

Protocol shared by all kinds
  * data come only from the frozen manifests (rv2.manifest), which assert no leakage;
  * all RNGs are re-seeded at the start of the run (set_all_seeds);
  * after every epoch the model is scored on VALIDATION macro-F1; the best state is kept
    (strict improvement, init -inf) and early stopping uses the same signal;
  * chronological test firewall: a TRAINING job (phase='train') writes train/val
    predictions, validation metrics and the best-state checkpoint only -- it never encodes the test
    partition.  The test partition is predicted exactly once, by the job's `test__*` job
    (phase='test'), which runs only after every selection the run is a candidate in is frozen.  It
    reloads the checkpoint, re-predicts validation and checks it against the stored validation
    predictions, then predicts test;
  * model inputs use the frozen per-task preprocessing (cfg.preprocessing; MTL applies the HS and
    Telecom settings to their own task); model weights are loaded at pinned revisions.
"""
from __future__ import annotations

import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .common import P, dir_hash, dump_json, load_json, set_all_seeds
from .manifest import load_hs, load_task, load_telecom
from .metrics import compute_metrics, macro_f1
from .textnorm import preprocess

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ utilities
def model_path(cfg, name):
    p = cfg["models"][name]
    local = P(cfg, p)
    return local if os.path.isdir(local) else p


def prep_mode(cfg, task):
    """Frozen per-task input preprocessing: HS -> preprocessing.hs, Telecom/SudSenti3 -> their own."""
    return cfg["preprocessing"]["hs" if task in ("binary", "3class") else task]


def _pinned(cfg, name):
    """(path, kwargs) for from_pretrained at the pinned revision; a local model's directory hash is verified."""
    mp = model_path(cfg, name)
    rev = cfg["model_revisions"][name]
    if os.path.isdir(mp):
        got = dir_hash(mp)
        if got != rev:
            raise RuntimeError(f"local model {name} at {mp} has {got}, pinned {rev}")
        return mp, {}
    return mp, {"revision": rev}


def load_pretrained(cls, cfg, name, **kw):
    """Load at the pinned revision; the revision actually loaded (HF _commit_hash, or the verified local
    directory hash) must equal the pin -- a missing value is an error -- and is stored on the model."""
    mp, rk = _pinned(cfg, name)
    m = cls.from_pretrained(mp, **rk, **kw)
    got = _revision(m) if rk else dir_hash(mp)
    if got != cfg["model_revisions"][name]:
        raise RuntimeError(f"{name}: loaded revision {got} != pinned {cfg['model_revisions'][name]}")
    m._rv2_loaded_revision = got
    return m, mp


def load_tokenizer(cfg, name):
    from transformers import AutoTokenizer
    mp, rk = _pinned(cfg, name)
    return AutoTokenizer.from_pretrained(mp, **rk)


def _subset(split: dict, n):
    if not n:
        return split
    return {k: v[:n] for k, v in split.items()}


def get_data(cfg, task, fold=None, smoke=None):
    d = load_task(cfg, task, fold, frozen_dir=cfg.get("_frozen_override"))
    if smoke:
        for s in ("train", "val", "test"):
            d[s] = _subset(d[s], smoke[s])
    return d


def encode(tok, texts, max_len):
    enc = tok(texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def make_loader(ids, mask, y, bs, shuffle, seed, extra=None):
    tensors = [ids, mask, torch.tensor(y, dtype=torch.long)]
    if extra is not None:
        tensors.append(extra)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=bs, shuffle=shuffle, generator=g if shuffle else None)


@torch.no_grad()
def predict_logits(forward, loader, fp16=False):
    outs = []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and DEV.type == "cuda"):
            lo = forward(b[0].to(DEV), b[1].to(DEV))
        outs.append(lo.float().cpu())
    return torch.cat(outs).numpy()


def softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def hf_param_groups(model, wd):
    """HF Trainer default grouping: no weight decay on biases and LayerNorm weights."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (n.endswith(".bias") or "LayerNorm" in n or "layer_norm" in n or "layernorm" in n.lower()) else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def linear_schedule(opt, warmup, total):
    from transformers import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total)


def fit(model, train_loader, step_loss, val_score, epochs, patience, opt, sched, fp16, clip, log):
    """Generic epoch loop; selection signal = val_score() (validation macro-F1)."""
    use_amp = fp16 and DEV.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, best_state, best_epoch, bad, hist = -math.inf, None, -1, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for b in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = step_loss(model, b)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            tot += float(loss.detach()); nb += 1
        model.eval()
        vs = val_score()
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "sec": time.time() - t0, **vs}
        hist.append(rec)
        log(f"  epoch {ep}: loss={rec['train_loss']:.4f} val_macro_f1={vs['val_macro_f1']:.4f}")
        if vs["val_macro_f1"] > best:
            best, best_epoch, bad = vs["val_macro_f1"], ep, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if patience and bad >= patience:
                log(f"  early stop at epoch {ep} (patience {patience})")
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"best_val_macro_f1": best, "best_epoch": best_epoch, "epochs_run": len(hist), "history": hist,
            "amp_fp16_used": bool(use_amp), "device": DEV.type}


def _save_preds(path, uids, y, logits):
    np.savez_compressed(path, uids=np.array(uids), y=np.array(y), logits=logits.astype(np.float32),
                        probs=softmax_np(logits).astype(np.float32))


def _predict_split(forward, data, s, enc_fn, bs, fp16):
    ids, mask = enc_fn(data[s]["text"])
    return predict_logits(forward, make_loader(ids, mask, data[s]["y"], bs, False, 0), fp16)


def _write_split(run_dir, s, data, lo, label_names):
    _save_preds(os.path.join(run_dir, f"preds_{s}.npz"), data[s]["uid"], data[s]["y"], lo)
    if s == "train":
        return None
    m = compute_metrics(data[s]["y"], lo.argmax(1), label_names)
    dump_json(m, os.path.join(run_dir, f"metrics_{s}.json"))
    return m


def _val_predictions(run_dir, forward, data, enc_fn, bs, fp16, label_names, extra_train=False):
    """TRAIN phase: validation (+ train for teachers) predictions after the best state is restored. No test."""
    out = {}
    for s in (["train"] if extra_train else []) + ["val"]:
        m = _write_split(run_dir, s, data, _predict_split(forward, data, s, enc_fn, bs, fp16), label_names)
        if m:
            out[s] = m
    return out


CKPT = "state.pt"


def save_state(model, run_dir, sub=""):
    d = os.path.join(run_dir, "ckpt", sub)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f"{CKPT}.tmp.{os.getpid()}")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, tmp)
    os.replace(tmp, os.path.join(d, CKPT))


def load_state(model, src_dir, sub=""):
    p = os.path.join(src_dir, "ckpt", sub, CKPT)
    if not os.path.exists(p):
        raise RuntimeError(f"checkpoint missing: {p}")
    model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    model.to(DEV).eval()


def _test_predictions(run_dir, src_dir, forward, data, enc_fn, bs, fp16, label_names, sub=""):
    """TEST phase: reload -> re-predict validation and compare with the stored validation predictions
    -> predict the test partition once.  Writes preds_test.npz / metrics_test.json into the test job dir."""
    od = os.path.join(run_dir, sub)
    os.makedirs(od, exist_ok=True)
    v_new = _predict_split(forward, data, "val", enc_fn, bs, fp16)
    z = np.load(os.path.join(src_dir, sub, "preds_val.npz"))
    assert list(z["uids"]) == data["val"]["uid"], "validation uids changed between training and test"
    agree = float((z["logits"].argmax(1) == v_new.argmax(1)).mean())
    chk = {"val_argmax_agreement": agree, "val_max_abs_logit_diff": float(np.abs(z["logits"] - v_new).max())}
    if agree < 0.99:
        raise RuntimeError(f"reloaded checkpoint does not reproduce validation predictions: {chk}")
    m = _write_split(od, "test", data, _predict_split(forward, data, "test", enc_fn, bs, fp16), label_names)
    return {"test": m}, chk


def _finish(phase, run_dir, src_dir, model, train_fn, fwd, data, enc, bs, fp16, names, extra_train=False):
    if phase == "train":
        info = train_fn()
        res = _val_predictions(run_dir, fwd, data, enc, bs, fp16, names, extra_train)
        save_state(model, run_dir)
        return info, res
    load_state(model, src_dir)
    res, chk = _test_predictions(run_dir, src_dir, fwd, data, enc, bs, fp16, names)
    return {"val_check": chk}, res


def _revision(model):
    if getattr(model, "_rv2_loaded_revision", None):
        return model._rv2_loaded_revision
    c = getattr(model, "config", None)
    return getattr(c, "_commit_hash", None) if c is not None else None


def _prep(cfg, data, task, mode=None):
    mode = mode or prep_mode(cfg, task)
    for s in ("train", "val", "test"):
        data[s]["text"] = [preprocess(t, mode) for t in data[s]["text"]]
    return mode


# ------------------------------------------------------------------ kinds
# Every kind takes phase='train' (fit -> val preds + checkpoint) or phase='test' (src_dir = training run).
def run_base_like(cfg, job, run_dir, log, smoke=None, hp_key="baseline", phase="train", src_dir=None):
    """'base' (hate speech) and 'sent' (Telecom / SudSenti3 fold) share this routine."""
    from transformers import AutoModelForSequenceClassification
    hp = dict(cfg[hp_key])
    if smoke:
        hp.update(smoke.get("hp", {}))
    data = get_data(cfg, job["task"], job.get("fold"), smoke and smoke["n"])
    names = data["label_names"]
    mode = _prep(cfg, data, job["task"])
    set_all_seeds(job["seed"])
    tok = load_tokenizer(cfg, job["model"])
    model, mp = load_pretrained(AutoModelForSequenceClassification, cfg, job["model"],
                                num_labels=len(names), id2label=dict(enumerate(names)),
                                label2id={n: i for i, n in enumerate(names)}, ignore_mismatched_sizes=True)
    model.to(DEV)
    enc = lambda texts: encode(tok, texts, hp["max_len"])
    fwd = lambda i, m: model(input_ids=i, attention_mask=m).logits

    def train():
        tr_ids, tr_mask = enc(data["train"]["text"])
        va_ids, va_mask = enc(data["val"]["text"])
        tl = make_loader(tr_ids, tr_mask, data["train"]["y"], hp["batch_size"], True, job["seed"])
        vl = make_loader(va_ids, va_mask, data["val"]["y"], hp["eval_batch_size"], False, 0)
        opt = torch.optim.AdamW(hf_param_groups(model, hp["weight_decay"]), lr=hp["lr"],
                                betas=tuple(hp.get("adam_betas", (0.9, 0.999))), eps=hp.get("adam_eps", 1e-8))
        total = len(tl) * hp["epochs"]
        sched = linear_schedule(opt, math.ceil(hp["warmup_ratio"] * total), total)   # HF Trainer uses ceil

        def step(mo, b):
            return F.cross_entropy(fwd(b[0].to(DEV), b[1].to(DEV)), b[2].to(DEV))

        def val_score():
            lo = predict_logits(fwd, vl, hp["fp16"])
            return {"val_macro_f1": macro_f1(data["val"]["y"], lo.argmax(1), len(names))}
        return fit(model, tl, step, val_score, hp["epochs"], hp["patience"], opt, sched, hp["fp16"],
                   hp.get("max_grad_norm", 1.0), log)
    info, res = _finish(phase, run_dir, src_dir, model, train, fwd, data, enc, hp["eval_batch_size"], hp["fp16"],
                        names, extra_train=(job["kind"] == "base"))
    info.update(model_path=mp, model_revision=_revision(model), pinned_revision=cfg["model_revisions"][job["model"]],
                preprocessing=mode, label_names=names, hp=hp)
    if phase == "test" and job.get("train_kind") in cfg.get("keep_checkpoints", []) and not smoke:
        # kept HF-format checkpoint goes into the TEST job's own (staged, fenced) directory; never into src_dir
        model.save_pretrained(os.path.join(run_dir, "ckpt")); tok.save_pretrained(os.path.join(run_dir, "ckpt"))
    return info, res


def run_bilstm(cfg, job, run_dir, log, smoke=None, phase="train", src_dir=None):
    from transformers import AutoModel
    from .models import BiLSTMAttentionClassifier
    hp = dict(cfg["bilstm"]); hp.update((smoke or {}).get("hp", {}))
    data = get_data(cfg, job["task"], None, smoke and smoke["n"])
    names = data["label_names"]
    mode = _prep(cfg, data, job["task"])
    set_all_seeds(job["seed"])
    tok = load_tokenizer(cfg, hp["encoder"])
    enc_model, mp = load_pretrained(AutoModel, cfg, hp["encoder"])
    model = BiLSTMAttentionClassifier(enc_model, enc_model.config.hidden_size, hp["lstm_hidden"], len(names),
                                      hp["dropout"], hp["lstm_layers"]).to(DEV)
    enc = lambda texts: encode(tok, texts, hp["max_len"])
    fwd = lambda i, m: model(i, m)

    def train():
        tl = make_loader(*enc(data["train"]["text"]), data["train"]["y"], hp["batch_size"], True, job["seed"])
        vl = make_loader(*enc(data["val"]["text"]), data["val"]["y"], hp["eval_batch_size"], False, 0)
        head = [p for n, p in model.named_parameters() if not n.startswith("transformer.")]
        opt = torch.optim.AdamW([{"params": model.transformer.parameters(), "lr": hp["lr"] * hp["encoder_lr_mult"]},
                                 {"params": head, "lr": hp["lr"]}], weight_decay=hp["weight_decay"])
        total = len(tl) * hp["epochs"]
        sched = linear_schedule(opt, int(total * hp["warmup_ratio"]), total)
        step = lambda mo, b: F.cross_entropy(fwd(b[0].to(DEV), b[1].to(DEV)), b[2].to(DEV))
        val_score = lambda: {"val_macro_f1": macro_f1(data["val"]["y"], predict_logits(fwd, vl).argmax(1), len(names))}
        return fit(model, tl, step, val_score, hp["epochs"], hp["patience"], opt, sched, hp["fp16"], hp["max_grad_norm"], log)
    info, res = _finish(phase, run_dir, src_dir, model, train, fwd, data, enc, hp["eval_batch_size"], hp["fp16"], names)
    info.update(model_path=mp, model_revision=_revision(enc_model), pinned_revision=cfg["model_revisions"][hp["encoder"]],
                preprocessing=mode, label_names=names, hp=hp)
    return info, res


def distill_teacher(cfg, job, runs_root):
    """Frozen teacher of a distill / distill4 run (selection written by post__teacher__<task>)."""
    from .select import read_selection
    name = "teacher" if job.get("kind", job.get("train_kind")) == "distill" else "teacher4"
    return read_selection(runs_root, f"{name}_{job['task']}")["chosen"]


def run_distill(cfg, job, run_dir, log, smoke=None, phase="train", src_dir=None):
    from transformers import AutoModelForSequenceClassification
    hp = dict(cfg["distill"]); hp.update((smoke or {}).get("hp", {}))
    runs_root = os.path.dirname(run_dir)
    teacher = distill_teacher(cfg, job, runs_root)
    tdir = os.path.join(runs_root, f"base__{teacher}__{job['task']}__s{job['seed']}")
    data = get_data(cfg, job["task"], None, smoke and smoke["n"])
    names = data["label_names"]
    mode = _prep(cfg, data, job["task"])
    tlog = {}
    if phase == "train":
        for s in ("train", "val"):
            z = np.load(os.path.join(tdir, f"preds_{s}.npz"))
            assert list(z["uids"]) == data[s]["uid"], f"teacher {s} uids do not align with the manifest"
            tlog[s] = torch.tensor(z["logits"], dtype=torch.float32)
    set_all_seeds(job["seed"])
    tok = load_tokenizer(cfg, hp["student"])
    model, mp = load_pretrained(AutoModelForSequenceClassification, cfg, hp["student"],
                                num_labels=len(names), id2label=dict(enumerate(names)),
                                label2id={n: i for i, n in enumerate(names)}, ignore_mismatched_sizes=True)
    model.to(DEV)
    enc = lambda texts: encode(tok, texts, hp["max_len"])
    fwd = lambda i, m: model(input_ids=i, attention_mask=m).logits
    T, a = hp["temperature"], hp["alpha"]

    def train():
        tl = make_loader(*enc(data["train"]["text"]), data["train"]["y"], hp["batch_size"], True, job["seed"], tlog["train"])
        vl = make_loader(*enc(data["val"]["text"]), data["val"]["y"], hp["eval_batch_size"], False, 0)
        opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        total = len(tl) * hp["epochs"]
        sched = linear_schedule(opt, int(total * hp["warmup_ratio"]), total)

        def step(mo, b):   # alpha*T^2*KL(p_t || p_s) + (1-alpha)*CE
            s_lo = fwd(b[0].to(DEV), b[1].to(DEV)); t_lo = b[3].to(DEV)
            soft = F.kl_div(F.log_softmax(s_lo / T, -1), F.softmax(t_lo / T, -1), reduction="batchmean") * T * T
            return a * soft + (1 - a) * F.cross_entropy(s_lo, b[2].to(DEV))
        val_score = lambda: {"val_macro_f1": macro_f1(data["val"]["y"], predict_logits(fwd, vl).argmax(1), len(names))}
        return fit(model, tl, step, val_score, hp["epochs"], hp["patience"], opt, sched, hp["fp16"], hp["max_grad_norm"], log)
    info, res = _finish(phase, run_dir, src_dir, model, train, fwd, data, enc, hp["eval_batch_size"], hp["fp16"], names)
    info.update(teacher=teacher, teacher_run=os.path.basename(tdir), model_path=mp, preprocessing=mode,
                model_revision=_revision(model), pinned_revision=cfg["model_revisions"][hp["student"]],
                label_names=names, hp=hp)
    return info, res


def run_distill4(cfg, job, run_dir, log, smoke=None, phase="train", src_dir=None):
    """Reproduction-sensitivity distillation (historical 4-candidate teacher set).  When that set picks the
    same teacher as the primary selection, the run is by construction identical to `distill` and is
    recorded as an alias (no GPU work); its test job copies the primary test predictions."""
    runs_root = os.path.dirname(run_dir)
    from .select import read_selection
    t4 = read_selection(runs_root, f"teacher4_{job['task']}")["chosen"]
    t8 = read_selection(runs_root, f"teacher_{job['task']}")["chosen"]
    if t4 != t8:
        return run_distill(cfg, job, run_dir, log, smoke, phase, src_dir)
    prim = f"distill__{job['task']}__s{job['seed']}"
    log(f"  teacher4 == teacher ({t4}): alias of {prim}")
    if phase == "test":
        import shutil
        pt = os.path.join(runs_root, "test__" + prim)
        for fn in ("preds_test.npz", "metrics_test.json"):
            shutil.copy2(os.path.join(pt, fn), os.path.join(run_dir, fn))
        return {"alias_of": prim, "teacher": t4}, {"test": load_json(os.path.join(run_dir, "metrics_test.json"))}
    return {"alias_of": prim, "teacher": t4}, {}


def run_mtl(cfg, job, run_dir, log, smoke=None, phase="train", src_dir=None):
    from itertools import cycle
    from transformers import AutoModel
    from .models import MTLModel
    hp = dict(cfg["mtl"]); hp.update((smoke or {}).get("hp", {}))
    hs = get_data(cfg, job["task"], None, smoke and smoke["n"])
    se = load_telecom(cfg, cfg.get("_frozen_override"), for_mtl_aux=hp["exclude_aux_colliding_with_hs_eval"])
    if smoke:
        for s in ("train", "val", "test"):
            se[s] = _subset(se[s], smoke["n"][s])
    mode_hs = _prep(cfg, hs, job["task"])            # each MTL task uses its own frozen preprocessing
    mode_se = _prep(cfg, se, "telecom")
    # auxiliary sentiment training texts must not coincide with HS validation/test texts
    hs_eval = set(hs["val"]["uid"]) | set(hs["test"]["uid"])
    assert not hs_eval & set(se["train"]["uid"]), "MTL auxiliary train overlaps HS val/test"
    set_all_seeds(job["seed"])
    tok = load_tokenizer(cfg, job["model"])
    enc_model, mp = load_pretrained(AutoModel, cfg, job["model"])
    model = MTLModel(enc_model, enc_model.config.hidden_size, len(hs["label_names"]), len(se["label_names"]),
                     hp["dropout"]).to(DEV)
    enc = lambda texts: encode(tok, texts, hp["max_len"])
    alpha = float(job["alpha"])
    fh = lambda i, m: model(i, m, task="hs")
    fs = lambda i, m: model(i, m, task="sent")
    if phase == "test":
        load_state(model, src_dir)
        res, chk = _test_predictions(run_dir, src_dir, fh, hs, enc, hp["eval_batch_size"], hp["fp16"], hs["label_names"])
        res["sentiment"], chk_s = _test_predictions(run_dir, src_dir, fs, se, enc, hp["eval_batch_size"], hp["fp16"],
                                                    se["label_names"], sub="sentiment")
        info = {"val_check": chk, "val_check_sentiment": chk_s}
    else:
        hs_tl = make_loader(*enc(hs["train"]["text"]), hs["train"]["y"], hp["batch_size"], True, job["seed"])
        se_tl = make_loader(*enc(se["train"]["text"]), se["train"]["y"], hp["batch_size"], True, job["seed"] + 1)
        hs_vl = make_loader(*enc(hs["val"]["text"]), hs["val"]["y"], hp["eval_batch_size"], False, 0)
        se_vl = make_loader(*enc(se["val"]["text"]), se["val"]["y"], hp["eval_batch_size"], False, 0)
        steps = max(len(hs_tl), len(se_tl))
        opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        sched = linear_schedule(opt, int(hp["warmup_ratio"] * steps * hp["epochs"]), steps * hp["epochs"])

        class _Paired:   # one HS batch + one sentiment batch per step; the shorter loader is cycled
            def __iter__(self):
                if len(hs_tl) >= len(se_tl):
                    return zip(iter(hs_tl), cycle(se_tl))
                return zip(cycle(hs_tl), iter(se_tl))

            def __len__(self):
                return steps

        def step(mo, pair):
            (hi, hm, hy), (si, sm, sy) = pair
            lh = F.cross_entropy(fh(hi.to(DEV), hm.to(DEV)), hy.to(DEV))
            ls = F.cross_entropy(fs(si.to(DEV), sm.to(DEV)), sy.to(DEV))
            return alpha * lh + (1 - alpha) * ls

        def val_score():   # selection on HS validation macro-F1 only; sentiment val is logged
            return {"val_macro_f1": macro_f1(hs["val"]["y"], predict_logits(fh, hs_vl).argmax(1), len(hs["label_names"])),
                    "val_sent_macro_f1": macro_f1(se["val"]["y"], predict_logits(fs, se_vl).argmax(1), len(se["label_names"]))}
        info = fit(model, _Paired(), step, val_score, hp["epochs"], 0, opt, sched, hp["fp16"], hp["max_grad_norm"], log)
        res = _val_predictions(run_dir, fh, hs, enc, hp["eval_batch_size"], hp["fp16"], hs["label_names"])
        sdir = os.path.join(run_dir, "sentiment"); os.makedirs(sdir, exist_ok=True)
        res["sentiment"] = _val_predictions(sdir, fs, se, enc, hp["eval_batch_size"], hp["fp16"], se["label_names"])
        save_state(model, run_dir)
    info.update(alpha=alpha, model_path=mp, model_revision=_revision(enc_model),
                pinned_revision=cfg["model_revisions"][job["model"]], preprocessing={"hs": mode_hs, "telecom": mode_se},
                label_names=hs["label_names"], aux_train_n=len(se["train"]["uid"]), hp=hp)
    return info, res


def run_cnn(cfg, job, run_dir, log, smoke=None, phase="train", src_dir=None):
    from .models import CNN_MODELS, KerasLikeTokenizer
    hp = dict(cfg["cnn"]); hp.update((smoke or {}).get("hp", {}))
    task = {"hs_binary": "binary", "hs_3class": "3class"}.get(job["task"], job["task"])
    data = get_data(cfg, task, job.get("fold"), smoke and smoke["n"])
    names = data["label_names"]
    _prep(cfg, data, task, hp["preprocessing"])
    set_all_seeds(job["seed"])
    ktok = KerasLikeTokenizer(hp["num_words"]).fit(data["train"]["text"])      # fitted on TRAIN only (deterministic)
    enc = lambda texts: (ktok.encode(texts, hp["max_len"]), torch.ones(len(texts), 1))
    model = CNN_MODELS[job["model"]](ktok.vocab_size, len(names), hp["embedding"], **(
        {"max_len": hp["max_len"]} if job["model"] == "scm_mma" else {})).to(DEV)
    fwd = lambda i, m: model(i)

    def train():
        tl = make_loader(*enc(data["train"]["text"]), data["train"]["y"], hp["batch_size"], True, job["seed"])
        vl = make_loader(*enc(data["val"]["text"]), data["val"]["y"], hp["batch_size"], False, 0)
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], eps=1e-7)   # Keras Adam defaults

        def step(mo, b):
            return F.cross_entropy(fwd(b[0].to(DEV), None), b[2].to(DEV)) + model.reg_loss()

        def val_score():
            lo = predict_logits(fwd, vl)
            r = {"val_macro_f1": macro_f1(data["val"]["y"], lo.argmax(1), len(names)),
                 "val_loss": float(F.cross_entropy(torch.tensor(lo), torch.tensor(data["val"]["y"])))}
            if hp.get("early_stop_monitor") == "val_loss":      # optional: early stopping on validation loss
                r = dict(r, val_macro_f1_reported=r["val_macro_f1"], val_macro_f1=-r["val_loss"])
            return r
        return fit(model, tl, step, val_score, hp["epochs"], hp["patience"], opt, None, False, None, log)
    info, res = _finish(phase, run_dir, src_dir, model, train, fwd, data, enc, hp["batch_size"], False, names)
    info.update(vocab_size=ktok.vocab_size, preprocessing=hp["preprocessing"], label_names=names, hp=hp)
    return info, res


KINDS = {"base": run_base_like,
         "sent": lambda c, j, r, l, smoke=None, phase="train", src_dir=None:
             run_base_like(c, j, r, l, smoke, "sentiment", phase, src_dir),
         "bilstm": run_bilstm, "distill": run_distill, "distill4": run_distill4, "mtl": run_mtl, "cnn": run_cnn}


def run_test(cfg, job, run_dir, log, smoke=None):
    """test__* job: resolve the training run (MTL: the frozen selected alpha), then predict test once."""
    from .select import assert_frozen, read_selection
    runs_root = os.path.dirname(run_dir)
    assert_frozen(runs_root)                  # global firewall: nothing below may run before FROZEN.json
    tk = job["train_kind"]
    cleanup = []
    if tk == "mtl":
        alpha = read_selection(runs_root, f"mtl_alpha_{job['model']}_{job['task']}")["chosen"]
        src_id = f"mtl__{job['model']}__{job['task']}__a{alpha}__s{job['seed']}"
        keep_xai = (cfg.get("xai") and job["seed"] == cfg["seeds"][0] and job["model"] == cfg["xai"]["compare_encoder"])
        cleanup = [os.path.join(runs_root, f"mtl__{job['model']}__{job['task']}__a{a}__s{job['seed']}", "ckpt")
                   for a in cfg["mtl"]["alphas"] if not (keep_xai and a == alpha)]   # xai_mtlcmp needs this one
        tjob = dict(job, kind="mtl", alpha=alpha, id=src_id)
    else:
        src_id = job["train_id"]
        tjob = dict(job, kind=tk, id=src_id)
        if tk not in cfg.get("keep_checkpoints", []):
            cleanup = [os.path.join(runs_root, src_id, "ckpt")]
    src = os.path.join(runs_root, src_id)
    if not os.path.exists(os.path.join(src, "DONE")):
        raise RuntimeError(f"training run {src_id} is not DONE")
    log(f"  test materialisation of {src_id}")
    info, res = KINDS[tk](cfg, tjob, run_dir, log, smoke, phase="test", src_dir=src)
    if tk in cfg.get("keep_checkpoints", []):
        cleanup = [os.path.join(src, "ckpt", CKPT)] if not smoke else []   # HF-format copy was written instead
    info.update(train_run=src_id, _cleanup=cleanup)
    return info, res
