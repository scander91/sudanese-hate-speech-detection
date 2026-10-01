"""End-to-end smoke test (CPU, tiny subsets, minutes).

    python tests/smoke_test.py

Proves: (1) Alg.3 reproduces the paper examples; (2) the manifest assertions catch injected
leakage and tampering; (3) the queue runs every job kind (base, sent, bilstm, distill, mtl, cnn,
post-hoc ensemble / MTL-alpha selection / report) and writes all outputs; (4) selections are
validation-only: they are reproduced with every test file hidden, and a run trained on a copy of
the data whose TEST labels are permuted has an identical training/validation trajectory;
(5) resume skips finished runs.
Also covered: per-task preprocessing, the chronological
test firewall (no training run holds test predictions; every selection is frozen before any candidate
test prediction exists; only the selected MTL alpha is tested), the post__teacher job and concurrent
locked selection writes, the campaign manifest and job-scoped fingerprints, the DONE re-check inside
the lock, lock-lease recovery, bounded retry with failure summaries, and --kinds forwarding/routing.
Writes smoke_runs/SMOKE_RESULT.json.
"""
import copy
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from rv2.launcher import cpu_safe_env               # noqa: E402  (MKL AVX pin on non-AVX2 CPUs)
os.environ.update(cpu_safe_env())
from rv2 import manifest as M                       # noqa: E402
from rv2.common import load_config, load_json       # noqa: E402
from rv2.run import run_job                         # noqa: E402
from rv2 import run as RUN                          # noqa: E402

SM = os.path.join(HERE, "smoke_runs")
PY = sys.executable
R = {"started": time.strftime("%F %T")}


def step(name, ok, **kw):
    R[name] = dict(ok=bool(ok), **kw)
    print(("PASS " if ok else "FAIL ") + name, kw if kw else "")
    assert ok, name


def main():
    shutil.rmtree(SM, ignore_errors=True)
    os.makedirs(SM)
    # 1 ---------------------------------------------------------------- text normalisation
    r = subprocess.run([PY, os.path.join(HERE, "tests", "test_textnorm.py")], capture_output=True, text=True)
    step("alg3_paper_examples", r.returncode == 0, out=r.stdout.strip())

    # 2 ---------------------------------------------------------------- manifest assertions
    cfg0 = load_config()
    frozen = os.path.join(cfg0["project_root"], cfg0["frozen_dir"])
    for t in ("binary", "3class"):
        d = M.load_hs(cfg0, t)
        step(f"manifest_load_{t}", True, sizes={s: len(d[s]["uid"]) for s in ("train", "val", "test")},
             labels=d["label_names"])
    df = M.read_manifest(os.path.join(frozen, "hs_manifest.tsv"))
    bad = df.copy()
    i_tr = bad.index[bad.split == "train"][0]
    i_te = bad.index[bad.split == "test"][0]
    bad.loc[i_te, "text"] = bad.loc[i_tr, "text"]           # inject a verbatim train sentence into test
    try:
        M.assert_disjoint(bad, "split"); caught = False
    except AssertionError as e:
        caught = "LEAKAGE" in str(e)
    step("leakage_injection_caught", caught)
    tamper = os.path.join(SM, "frozen_tamper"); shutil.copytree(frozen, tamper)
    with open(os.path.join(tamper, "hs_manifest.tsv"), "a") as f:
        f.write("\n")
    try:
        M.verify_checksums(tamper, ["hs_manifest.tsv"]); caught = False
    except RuntimeError:
        caught = True
    step("checksum_tamper_caught", caught)
    shutil.rmtree(tamper)
    rep0 = load_json(os.path.join(frozen, "phase0_report.json"))
    f2 = [M.load_fold_task(cfg0, "sudsenti2", f) for f in range(10)]
    oof = sorted(u for d in f2 for u in d["test"]["uid"])
    step("sudsenti2_phase0_addendum_frozen", rep0["sudsenti2_dedup"]["rows_out"] == len(oof) == len(set(oof))
         and f2[0]["label_names"] == ["neg", "pos"] and len(rep0["assertions"]["sudsenti2"]) == 10
         and rep0["addenda"][0]["dataset"] == "sudsenti2"
         and rep0["sudsenti2_mhamed_groups"]["groups_size_gt1"] >= 1
         and all(v == 0 for f in rep0["assertions"]["sudsenti2"].values() for v in f.values()),
         dedup=rep0["sudsenti2_dedup"], fold0={k: len(f2[0][k]["uid"]) for k in ("train", "val", "test")},
         mhamed_groups=rep0["sudsenti2_mhamed_groups"],
         labels=rep0["sudsenti2_label_dist"])

    # SudSenti2 is fail-closed for EVERY representation, including the CNN (mhamed) input
    s2 = M.read_manifest(os.path.join(frozen, "sudsenti2_manifest.tsv")).head(40).copy()
    s2["part"] = ["train"] * 20 + ["val"] * 10 + ["test"] * 10
    j_tr = s2.index[0]
    s2.loc[s2.index[35], "text"] = s2.loc[j_tr, "text"] + " في"        # differs raw/minimal, equal after mhamed stop words
    s2.loc[s2.index[35], "uid"] = "x"; s2.loc[s2.index[35], "raw_sha256"] = "y"
    from rv2.textnorm import preprocess as _pp
    same_mh = _pp(s2.loc[s2.index[35], "text"], "mhamed") == _pp(s2.loc[j_tr, "text"], "mhamed")
    try:
        M.assert_disjoint(s2, "part", fatal_all=True); caught = False
    except AssertionError as e:
        caught = "mhamed" in str(e)
    rep_only = M.assert_disjoint(s2, "part", fatal_all=False)
    step("sudsenti2_fail_closed_on_cnn_representation_collision", same_mh and caught
         and rep_only["mhamed:train&test"] >= 1 and "sudsenti2" in cfg0["data"]["fail_closed_all_representations"])

    # 3 ---------------------------------------------------------------- smoke config + queue
    with open(os.path.join(HERE, "configs", "default.yaml")) as f:
        y = yaml.safe_load(f)
    y["out_root"] = os.path.join(SM, "runs")
    y["seeds"] = [42, 43]
    y["hs_tasks"] = ["binary"]
    y["models"] = {k: y["models"][k] for k in ("sudabert_v2", "arabertv2")}
    y["ensemble"].update(candidates=["sudabert_v2", "arabertv2"], size=2,
                         also_report_fixed_paper_members={"binary": ["sudabert_v2", "arabertv2"]})
    y["distill"]["teacher_candidates"] = ["arabertv2"]
    y["distill"]["repro_teacher_candidates"] = ["sudabert_v2"]     # different from primary -> real distill4 runs
    y["mtl"].update(encoders=["arabertv2"], alphas=[0.5, 0.7])
    y["sentiment"].update(models=["sudabert_v2", "arabertv2"], datasets=["telecom", "sudsenti2"])
    y["sentiment"]["telecom_ensemble"]["size"] = 2
    y["cnn"]["datasets"] = ["hs_binary", "sudsenti2"]
    y["data"]["fold_subset"] = [0]                        # smoke only: fold 0 of the 10 SudSenti2 folds
    y["xai"].update(shap={"n_samples": 4, "masker_regex": "\\s+", "max_evals": 16, "batch_size": 8},
                    lime={"n_examples": 2, "num_features": 5, "num_samples": 20, "split_expression": "\\s+"})
    y["smoke"] = {"n": {"train": 48, "val": 32, "test": 32}, "hp": {"epochs": 2, "patience": 1, "max_len": 32}}
    y["evaluation"]["bootstrap_B"] = 200
    y["launcher"].update(poll_seconds=2, heartbeat_seconds=2)
    scfg = os.path.join(SM, "smoke.yaml")
    with open(scfg, "w") as f:
        yaml.safe_dump(y, f, allow_unicode=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    cfg = load_config(scfg)
    from rv2.jobs import all_jobs
    from rv2.run import status
    jobs = all_jobs(cfg)
    by = {j["id"]: j for j in jobs}
    rr = cfg["out_root"]

    # 3a: workers refuse to start without a campaign manifest; init writes it
    r = subprocess.run([PY, "-m", "rv2.launcher", "worker", "--config", scfg, "--max-jobs", "1"], cwd=HERE, env=env,
                       capture_output=True, text=True)
    step("worker_refuses_without_campaign_manifest", r.returncode != 0 and "no campaign manifest" in r.stderr)
    r = subprocess.run([PY, "-m", "rv2.launcher", "init", "--config", scfg], cwd=HERE, env=env, capture_output=True, text=True)
    camp = load_json(os.path.join(rr, "CAMPAIGN.json"))
    step("campaign_manifest_written", r.returncode == 0 and camp["n_jobs"] == len(jobs),
         campaign=camp["campaign_sha256"][:16], n_jobs=camp["n_jobs"], code=camp["code"])

    # 3b: --kinds is honoured by a worker and forwarded by `start`; host profile routes MTL
    r = subprocess.run([PY, "-m", "rv2.launcher", "worker", "--config", scfg, "--kinds", "cnn", "--max-jobs", "1"],
                       cwd=HERE, env=env, capture_output=True, text=True)
    ran = [ln.split()[3] for ln in r.stdout.splitlines() if " RUN " in ln]
    step("worker_kinds_filter", len(ran) == 1 and ran[0].startswith("cnn__"), ran=ran)
    from rv2 import launcher as L
    cmds = L.start(cfg, "host1", [0], scfg, kinds=["mtl", "base"], max_jobs=None, dry=True)
    rc = cmds[0][-1]
    step("start_forwards_kinds_host_config", "--kinds mtl,base" in rc and "--host host1" in rc and "--config " in rc,
         remote=rc)
    est = {"mtl": 41.0, "bilstm": 20.0, "distill": 14.0}
    fake = [dict(by[i]) for i in by if by[i]["kind"] in ("base", "mtl", "bilstm", "cnn")]
    st0 = {j["id"]: "todo" for j in fake}
    o11, _ = L.runnable(cfg, fake, True, profile=y["hosts"]["host1"], est=est, st=st0)
    o20, _ = L.runnable(cfg, fake, True, profile=y["hosts"]["host2"], est=est, st=st0)
    ft = [dict(by[i]) for i in by if by[i]["kind"] == "test"]
    t20, _ = L.runnable(cfg, ft, True, profile=y["hosts"]["host2"], est=est, st={i: "done" for i in by} | {j["id"]: "todo" for j in ft})
    import socket as _s
    orig_host = RUN.socket.gethostname
    RUN.socket.gethostname = lambda: "host2.example.org"
    try:
        RUN._host_allows(cfg, by["mtl__arabertv2__binary__a0.7__s42"]); refused = False
    except RuntimeError as e:
        refused = "kind excluded" in str(e)
    try:
        RUN._host_allows(cfg, by["bilstm__binary__s42"]); bil_ok = True
    except RuntimeError:
        bil_ok = False
    RUN.socket.gethostname = orig_host
    step("routing_mtl_only_on_host1_first_never_on_host2",
         o11[0]["kind"] == "mtl" and not any(j["kind"] == "mtl" for j in o20) and o20[0]["kind"] == "bilstm"
         and not any(j.get("train_kind") == "mtl" for j in t20) and len(t20) > 0 and refused and bil_ok,
         host1_first=o11[0]["id"], host2_first=o20[0]["id"], host2_mtl_jobs=0, run_refuses_mtl_on_host2=refused)

    # 3c: full queue.  One GPU-kind worker (on CPU here) also runs the CPU post jobs; one injected failure
    #     (first attempt of a CNN test job) must be retried automatically.
    inj = "test__cnn__scm_mma__hs_binary__s43"
    t0 = time.time()
    for extra in ([], ["--cpu"]):
        e2 = dict(env, RV2_INJECT_FAIL=inj, RV2_INJECT_FAIL_N="1", RV2_INJECT_KILL="test__cnn__cnn_baseline__hs_binary__s43")
        r = subprocess.run([PY, "-m", "rv2.launcher", "worker", "--config", scfg] + extra, cwd=HERE, env=e2,
                           capture_output=True, text=True, timeout=7200)
        open(os.path.join(SM, f"worker{'_cpu' if extra else ''}.log"), "w").write(r.stdout + r.stderr)
    st = {j["id"]: status(cfg, j) for j in jobs}
    step("queue_all_jobs_done", all(v == "done" for v in st.values()), n_jobs=len(jobs),
         kinds=sorted({j["kind"] for j in jobs}), wall_min=round((time.time() - t0) / 60, 1),
         not_done={k: v for k, v in st.items() if v != "done"})
    a = load_json(os.path.join(rr, "_ctl", inj, "ATTEMPTS.json"))
    step("retry_after_injected_failure", a["n"] == 1 and st[inj] == "done", attempts=a["n"],
         error=a["errors"][0]["error"].strip().splitlines()[-1])
    kj = "test__cnn__cnn_baseline__hs_binary__s43"
    ak = load_json(os.path.join(rr, "_ctl", kj, "ATTEMPTS.json"))
    rl = open(os.path.join(rr, "_ctl", kj, "lock_recovery.log")).read()
    step("killed_child_lock_cleared_only_via_verified_own_child_path", ak["n"] == 1 and "-9" in ak["errors"][0]["error"]
         and "own child pid" in rl and "verified dead" in rl and st[kj] == "done", log=rl.strip()[-160:])

    # 4 ---------------------------------------------------------------- outputs per run + firewall
    missing, leaks = [], []
    for j in jobs:
        d = os.path.join(rr, j["id"])
        if j["kind"] in ("base", "sent", "bilstm", "distill", "distill4", "mtl", "cnn"):
            for fn in ("preds_val.npz", "metrics_val.json", "provenance.json", "history.json", "info.json", "DONE"):
                if not os.path.exists(os.path.join(d, fn)):
                    missing.append(f"{j['id']}/{fn}")
            leaks += [p for p in glob.glob(os.path.join(d, "**", "*test*"), recursive=True)]
            h = load_json(os.path.join(d, "history.json"))
            info = load_json(os.path.join(d, "info.json"))
            v = [e["val_macro_f1"] for e in h]
            if info["best_epoch"] != 1 + int(np.argmax(v)):
                missing.append(f"{j['id']}: best_epoch {info['best_epoch']} != argmax(val)")
        elif j["kind"] == "test":
            for fn in ("preds_test.npz", "metrics_test.json", "provenance.json", "info.json", "DONE"):
                if not os.path.exists(os.path.join(d, fn)):
                    missing.append(f"{j['id']}/{fn}")
    step("outputs_written_and_checkpoint_is_val_argmax", not missing, problems=missing[:10])
    step("firewall_no_test_output_in_any_training_run", not leaks, leaks=leaks[:5])
    # every selection file was frozen before the first test prediction of any of its candidates
    sels = {os.path.basename(p)[:-5]: p for p in glob.glob(os.path.join(rr, "_selection", "*.json"))
            if not p.endswith("FROZEN.json")}
    def t_test(tid):
        return os.path.getmtime(os.path.join(rr, tid, "preds_test.npz"))
    order_bad = []
    for name, p in sels.items():
        ts = os.path.getmtime(p)
        if name.startswith(("teacher", "ensemble_binary", "best_single_binary")):
            cands = [j for j in jobs if j["kind"] == "test" and j.get("train_kind") == "base"]
        elif name.endswith("telecom"):
            cands = [j for j in jobs if j["kind"] == "test" and j.get("train_kind") == "sent" and j["task"] == "telecom"]
        elif name.startswith("mtl_alpha"):
            cands = [j for j in jobs if j["kind"] == "test" and j.get("train_kind") == "mtl"]
        else:
            cands = []
        first = min(t_test(j["id"]) for j in cands) if cands else None
        if first is None or ts >= first:
            order_bad.append((name, ts, first))
    step("firewall_selections_frozen_before_candidate_tests", not order_bad and len(sels) >= 7,
         selections=sorted(sels), problems=order_bad[:3])
    fz = os.path.join(rr, "_selection", "FROZEN.json")
    first_test = min(os.path.getmtime(os.path.join(rr, j["id"], "preds_test.npz")) for j in jobs if j["kind"] == "test")
    fzr = load_json(fz)
    step("firewall_global_freeze_before_every_test", os.path.getmtime(fz) < first_test
         and all("post__freeze" in j["deps"] for j in jobs if j["kind"] == "test")
         and sorted(fzr["selections"]) == sorted(k + ".json" for k in sels)
         and all(max(os.path.getmtime(p) for p in sels.values()) <= os.path.getmtime(fz) for _ in [0]),
         n_selections=len(fzr["selections"]), n_test_jobs=sum(j["kind"] == "test" for j in jobs))
    from rv2 import select as SEL
    os.rename(fz, fz + ".away")
    try:
        SEL.assert_frozen(rr); c1 = False
    except RuntimeError as e:
        c1 = "missing" in str(e)
    finally:
        os.rename(fz + ".away", fz)
    tp = sels["teacher_binary"]
    t_orig = open(tp).read()
    open(tp, "w").write(t_orig + " ")
    try:
        SEL.assert_frozen(rr); c2 = False
    except RuntimeError as e:
        c2 = "changed after the global freeze" in str(e)
    finally:
        open(tp, "w").write(t_orig)
    step("test_jobs_refuse_without_or_after_changed_freeze", c1 and c2 and SEL.assert_frozen(rr) is not None)
    a_sel = load_json(sels["mtl_alpha_arabertv2_binary"])["chosen"]
    mt = [j for j in jobs if j["kind"] == "test" and j.get("train_kind") == "mtl"]
    tr = [load_json(os.path.join(rr, j["id"], "info.json"))["train_run"] for j in mt]
    ck = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(rr, "mtl__*", "ckpt")))
    step("firewall_only_selected_mtl_alpha_tested", len(mt) == 2 and all(f"__a{a_sel}__" in t for t in tr)
         and ck == [f"mtl__arabertv2__binary__a{a_sel}__s42"], alpha=a_sel, tested=tr, kept_ckpt_for_xai=ck)
    vc = [load_json(os.path.join(rr, j["id"], "info.json")).get("val_check") for j in jobs if j["kind"] == "test"]
    vc = [v for v in vc if v]
    step("test_jobs_reproduce_stored_validation_predictions", all(v["val_argmax_agreement"] == 1.0 for v in vc),
         n=len(vc), max_abs_logit_diff=max(v["val_max_abs_logit_diff"] for v in vc))

    # 4b ---------------------------------------------------------------- per-task preprocessing
    pp = {jid: load_json(os.path.join(rr, jid, "info.json"))["preprocessing"] for jid in
          ("base__arabertv2__binary__s42", "sent__arabertv2__telecom__s42", "mtl__arabertv2__binary__a0.7__s42",
           "cnn__cnn_baseline__hs_binary__s42")}
    step("per_task_preprocessing_frozen", pp["base__arabertv2__binary__s42"] == "alg3"
         and pp["sent__arabertv2__telecom__s42"] == "minimal"
         and pp["mtl__arabertv2__binary__a0.7__s42"] == {"hs": "alg3", "telecom": "minimal"}
         and pp["cnn__cnn_baseline__hs_binary__s42"] == "mhamed", modes=pp)

    # 4c --------------------------------------------------------------- SudSenti2 runs + post-freeze XAI
    s2 = load_json(os.path.join(rr, "sent__arabertv2__sudsenti2__f0__s42", "info.json"))
    step("sudsenti2_jobs_run_minimal_preprocessing_binary_labels", s2["preprocessing"] == "minimal"
         and s2["label_names"] == ["neg", "pos"] and st["test__sent__arabertv2__sudsenti2__f0__s42"] == "done"
         and st["test__cnn__scm_mma__sudsenti2__f0__s42"] == "done", n_sent_sudsenti2=sum(
             j["kind"] == "sent" and j["task"] == "sudsenti2" for j in jobs))
    from rv2.xai import sample_indices
    xd, xm = os.path.join(rr, "xai__binary"), os.path.join(rr, "xai_mtlcmp__binary")
    ex = load_json(os.path.join(xd, "examples_used.json"))
    xi = load_json(os.path.join(xd, "info.json"))
    best = load_json(sels["best_single_binary"])["chosen"]
    tst = M.load_hs(cfg0, "binary")["test"]
    tst = {k: v[:cfg["smoke"]["n"]["test"]] for k, v in tst.items()}
    re_uids = [tst["uid"][i] for i in sample_indices(tst["y"], cfg["xai"]["shap"]["n_samples"], 2, cfg["xai"]["sample_seed"])]
    files = [f for f in ("shap_global_topk.csv", "shap_per_class_topk.csv", "shap_values.json", "lime_examples.json")
             if os.path.exists(os.path.join(xd, f))]
    step("xai_on_selected_frozen_checkpoint_after_freeze", st["xai__binary"] == "done" and len(files) == 4
         and ex["model_run"] == f"base__{best}__binary__s42" and ex["shap_uids"] == re_uids
         and xi["reload_check"]["argmax_agreement_with_test_job"] == 1.0
         and os.path.getmtime(os.path.join(xd, "shap_values.json")) > os.path.getmtime(fz)
         and "post__freeze" in by["xai__binary"]["deps"], model_run=ex["model_run"], files=files,
         n_shap=xi["shap"], n_lime=xi["lime_n"])
    xs = load_json(os.path.join(xm, "summary.json"))
    step("xai_mtl_vs_baseline_lime_same_instances", st["xai_mtlcmp__binary"] == "done" and xs["alpha"] == a_sel
         and xs["mtl_run"] == f"mtl__arabertv2__binary__a{a_sel}__s42" and xs["base_run"] == "base__arabertv2__binary__s42"
         and xs["reload_check"]["mtl"]["argmax_agreement_with_test_job"] == 1.0
         and len(load_json(os.path.join(xm, "lime_mtl_vs_base.json"))) == xs["n_instances"] > 0,
         summary={k: xs[k] for k in xs if k.startswith("mean") or k in ("alpha", "n_instances")})

    # 5 ---------------------------------------------------------------- teacher job + selection locking
    from rv2 import select
    t8, t4 = load_json(sels["teacher_binary"]), load_json(sels["teacher4_binary"])
    dinfo = load_json(os.path.join(rr, "distill__binary__s42", "info.json"))
    d4info = load_json(os.path.join(rr, "distill4__binary__s42", "info.json"))
    step("teacher_selected_by_dedicated_cpu_job", by["post__teacher__binary"]["gpu"] is False
         and "post__teacher__binary" in by["distill__binary__s42"]["deps"]
         and dinfo["teacher"] == t8["chosen"] and d4info["teacher"] == t4["chosen"]
         and bool(t8.get("inputs_sha256")), teacher=t8["chosen"], teacher4=t4["chosen"],
         inputs_sha256=t8["inputs_sha256"][:16])
    # concurrent writers of one selection: all agree, no failure, no leftover tmp/lock
    code = ("import sys; sys.path.insert(0, %r); from rv2 import select; "
            "print(select.write_selection(%r, 'concurrency_probe', {'chosen': 'x', 'inputs_sha256': 'h'})['frozen_by'])"
            % (HERE, rr))
    ps = [subprocess.Popen([PY, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(6)]
    outs = [p.communicate() + (p.returncode,) for p in ps]
    left = [f for f in os.listdir(os.path.join(rr, "_selection")) if ".tmp" in f or f.startswith(".lock")]
    step("concurrent_selection_writes_are_atomic", all(o[2] == 0 for o in outs) and len({o[0] for o in outs}) == 1
         and not left, writers=len(outs), frozen_by=outs[0][0].strip(), leftovers=left)
    os.remove(os.path.join(rr, "_selection", "concurrency_probe.json"))
    try:
        select.write_selection(rr, "teacher_binary", dict(t8, inputs_sha256="0" * 64)); caught = False
    except RuntimeError as e:
        caught = "candidate inputs changed" in str(e)
    h0 = select.inputs_hash(rr, ["base__arabertv2__binary__s42"])
    mv = os.path.join(rr, "base__arabertv2__binary__s42", "metrics_val.json")
    orig = open(mv).read()
    open(mv, "w").write(orig + " ")                      # any byte change of a candidate validation file
    h1 = select.inputs_hash(rr, ["base__arabertv2__binary__s42"])
    open(mv, "w").write(orig)
    step("selection_rejects_changed_candidate_inputs", caught and h0 != h1)

    # 6 ---------------------------------------------------------------- selection with test hidden
    hidden = []
    for p in glob.glob(os.path.join(rr, "*", "preds_test.npz")) + glob.glob(os.path.join(rr, "*", "metrics_test.json")):
        os.rename(p, p + ".hidden"); hidden.append(p)
    try:
        before = {k: load_json(v)["chosen"] for k, v in sels.items()}
        shutil.move(os.path.join(rr, "_selection"), os.path.join(SM, "_selection_first"))
        select.ensemble_select(cfg, rr, "binary", log=lambda *a: None)
        select.ensemble_select(cfg, rr, "telecom", log=lambda *a: None)
        select.run_post_mtlsel(cfg, rr, "binary", log=lambda *a: None)
        select.run_post_teacher(cfg, rr, "binary", log=lambda *a: None)
        after = {os.path.basename(p)[:-5]: load_json(p)["chosen"] for p in glob.glob(os.path.join(rr, "_selection", "*.json"))
                 if not p.endswith("FROZEN.json")}
        step("selection_reproduced_with_all_test_files_hidden", before == after, after=after)
    finally:
        for p in hidden:
            os.rename(p + ".hidden", p)
        shutil.rmtree(os.path.join(rr, "_selection"))
        shutil.move(os.path.join(SM, "_selection_first"), os.path.join(rr, "_selection"))

    # 7 ---------------------------------------------------------------- campaign manifest + job-scoped fingerprints
    def cfg_with(**over):
        c = load_config(scfg, over)
        return c
    c_lr = cfg_with(**{"baseline.lr": 3e-5})
    try:
        RUN.check_campaign(c_lr); caught = False
    except RuntimeError as e:
        caught = "campaign manifest mismatch" in str(e)
    try:
        status(c_lr, by["cnn__cnn_baseline__hs_binary__s42"]); c_st = False
    except RuntimeError as e:
        c_st = "campaign manifest mismatch" in str(e)
    try:
        RUN.deps_done(c_lr, by["post__freeze"]); c_dd = False
    except RuntimeError as e:
        c_dd = "campaign manifest mismatch" in str(e)
    step("status_and_deps_done_validate_campaign_manifest", c_st and c_dd)
    s_lr = {j["id"]: status(c_lr, j, check=False) for j in jobs}
    stale = sorted(k for k, v in s_lr.items() if v == "stale")
    step("campaign_mismatch_refused_and_status_validates_fingerprint", caught
         and s_lr["base__arabertv2__binary__s42"] == "stale" and s_lr["cnn__cnn_baseline__hs_binary__s42"] == "done"
         and s_lr["sent__arabertv2__telecom__s42"] == "done" and s_lr["distill__binary__s42"] == "stale",
         n_stale=len(stale), example_stale=stale[:4])
    c_b = cfg_with(**{"evaluation.bootstrap_B": 999})
    s_b = {j["id"]: status(c_b, j, check=False) for j in jobs}
    step("report_only_edit_invalidates_only_report", sorted(k for k, v in s_b.items() if v == "stale") == ["post__report"])
    r = subprocess.run([PY, "-m", "rv2.run", "--config", scfg, "--job", "cnn__cnn_baseline__hs_binary__s42"],
                       cwd=HERE, env=env, capture_output=True, text=True)
    from rv2 import common as CM
    CM._CODE_CACHE["train"] = "edited"
    c_code = load_config(scfg)
    s_c = {j["id"]: status(c_code, j, check=False) for j in jobs}
    CM._CODE_CACHE.clear()
    step("training_code_edit_invalidates_training_not_selection_code",
         s_c["cnn__cnn_baseline__hs_binary__s42"] == "stale" and s_c["post__report"] == "stale"
         and r.stdout.strip().endswith("skipped (done)"), resume_stdout=r.stdout.strip()[-40:])

    # 8 ---------------------------------------------------------------- DONE re-check inside the lock + lock lease
    jid = "cnn__cnn_baseline__hs_binary__s42"
    rd = os.path.join(rr, jid)
    done_txt = open(os.path.join(rd, "DONE")).read()
    os.rename(os.path.join(rd, "DONE"), os.path.join(rd, "DONE.bak"))
    orig_acq = RUN.acquire_lock

    def racing_acquire(d):             # another worker finishes the job between the DONE check and the lock
        os.rename(os.path.join(rd, "DONE.bak"), os.path.join(rd, "DONE"))
        return orig_acq(d)
    RUN.acquire_lock = racing_acquire
    try:
        res = run_job(cfg, by[jid])
    finally:
        RUN.acquire_lock = orig_acq
    cdir = RUN.ctl_dir(cfg, jid)
    step("done_rechecked_inside_lock", res == "skipped (done)" and open(os.path.join(rd, "DONE")).read() == done_txt
         and not os.path.isdir(os.path.join(cdir, ".lock")), result=res)
    # 8a --------------------------------------------------------------- no lease breaking: stale locks are
    #      reported, never taken over; `unlock` refuses while the owner lives and works after it died
    import socket
    lk = os.path.join(cdir, ".lock")
    sleeper = subprocess.Popen(["sleep", "300"])
    os.mkdir(lk)
    json.dump({"host": socket.gethostname(), "pid": sleeper.pid, "time": 0, "heartbeat": time.time() - 10 ** 5,
               "token": "livetoken"}, open(os.path.join(lk, "owner.json"), "w"))
    os.rename(os.path.join(rd, "DONE"), os.path.join(rd, "DONE.bak"))  # make the job look runnable
    r_other = run_job(cfg, by[jid])                         # another worker: must NOT take the stale lock
    not_taken = r_other == "locked" and RUN.still_owner(cdir, "livetoken")
    stale = [x for x in L.stale_locks(cfg) if x["job"] == jid]
    rs = subprocess.run([PY, "-m", "rv2.launcher", "status", "--config", scfg], cwd=HERE, env=env,
                        capture_output=True, text=True)
    listed = f"{jid}: owner {socket.gethostname()}:{sleeper.pid}" in rs.stdout
    ru = subprocess.run([PY, "-m", "rv2.launcher", "unlock", "--job", jid, "--config", scfg], cwd=HERE, env=env,
                        capture_output=True, text=True)
    refused_live = ru.returncode != 0 and "REFUSED" in ru.stdout and RUN.still_owner(cdir, "livetoken")
    released_foreign = RUN.release_lock(cdir, "not-the-owner")      # never deletes a foreign lock
    sleeper.kill(); sleeper.wait()
    ru2 = subprocess.run([PY, "-m", "rv2.launcher", "unlock", "--job", jid, "--config", scfg], cwd=HERE, env=env,
                         capture_output=True, text=True)
    unlocked_dead = ru2.returncode == 0 and "UNLOCKED" in ru2.stdout and not os.path.isdir(lk)
    os.mkdir(lk)                                             # remote owner that cannot be verified -> refuse
    json.dump({"host": "nosuchhost-rv2", "pid": 12345, "time": 0, "heartbeat": 0, "token": "t"},
              open(os.path.join(lk, "owner.json"), "w"))
    ok_r, msg_r = RUN.unlock_verified(cdir, "smoke")
    refused_unverifiable = (not ok_r) and os.path.isdir(lk)
    shutil.rmtree(lk)
    os.rename(os.path.join(rd, "DONE.bak"), os.path.join(rd, "DONE"))
    step("stale_lock_not_taken_over_unlock_verifies_owner_pid", not_taken and len(stale) == 1 and listed
         and refused_live and released_foreign is False and unlocked_dead and refused_unverifiable
         and status(cfg, by[jid]) == "done", other_worker=r_other, status_lists=listed,
         unlock_live=ru.stdout.strip()[-90:], unlock_dead=ru2.stdout.strip()[-90:], unverifiable=msg_r)

    # 8b --------------------------------------------------------------- own-token release, no resurrection, 1 winner
    fd = os.path.join(SM, "lockfence")
    os.makedirs(fd, exist_ok=True)
    tok1 = RUN.acquire_lock(fd)
    hb1 = RUN._Heartbeat(fd, tok1, 0.05); hb1.start()
    time.sleep(0.2)
    beat_ok = RUN.lock_age(fd) < 1.0
    rel_wrong = RUN.release_lock(fd, "someone-else")
    hb1.halt()
    rel_own = RUN.release_lock(fd, tok1)
    time.sleep(0.3)
    gone = not os.path.exists(os.path.join(fd, ".lock"))    # joined heartbeat never resurrects it
    code = ("import sys; sys.path.insert(0, %r); from rv2 import run; t = run.acquire_lock(%r); print(t or '')" % (HERE, fd))
    ps = [subprocess.Popen([PY, "-c", code], stdout=subprocess.PIPE, text=True) for _ in range(6)]
    toks = [p.communicate()[0].strip() for p in ps]
    winners = [t for t in toks if t]
    one = len(winners) == 1 and RUN.still_owner(fd, winners[0])
    RUN.release_lock(fd, winners[0] if winners else None)
    step("lock_release_own_token_only_no_resurrection", beat_ok and rel_wrong is False and rel_own and gone and one,
         concurrent_winners=len(winners))

    # 8c --------------------------------------------------------------- operator error DURING computation: the lock
    #      vanishes mid-run (e.g. a wrong manual rm) and a second worker runs and publishes; the first keeps
    #      writing artifacts, then refuses to publish -> the published result is the second worker's only.
    jl = by["cnn__cnn_baseline__hs_binary__s43"]
    rdl, cdl = os.path.join(rr, jl["id"]), RUN.ctl_dir(cfg, jl["id"])
    shutil.rmtree(rdl)
    import rv2.train  # noqa: F401
    orig_rt = sys.modules["rv2.train"].KINDS["cnn"]
    seen = {"calls": 0}

    def overlapping(c, j, r, l, smoke=None, **kw):
        seen["calls"] += 1
        if seen["calls"] > 1:                            # worker B: a normal run
            seen["b_stage"] = r
            return orig_rt(c, j, r, l, smoke, **kw)
        open(os.path.join(r, "A_partial.marker"), "w").write("A before")
        shutil.rmtree(os.path.join(cdl, ".lock"))        # operator error, simulated
        seen["b_result"] = run_job(cfg, jl)              # B acquires, computes, publishes
        open(os.path.join(r, "A_late.marker"), "w").write("A after")
        return orig_rt(c, j, r, l, smoke, **kw)          # A writes a full set of artifacts after B published
    sys.modules["rv2.train"].KINDS["cnn"] = overlapping
    try:
        run_job(cfg, jl); a_err = ""
    except RuntimeError as e:
        a_err = str(e)
    finally:
        sys.modules["rv2.train"].KINDS["cnn"] = orig_rt
    pub = sorted(os.listdir(rdl))
    b_tok = load_json(os.path.join(rdl, "DONE"))["token"]
    stages = [d for d in os.listdir(rr) if d.startswith(".stage.")]
    step("lock_lost_during_computation_fenced_by_staged_publish",
         seen["b_result"] == "done" and "ownership lost" in a_err and not any(f.endswith(".marker") for f in pub)
         and b_tok == load_json(os.path.join(rdl, "provenance.json"))["lock_token"] and b_tok in seen["b_stage"]
         and status(cfg, jl) == "done" and not os.path.isdir(os.path.join(cdl, ".lock")) and not stages,
         a_error=a_err[:80], published=pub, leftover_stages=stages)

    # 9 ---------------------------------------------------------------- permanent failure: bounded retry + summary
    y3 = copy.deepcopy(y)
    y3["out_root"] = os.path.join(SM, "runs_fail")
    c3 = os.path.join(SM, "smoke_fail.yaml")
    yaml.safe_dump(y3, open(c3, "w"), allow_unicode=True)
    subprocess.run([PY, "-m", "rv2.launcher", "init", "--config", c3], cwd=HERE, env=env, capture_output=True)
    bad = "cnn__cnn_baseline__hs_binary__s42"
    r = subprocess.run([PY, "-m", "rv2.launcher", "worker", "--config", c3, "--kinds", "cnn", "--max-jobs", "4"],
                       cwd=HERE, env=dict(env, RV2_INJECT_FAIL=bad, RV2_INJECT_FAIL_N="99"), capture_output=True, text=True)
    cfg3 = load_config(c3)
    j3 = all_jobs(cfg3)
    st3 = {j["id"]: status(cfg3, j) for j in j3}
    dead = L._blocked_forever(j3, st3)
    fs = L.failure_summary(cfg3, j3, st3)
    step("bounded_retry_then_failed_with_summary", st3[bad] == "failed"
         and load_json(os.path.join(y3["out_root"], "_ctl", bad, "ATTEMPTS.json"))["n"] == 3
         and "test__" + bad in dead and "post__report" in dead and any(bad in ln for ln in fs), summary=fs)

    # 10 --------------------------------------------------------------- permuted TEST labels
    perm = os.path.join(SM, "frozen_permuted_test")
    shutil.copytree(frozen, perm)
    df = M.read_manifest(os.path.join(perm, "hs_manifest.tsv"))
    te = df.index[df.split == "test"]
    for col in ("label3", "label_bin"):
        vals = df.loc[te, col].values.copy()
        df.loc[te, col] = np.where(vals == "NEUTRAL", "HATE" if col == "label3" else "HARMFUL", "NEUTRAL")
    M.write_manifest(df, os.path.join(perm, "hs_manifest.tsv"))
    M.write_checksums(perm)
    y2 = copy.deepcopy(y)
    y2["out_root"] = os.path.join(SM, "runs_permuted_test")
    y2["frozen_override"] = perm
    pcfg = os.path.join(SM, "smoke_perm.yaml")
    with open(pcfg, "w") as f:
        yaml.safe_dump(y2, f, allow_unicode=True)
    subprocess.run([PY, "-m", "rv2.launcher", "init", "--config", pcfg], cwd=HERE, env=env, capture_output=True)
    jid = "base__arabertv2__binary__s42"
    r = subprocess.run([PY, "-m", "rv2.run", "--config", pcfg, "--job", jid], cwd=HERE, env=env, capture_output=True, text=True)
    cfgp = load_config(pcfg)
    from rv2.train import run_test
    SEL.run_post_freeze(cfgp, y2["out_root"], [], lambda *a: None)
    tdir = os.path.join(y2["out_root"], "test__" + jid)
    os.makedirs(tdir, exist_ok=True)
    run_test(cfgp, by["test__" + jid], tdir, lambda *a: None, cfgp["smoke"])
    a, b = os.path.join(rr, jid), os.path.join(y2["out_root"], jid)
    ha, hb = load_json(os.path.join(a, "history.json")), load_json(os.path.join(b, "history.json"))
    va, vb = np.load(os.path.join(a, "preds_val.npz")), np.load(os.path.join(b, "preds_val.npz"))
    ta, tb = np.load(os.path.join(rr, "test__" + jid, "preds_test.npz")), np.load(os.path.join(tdir, "preds_test.npz"))
    same_traj = [e["val_macro_f1"] for e in ha] == [e["val_macro_f1"] for e in hb] and \
        load_json(os.path.join(a, "info.json"))["best_epoch"] == load_json(os.path.join(b, "info.json"))["best_epoch"]
    step("test_label_permutation_does_not_change_training_or_selection",
         r.returncode == 0 and same_traj and np.allclose(va["logits"], vb["logits"], atol=1e-5)
         and np.allclose(ta["logits"], tb["logits"], atol=1e-5) and not np.array_equal(ta["y"], tb["y"]),
         val_f1=[e["val_macro_f1"] for e in ha],
         test_f1_original=load_json(os.path.join(rr, "test__" + jid, "metrics_test.json"))["macro_f1"],
         test_f1_permuted=load_json(os.path.join(tdir, "metrics_test.json"))["macro_f1"])

    # 11 --------------------------------------------------------------- resume + SudSenti3 fold jobs + report
    step("resume_skips_done", run_job(cfg, by[jid]) == "skipped (done)")
    for job in ({"id": "sent__sudabert_v2__sudsenti3__f0__s42", "kind": "sent", "gpu": True, "deps": [],
                 "model": "sudabert_v2", "task": "sudsenti3", "fold": 0, "seed": 42},
                {"id": "cnn__scm_mma__sudsenti3__f0__s42", "kind": "cnn", "gpu": True, "deps": [],
                 "model": "scm_mma", "task": "sudsenti3", "fold": 0, "seed": 42}):
        ok = run_job(cfg, job) == "done"
        tj = dict(job, id="test__" + job["id"], kind="test", train_kind=job["kind"], train_id=job["id"], deps=[job["id"]])
        ok = ok and run_job(cfg, tj) == "done"
        pm = load_json(os.path.join(rr, job["id"], "info.json"))["preprocessing"]
        step(f"fold_job_{job['id']}", ok and pm == ("minimal" if job["kind"] == "sent" else "mhamed"), preprocessing=pm)
    rep = load_json(os.path.join(rr, "_report", "summary.json"))
    step("report_written", "binary" in rep and rep["binary"]["systems"],
         systems=sorted(rep["binary"]["systems"]), comparisons=[(c["a"], c["b"], c["p"], c["p_holm"])
                                                                for c in rep["binary"]["comparisons"]])
    R["finished"] = time.strftime("%F %T")
    with open(os.path.join(SM, "SMOKE_RESULT.json"), "w") as f:
        json.dump(R, f, indent=2, ensure_ascii=False, default=str)
    print("SMOKE TEST: ALL PASS")


if __name__ == "__main__":
    main()
