"""GPU smoke gate: base (fp16), BiLSTM and MTL on one real GPU, minutes each.

    CUDA_VISIBLE_DEVICES=<g> python tests/gpu_gate.py --tag host1_gpu0

Each job runs as `python -m rv2.run` (same path as a campaign worker) on a stratified subset
(train 2000 / val 500 / test 500, 2 epochs, max_len 64) under its own campaign manifest in
gate_runs/<tag>/, then its test phase is materialised from the checkpoint (reload + validation
re-check).  PASS requires cuda: true in provenance, provenance fingerprint == campaign fingerprint, the loaded
model revision (HF commit or local directory sha256) present and equal to the pin, fp16 autocast
actually used exactly where the config asks for it (base: yes; bilstm/mtl: no), validation macro-F1
above a non-trivial threshold, and the reloaded checkpoint reproducing the validation predictions.
On a host whose profile excludes a kind (e.g. mtl) the gate instead requires rv2.run to refuse it.  Writes gate_runs/<tag>/GATE_RESULT.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from rv2.common import load_config, load_json   # noqa: E402
from rv2.jobs import all_jobs                    # noqa: E402
from rv2.run import init_campaign, run_job       # noqa: E402

JOBS = ["base__marbertv2__binary__s42", "bilstm__binary__s42", "mtl__marbertv2__binary__a0.7__s42"]
MIN_VAL_F1 = 0.55        # binary macro-F1: majority-class prediction gives <=0.39 on this val subset; bar = +0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out = os.path.join(HERE, "gate_runs", a.tag)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    y = yaml.safe_load(open(os.path.join(HERE, "configs", "default.yaml")))
    y["out_root"] = os.path.join(out, "runs")
    y["seeds"] = [42]
    y["hs_tasks"] = ["binary"]
    y["mtl"].update(encoders=["marbertv2"], alphas=[0.7])
    y["xai"].update(compare_encoder="marbertv2",                      # gate: XAI on real settings, fewer examples
                    shap=dict(y["xai"]["shap"], n_samples=20), lime=dict(y["xai"]["lime"], n_examples=4))
    y["smoke"] = {"n": {"train": 2000, "val": 500, "test": 500}, "hp": {"epochs": 2, "patience": 1, "max_len": 64}}
    cpath = os.path.join(out, "gate.yaml")
    yaml.safe_dump(y, open(cpath, "w"), allow_unicode=True)
    cfg = load_config(cpath)
    camp = init_campaign(cfg)
    by = {j["id"]: j for j in all_jobs(cfg)}
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    R = {"tag": a.tag, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "jobs": {}}
    from rv2.run import host_excludes, host_profile
    prof = host_profile(cfg)
    ok_all = True
    trained = []
    for jid in JOBS:
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "rv2.run", "--config", cpath, "--job", jid], cwd=HERE, env=env,
                           capture_output=True, text=True)
        rec = {"returncode": r.returncode, "wall_seconds_incl_load": round(time.time() - t0, 1)}
        if host_excludes(prof, by[jid]):          # routing: this host must refuse the kind (e.g. mtl)
            rec["expected_refusal"] = True
            rec["pass"] = r.returncode != 0 and "kind excluded on host" in r.stderr
            rec["stderr_tail"] = r.stderr.strip().splitlines()[-1][-300:]
            ok_all &= rec["pass"]
            R["jobs"][jid] = rec
            print(jid, json.dumps(rec), flush=True)
            continue
        if r.returncode != 0:
            rec.update(stderr_tail=r.stderr[-2000:], **{"pass": False})
            R["jobs"][jid] = rec; ok_all = False
            continue
        R["jobs"][jid] = rec
        trained.append(jid)
    # freeze barrier (gate only: the MTL alpha choice is trivial here), then the test phases
    from rv2.select import run_post_freeze, write_selection
    names = []
    if "mtl__marbertv2__binary__a0.7__s42" in trained:
        write_selection(cfg["out_root"], "mtl_alpha_marbertv2_binary",
                        {"chosen": 0.7, "inputs_sha256": "gpu-gate", "what": "gate only"})
        names.append("mtl_alpha_marbertv2_binary")
    if "base__marbertv2__binary__s42" in trained:
        write_selection(cfg["out_root"], "best_single_binary", {"chosen": "marbertv2", "inputs_sha256": "gpu-gate"})
        names.append("best_single_binary")
    run_post_freeze(cfg, cfg["out_root"], names, lambda *x: None)
    from rv2.train import run_test
    for jid in trained:
        rec = R["jobs"][jid]
        rd = os.path.join(cfg["out_root"], jid)
        prov, info = load_json(os.path.join(rd, "provenance.json")), load_json(os.path.join(rd, "info.json"))
        tj = by["test__mtl__marbertv2__binary__s42"] if jid.startswith("mtl") else by["test__" + jid]
        td = os.path.join(cfg["out_root"], tj["id"])
        os.makedirs(td, exist_ok=True)
        t1 = time.time()
        tinfo, tres = run_test(cfg, tj, td, lambda *x: None, cfg["smoke"])
        want_fp16 = bool(info["hp"].get("fp16"))
        rec.update(
            cuda=prov["env"].get("cuda"), gpu=prov["env"].get("gpu"), host=prov["env"].get("host"),
            torch=prov["env"].get("torch"), fingerprint_ok=prov["fingerprint"] == camp["fingerprints"][jid],
            campaign_ok=prov["campaign_sha256"] == camp["campaign_sha256"],
            pinned_revision=info.get("pinned_revision"), loaded_revision=info.get("model_revision"),
            revision_ok=info.get("model_revision") is not None and info.get("model_revision") == info.get("pinned_revision"),
            fp16_config=want_fp16, amp_fp16_used=info.get("amp_fp16_used"),
            fp16_ok=info.get("amp_fp16_used") is want_fp16 and info.get("device") == "cuda",
            train_wall_seconds=round(info["wall_seconds"], 1),
            epochs_run=info["epochs_run"], best_val_macro_f1=info["best_val_macro_f1"],
            val_history=[h["val_macro_f1"] for h in load_json(os.path.join(rd, "history.json"))],
            cuda_max_mem_gb=round(info.get("cuda_max_mem_allocated_gb", 0), 2),
            test_phase_seconds=round(time.time() - t1, 1), test_val_check=tinfo.get("val_check"),
            test_macro_f1_gate_subset=tres["test"]["macro_f1"])
        rec["pass"] = bool(rec["cuda"] and rec["fingerprint_ok"] and rec["campaign_ok"] and rec["revision_ok"]
                           and rec["fp16_ok"] and rec["best_val_macro_f1"] >= MIN_VAL_F1
                           and rec["test_val_check"]["val_argmax_agreement"] >= 0.99)
        ok_all &= rec["pass"]
        print(jid, json.dumps({k: rec[k] for k in ("pass", "gpu", "amp_fp16_used", "revision_ok", "train_wall_seconds",
                                                   "cuda_max_mem_gb", "best_val_macro_f1", "test_val_check")}), flush=True)
    # XAI on the GPU (in-process; registry deps on the 9 baselines are bypassed on purpose: gate only)
    from rv2 import xai
    for kind, fn in (("xai", xai.run_xai), ("xai_mtlcmp", xai.run_xai_mtlcmp)):
        jid = f"{kind}__binary"
        if kind == "xai_mtlcmp" and "mtl__marbertv2__binary__a0.7__s42" not in trained:
            continue
        xd = os.path.join(cfg["out_root"], jid); os.makedirs(xd, exist_ok=True)
        t2 = time.time()
        try:
            xi = fn(cfg, by[jid], xd, lambda *x: None, cfg["smoke"])
            rc = xi.get("reload_check", {})
            ok = all(v["argmax_agreement_with_test_job"] >= 0.99 for v in (rc.values() if kind == "xai_mtlcmp" else [rc]))
            rec = {"pass": ok, "seconds": round(time.time() - t2, 1), "info": xi}
        except Exception as e:
            import traceback
            rec = {"pass": False, "error": traceback.format_exc()[-1500:]}
        ok_all &= rec["pass"]
        R["jobs"][jid] = rec
        print(jid, json.dumps({k: rec.get(k) for k in ("pass", "seconds")}), flush=True)
    R["pass"] = ok_all
    json.dump(R, open(os.path.join(out, "GATE_RESULT.json"), "w"), indent=2)
    print("GPU GATE:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
