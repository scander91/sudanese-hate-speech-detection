"""Single entry point for one job.

    python -m rv2.run --job base__marbertv2__binary__s42 [--config ...] [--runs DIR]
    python -m rv2.run --list [--kind base]

A training run directory holds: job.json, provenance.json, log.txt, history.json, info.json,
preds_{train,val}.npz, metrics_val.json, ckpt/ (removed after its test job unless kept), DONE.
A test__ run directory holds preds_test.npz / metrics_test.json.
DONE stores the job fingerprint = job + the config sections this job reads + the job's code-group hash
+ frozen-data checksums + pinned model revisions + the fingerprints of its dependencies.  `status()`
accepts a DONE only when its fingerprint equals the current one ('stale' otherwise), and every job
refuses to run unless runs/CAMPAIGN.json (written once by `launcher init`) matches the current code,
config and model revisions.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import threading
import time
import traceback
import uuid

from .common import P, group_code_hash, dump_json, env_info, load_config, load_json, sha256_str
from .jobs import all_jobs

# config sections each job kind reads (job-scoped, so unrelated edits do not invalidate runs)
_SECTIONS = {"base": ["baseline", "keep_checkpoints"], "sent": ["sentiment"], "bilstm": ["bilstm"],
             "distill": ["distill"], "distill4": ["distill"], "mtl": ["mtl"], "cnn": ["cnn"],
             "post_teacher": ["distill", "seeds"], "post_ens": ["ensemble", "sentiment", "seeds"],
             "post_enstest": ["ensemble", "sentiment", "seeds"], "post_mtlsel": ["mtl", "seeds"],
             "post_freeze": ["distill", "ensemble", "mtl", "sentiment", "hs_tasks"],
             "xai": ["xai", "baseline", "models", "model_revisions", "seeds"],
             "xai_mtlcmp": ["xai", "baseline", "mtl", "models", "model_revisions", "seeds"]}
_COMMON = ["data", "preprocessing", "smoke"]
_EXCLUDE_REPORT = ("hosts", "python", "launcher")


def _kind(job):
    return job.get("train_kind") if job["kind"] == "test" else job["kind"]


def code_group(job) -> str:
    k = job["kind"]
    if k == "post_report":
        return "report"
    return "select" if k.startswith("post_") else "train"


def _frozen_checksums(cfg) -> str:
    frozen = cfg.get("_frozen_override") or P(cfg, cfg["frozen_dir"])
    with open(os.path.join(frozen, "CHECKSUMS.sha256")) as f:
        return f.read()


def _models_of(cfg, job):
    k = _kind(job)
    if k in ("base", "sent", "mtl"):
        return [job["model"]]
    if k == "bilstm":
        return [cfg["bilstm"]["encoder"]]
    if k in ("distill", "distill4"):
        return [cfg["distill"]["student"]]
    return []


def _own_fingerprint_payload(cfg, job, dep_fps):
    k = _kind(job)
    if job["kind"] == "post_report":
        c = {kk: v for kk, v in cfg.items() if not kk.startswith("_") and kk not in _EXCLUDE_REPORT}
    else:
        c = {kk: cfg.get(kk) for kk in _COMMON + _SECTIONS.get(k, [])}
    ms = _models_of(cfg, job)
    c["models"] = {m: cfg["models"][m] for m in ms}
    c["model_revisions"] = {m: cfg["model_revisions"][m] for m in ms}
    return {"job": job, "cfg": c, "data": _frozen_checksums(cfg), "code": group_code_hash(code_group(job)),
            "deps": dep_fps}


def fingerprints(cfg, jobs=None) -> dict:
    """Fingerprint of every registered job (memoised per config object)."""
    if "_fp" in cfg:
        return cfg["_fp"]
    jobs = jobs or all_jobs(cfg)
    by = {j["id"]: j for j in jobs}
    fp = {}

    def f(jid):
        if jid not in fp:
            j = by[jid]
            fp[jid] = sha256_str(json.dumps(_own_fingerprint_payload(cfg, j, {d: f(d) for d in j["deps"]}),
                                            sort_keys=True, default=str))
        return fp[jid]
    for jid in by:
        f(jid)
    cfg["_fp"] = fp
    return fp


def fingerprint(cfg, job) -> str:
    fps = fingerprints(cfg)
    if job["id"] in fps and all(d in fps for d in job["deps"]):
        return fps[job["id"]]
    return sha256_str(json.dumps(_own_fingerprint_payload(cfg, job, {d: fps.get(d) for d in job["deps"]}),
                                 sort_keys=True, default=str))          # ad hoc job outside the registry


def runs_root(cfg):
    return cfg.get("_runs_override") or P(cfg, cfg["out_root"])


def ctl_dir(cfg, jid):
    """Control directory of a job (lock, attempts, FAILED, log, stdout).  Results live in runs/<jid>/, which is
    only ever created by ONE atomic rename of a token-scoped staging directory (publish)."""
    return os.path.join(runs_root(cfg), "_ctl", jid)


def stage_dir(rr, jid, token):
    """Token-scoped staging directory (hidden: result globs never match it; dirname == runs root)."""
    return os.path.join(rr, f".stage.{token}.{jid}")


# ------------------------------------------------------------------ campaign manifest
def campaign_payload(cfg) -> dict:
    fps = fingerprints(cfg)
    return {"campaign_sha256": sha256_str(json.dumps(fps, sort_keys=True)),
            "code": {g: group_code_hash(g) for g in ("train", "select", "report")},
            "model_revisions": cfg["model_revisions"], "preprocessing": cfg["preprocessing"],
            "frozen_checksums_sha256": sha256_str(_frozen_checksums(cfg)), "config": cfg["_config_path"],
            "n_jobs": len(fps), "fingerprints": fps}


def campaign_path(cfg):
    return os.path.join(runs_root(cfg), "CAMPAIGN.json")


def init_campaign(cfg) -> dict:
    p = campaign_path(cfg)
    cur = campaign_payload(cfg)
    if os.path.exists(p):
        check_campaign(cfg)
        return load_json(p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    try:                                   # write-once, race-free: O_EXCL on a sibling, then rename
        fd = os.open(p + ".init", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"another process is initialising {p}")
    with os.fdopen(fd, "w") as f:
        json.dump(dict(cur, created=time.strftime("%Y-%m-%dT%H:%M:%S"), host=socket.gethostname()), f, indent=1)
    os.replace(p + ".init", p)
    return cur


def check_campaign(cfg) -> None:
    """Refuse to run when the code/config/model revisions differ from the frozen campaign manifest.
    A successful check is memoised on the config object (one process = one code/config state)."""
    if cfg.get("_campaign_ok"):
        return
    _check_campaign(cfg)
    cfg["_campaign_ok"] = True


def _check_campaign(cfg) -> None:
    p = campaign_path(cfg)
    if not os.path.exists(p):
        raise RuntimeError(f"no campaign manifest {p}: run `python -m rv2.launcher init` first")
    old, cur = load_json(p), campaign_payload(cfg)
    if old["campaign_sha256"] == cur["campaign_sha256"]:
        return
    diff = [k for k in ("code", "model_revisions", "preprocessing", "frozen_checksums_sha256")
            if old.get(k) != cur.get(k)]
    changed = [j for j, f in cur["fingerprints"].items() if old["fingerprints"].get(j) != f]
    raise RuntimeError(f"campaign manifest mismatch ({p}): differs in {diff or ['job config sections']}; "
                       f"{len(changed)} job fingerprints changed, e.g. {changed[:5]}")


# ------------------------------------------------------------------ locks with heartbeat lease
def _owner(run_dir):
    try:
        return load_json(os.path.join(run_dir, ".lock", "owner.json"))
    except Exception:
        return None


def acquire_lock(run_dir):
    """mkdir is atomic on NFS.  Returns the owner token (str) or None.  The owner file carries host/pid/time,
    a random token, and a heartbeat on the owner's clock (never NFS mtimes: the file server clock may differ)."""
    os.makedirs(run_dir, exist_ok=True)
    lk = os.path.join(run_dir, ".lock")
    try:
        os.mkdir(lk)
    except FileExistsError:
        return None
    now = time.time()
    tok = uuid.uuid4().hex
    _write_owner(lk, {"host": socket.gethostname(), "pid": os.getpid(), "time": now, "heartbeat": now, "token": tok})
    return tok


def _write_owner(lk, o) -> bool:
    """Write owner.json INSIDE an existing lock dir; never creates the directory (no lock resurrection)."""
    tmp = os.path.join(lk, f".owner.{socket.gethostname()}.{os.getpid()}.{threading.get_ident()}")
    try:
        with open(tmp, "w") as f:          # fails with FileNotFoundError if the lock dir is gone
            json.dump(o, f)
        os.replace(tmp, os.path.join(lk, "owner.json"))
        return True
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def still_owner(run_dir, token) -> bool:
    o = _owner(run_dir)
    return bool(o) and o.get("token") == token


def release_lock(run_dir, token) -> bool:
    """Delete the lock ONLY if it carries our own token.  Locks are never broken or taken
    over by any worker, so a lock carrying our token is ours for its whole life; a foreign lock is never
    touched (no rename).  The delete itself is rename-to-private then rmtree (NFS silly-rename safe)."""
    lk = os.path.join(run_dir, ".lock")
    if not token or not still_owner(run_dir, token):
        return False
    priv = f"{lk}.release.{token}"
    try:
        os.rename(lk, priv)
    except OSError:
        return False
    shutil.rmtree(priv, ignore_errors=True)
    return True


def lock_age(run_dir):
    """Seconds since the owner's last heartbeat, on the hosts' clocks (None if the owner file is unreadable).
    Reporting only (`launcher status`); a stale heartbeat never leads to automatic lock removal."""
    o = _owner(run_dir)
    if o:
        return time.time() - float(o.get("heartbeat", o.get("time", 0)))
    return None


def owner_alive(host, pid, ssh_fmt=None):
    """True / False if the owner process is known to be running / not running; None if it cannot be verified.
    Local host: os.kill(pid, 0).  Remote: ssh <host> 'ps -p <pid>' (BatchMode, 20 s timeout)."""
    if not host or not pid:
        return None
    me = socket.gethostname()
    if host == me or host.split(".")[0] == me.split(".")[0]:
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    import subprocess
    fmt = ssh_fmt or "ssh -o BatchMode=yes -o ConnectTimeout=10 {host}"
    cmd = os.path.expanduser(fmt.format(host=host.split(".")[0])).split() + [f"ps -p {int(pid)} -o pid= || echo DEAD"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    out = r.stdout.strip()
    if out == "DEAD":
        return False
    return True if out.split() and out.split()[0] == str(int(pid)) else None


def unlock_verified(run_dir, why, ssh_fmt=None) -> tuple[bool, str]:
    """`launcher unlock`: remove a lock only after verifying that its owner PID is not running on the owner
    host.  Refuses when the owner is alive or cannot be verified.  Also discards the dead owner's staging dir."""
    lk = os.path.join(run_dir, ".lock")
    if not os.path.isdir(lk):
        return False, "no lock"
    o = _owner(run_dir)
    if not o:
        return False, "lock has no readable owner.json; inspect by hand"
    alive = owner_alive(o.get("host"), o.get("pid"), ssh_fmt)
    if alive is not False:
        return False, (f"owner {o.get('host')}:{o.get('pid')} is running" if alive else
                       f"cannot verify owner {o.get('host')}:{o.get('pid')} (ssh failed); refusing")
    priv = f"{lk}.unlocked.{o.get('token')}"
    try:
        os.rename(lk, priv)
    except OSError as e:
        return False, f"rename failed: {e}"
    moved = load_json(os.path.join(priv, "owner.json")) if os.path.exists(os.path.join(priv, "owner.json")) else {}
    if moved.get("token") != o.get("token"):        # cannot happen (no one else removes locks): put it back
        os.rename(priv, lk)
        return False, "lock changed during unlock; restored"
    tok = o.get("token")
    if tok and os.path.basename(os.path.dirname(run_dir)) == "_ctl":
        rr = os.path.dirname(os.path.dirname(run_dir))
        shutil.rmtree(stage_dir(rr, os.path.basename(run_dir), tok), ignore_errors=True)
    with open(os.path.join(run_dir, "lock_recovery.log"), "a") as f:
        f.write(f"{time.strftime('%F %T')} {socket.gethostname()}:{os.getpid()} unlocked (owner "
                f"{o.get('host')}:{o.get('pid')} verified dead): {why}\n")
    shutil.rmtree(priv, ignore_errors=True)
    return True, f"unlocked; owner {o.get('host')}:{o.get('pid')} verified not running"


class _Heartbeat(threading.Thread):
    """Refreshes the heartbeat (for `launcher status` reporting only) while the lock carries our token; never
    recreates the lock.  Sets `lost` if the lock vanished (only possible by operator error), so the job
    refuses to publish."""

    def __init__(self, run_dir, token, every):
        super().__init__(daemon=True)
        self.rd, self.token, self.every = run_dir, token, every
        self.stop, self.lost = threading.Event(), threading.Event()

    def run(self):
        lk = os.path.join(self.rd, ".lock")
        while not self.stop.wait(self.every):
            o = _owner(self.rd)
            if not o or o.get("token") != self.token:
                self.lost.set()
                return
            o["heartbeat"] = time.time()
            if not _write_owner(lk, o):
                self.lost.set()
                return

    def halt(self):
        self.stop.set()
        self.join()


# ------------------------------------------------------------------ status
def max_attempts(cfg):
    return int(cfg.get("launcher", {}).get("max_attempts", 3))


def attempts(rd) -> int:
    p = os.path.join(rd, "ATTEMPTS.json")
    return load_json(p)["n"] if os.path.exists(p) else 0


def record_failure(cfg, rd, msg):
    p = os.path.join(rd, "ATTEMPTS.json")
    a = load_json(p) if os.path.exists(p) else {"n": 0, "errors": []}
    a["n"] += 1
    a["errors"].append({"time": time.strftime("%F %T"), "host": socket.gethostname(), "error": msg[-2000:]})
    dump_json(a, p)
    with open(os.path.join(rd, "FAILED"), "w") as f:
        f.write(msg)


def status(cfg, job, check=True) -> str:
    """done | stale (DONE with another fingerprint) | running | retry | failed | todo.
    Validates runs/CAMPAIGN.json first (raises on a missing/mismatching manifest once any result exists).
    check=False is a diagnostic only (fingerprint scoping tests); no scheduler path uses it."""
    rd = os.path.join(runs_root(cfg), job["id"])
    cd = ctl_dir(cfg, job["id"])
    d = os.path.join(rd, "DONE")
    if check and (os.path.exists(campaign_path(cfg)) or os.path.exists(d)):
        check_campaign(cfg)
    if os.path.exists(d):
        try:
            ok = load_json(d).get("fingerprint") == fingerprint(cfg, job)
        except Exception:
            ok = False
        return "done" if ok else "stale"
    if os.path.isdir(os.path.join(cd, ".lock")):
        return "running"
    if os.path.exists(os.path.join(cd, "FAILED")):
        return "retry" if attempts(cd) < max_attempts(cfg) else "failed"
    return "todo"


def deps_done(cfg, job) -> bool:
    check_campaign(cfg)
    by = {j["id"]: j for j in all_jobs(cfg)}
    return all(status(cfg, by[d]) == "done" if d in by else os.path.exists(os.path.join(runs_root(cfg), d, "DONE"))
               for d in job["deps"])


def host_profile(cfg, host=None):
    return (cfg.get("hosts", {}) or {}).get((host or socket.gethostname()).split(".")[0], {}) or {}


def host_excludes(profile, job) -> bool:
    ex = profile.get("exclude_kinds") or []
    return job["kind"] in ex or (job["kind"] == "test" and job.get("train_kind") in ex)


def _host_allows(cfg, job):
    """Hard routing: a host whose profile excludes a kind never runs it, even by hand."""
    if host_excludes(host_profile(cfg), job):
        raise RuntimeError(f"{job['id']}: kind excluded on host {socket.gethostname()} (hosts.*.exclude_kinds)")


def run_job(cfg, job, smoke=None, force=False, lock=True) -> str:
    """Fenced execution: all artifacts are written into a token-scoped staging dir; the
    result dir runs/<jid>/ appears only through one atomic rename, done while the job still holds its lock
    (token re-checked immediately before).  A stale worker whose lease was broken can never overwrite or
    mix artifacts: its staging dir is private and discarded, and the rename fails if a result exists."""
    rr = runs_root(cfg)
    rd = os.path.join(rr, job["id"])
    cd = ctl_dir(cfg, job["id"])
    smoke = smoke or cfg.get("smoke")
    check_campaign(cfg)
    fp = fingerprint(cfg, job)
    done = os.path.join(rd, "DONE")

    def done_state():
        if not os.path.exists(done):
            return None
        return "same" if load_json(done).get("fingerprint") == fp else "stale"
    ds = done_state()
    if ds == "same" and not force:
        return "skipped (done)"
    if ds == "stale" and not force:
        raise RuntimeError(f"{job['id']}: existing result has a different fingerprint (stale); use --force")
    if not deps_done(cfg, job):
        return "blocked (deps)"
    _host_allows(cfg, job)
    os.makedirs(cd, exist_ok=True)
    token = acquire_lock(cd) if lock else uuid.uuid4().hex     # never breakable: only its owner releases it
    if lock and not token:
        return "locked"
    if lock and not force and done_state() == "same":        # re-check DONE inside the lock
        release_lock(cd, token)
        return "skipped (done)"
    hb = None
    if lock:
        hb = _Heartbeat(cd, token, cfg.get("launcher", {}).get("heartbeat_seconds", 60)); hb.start()
    sd = stage_dir(rr, job["id"], token)
    shutil.rmtree(sd, ignore_errors=True)
    os.makedirs(sd)
    logf = open(os.path.join(cd, "log.txt"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    def owned():
        return not lock or (not hb.lost.is_set() and still_owner(cd, token))
    try:
        if os.path.exists(os.path.join(cd, "FAILED")):
            os.remove(os.path.join(cd, "FAILED"))
        dump_json(job, os.path.join(sd, "job.json"))
        frozen = cfg.get("_frozen_override") or P(cfg, cfg["frozen_dir"])
        dump_json({"env": env_info(), "code_group": code_group(job), "code_sha256": group_code_hash(code_group(job)),
                   "fingerprint": fp, "campaign_sha256": load_json(os.path.join(rr, "CAMPAIGN.json"))["campaign_sha256"],
                   "config": cfg["_config_path"], "frozen_dir": frozen,
                   "frozen_checksums": _frozen_checksums(cfg).splitlines(),
                   "preprocessing": cfg["preprocessing"], "smoke": smoke,
                   "model_revisions": {m: cfg["model_revisions"][m] for m in _models_of(cfg, job)},
                   "selection_signal": "validation macro-F1 only; test predicted only by test__ jobs after selections are frozen",
                   "attempt": attempts(cd) + 1, "lock_token": token,
                   "env_vars": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED",
                                                               "CUBLAS_WORKSPACE_CONFIG", "HF_HUB_OFFLINE")}},
                  os.path.join(sd, "provenance.json"))
        log(f"START {job['id']} (token {token[:8]})")
        if os.environ.get("RV2_INJECT_FAIL") == job["id"] and attempts(cd) < int(os.environ.get("RV2_INJECT_FAIL_N", "1")):
            raise RuntimeError("injected failure (smoke test of retry)")      # test hook, inert unless set
        if os.environ.get("RV2_INJECT_KILL") == job["id"] and attempts(cd) < 1:
            os.kill(os.getpid(), 9)             # test hook: die holding the lock (like SIGILL / OOM kill)
        t0 = time.time()
        k = job["kind"]
        if k.startswith("post_"):
            from . import select, evaluate
            fn = {"post_teacher": select.run_post_teacher, "post_ens": select.run_post_ens,
                  "post_enstest": select.run_post_enstest, "post_mtlsel": select.run_post_mtlsel}.get(k)
            if k == "post_report":
                evaluate.build_report(cfg, rr, log)
            elif k == "post_freeze":
                select.run_post_freeze(cfg, rr, select.expected_selections(cfg), log)
            else:
                fn(cfg, rr, job["task"], log)
            info = {"kind": k}
        elif k in ("xai", "xai_mtlcmp"):
            from . import xai
            info = (xai.run_xai if k == "xai" else xai.run_xai_mtlcmp)(cfg, job, sd, log, smoke)
        elif k == "test":
            from .train import run_test
            info, res = run_test(cfg, job, sd, log, smoke)
            log(f"test_macro_f1={res['test']['macro_f1']:.4f} val_check={info.get('val_check')}")
        else:
            from .train import KINDS
            info, res = KINDS[k](cfg, job, sd, log, smoke)
            if "history" in info:
                dump_json(info.pop("history"), os.path.join(sd, "history.json"))
                log(f"best_epoch={info['best_epoch']} best_val_macro_f1={info['best_val_macro_f1']:.4f}")
        cleanup = info.pop("_cleanup", [])
        info["wall_seconds"] = time.time() - t0
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
            info["cuda_max_mem_allocated_gb"] = sys.modules["torch"].cuda.max_memory_allocated() / 2 ** 30
        dump_json(info, os.path.join(sd, "info.json"))
        dump_json({"fingerprint": fp, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "host": socket.gethostname(), "token": token}, os.path.join(sd, "DONE"))
        logf.flush()
        shutil.copy2(os.path.join(cd, "log.txt"), os.path.join(sd, "log.txt"))
        # ---- publish: token re-checked immediately before the single atomic rename
        if not owned():
            raise RuntimeError("lock ownership lost during the run (lease broken); refusing to publish")
        if os.path.lexists(rd):
            if not force and os.path.exists(done):
                raise RuntimeError(f"{job['id']}: a result was published concurrently; refusing to overwrite")
            trash = os.path.join(rr, ".trash", f"{job['id']}.{int(time.time())}.{token[:8]}")
            os.makedirs(os.path.dirname(trash), exist_ok=True)
            os.rename(rd, trash)                # previous (stale / partial) result kept aside, never mixed
        os.rename(sd, rd)                       # fails if another publisher won the race: no overwrite, no mix
        if owned():                             # side effects on OTHER runs only while still the owner
            for c in cleanup:                   # checkpoints no longer needed once the test is materialised
                shutil.rmtree(c, ignore_errors=True) if os.path.isdir(c) else (os.path.exists(c) and os.remove(c))
        log(f"DONE {job['id']} in {info['wall_seconds']:.0f}s (published)")
        return "done"
    except Exception:
        tb = traceback.format_exc()
        log("FAILED\n" + tb)
        record_failure(cfg, cd, tb)
        raise
    finally:
        shutil.rmtree(sd, ignore_errors=True)   # unpublished staging never survives the attempt
        if hb:
            hb.halt()                       # join before release: no heartbeat write after the lock is gone
        logf.close()
        if lock:
            release_lock(cd, token)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--job")
    ap.add_argument("--runs", help="override runs root")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--kind")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.runs:
        cfg["_runs_override"] = a.runs
    if cfg.get("frozen_override"):
        cfg["_frozen_override"] = os.path.expanduser(cfg["frozen_override"])
    jobs = {j["id"]: j for j in all_jobs(cfg)}
    if a.list:
        for j in jobs.values():
            if not a.kind or j["kind"] == a.kind:
                print(j["id"], status(cfg, j))
        return
    print(run_job(cfg, jobs[a.job], force=a.force))


if __name__ == "__main__":
    main()
