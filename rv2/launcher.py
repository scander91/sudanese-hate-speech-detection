"""Queue / launcher over the hosts listed in config `hosts` (shared filesystem).

    python -m rv2.launcher init                       # freeze runs/CAMPAIGN.json (once, before any worker)
    python -m rv2.launcher plan                       # run counts + GPU-hour estimates
    python -m rv2.launcher status                     # per-kind done/stale/running/retry/failed/todo + failures
    python -m rv2.launcher worker --gpu 0 [--host host1] [--kinds mtl,base]   # run on THIS host
    python -m rv2.launcher start --host host2 [--gpus 0,1] [--kinds ...]      # ssh + one nohup worker per GPU
    python -m rv2.launcher unlock --job ID            # remove a stale lock (after checking the owner)

Workers pick the next runnable job (deps DONE with a valid fingerprint, no lock, no valid DONE,
fewer than launcher.max_attempts failures); each job runs in a fresh subprocess
(`python -m rv2.run --job ID`) pinned with CUDA_VISIBLE_DEVICES.  Order: CPU post jobs and test
jobs first (they unblock others and free checkpoints), then training jobs longest-estimated first;
the host profile (hosts.<h>.kinds_first / exclude_kinds) routes kinds to hosts (e.g. all MTL to one host).  GPU workers also run
runnable CPU post jobs.  Locks are directories (atomic mkdir) with an owner heartbeat; no lock is ever
broken or taken over automatically:
`status` lists locks whose heartbeat is older than launcher.lock_lease_seconds, and
`unlock --job ID` clears one only after verifying (locally or over ssh) that the owner PID is dead.  Workers wait while unfinished upstream work exists and exit with a
failure summary when only permanently failed/blocked jobs remain.  Every worker refuses to start
unless runs/CAMPAIGN.json matches the current code/config/model revisions.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict

from .common import RERUN_DIR, load_config, load_json
from .jobs import all_jobs, job_minutes, est_minutes
from .run import (_owner, attempts, check_campaign, ctl_dir, host_excludes, host_profile, init_campaign,
                  lock_age, max_attempts, record_failure, runs_root, status, unlock_verified)

KIND_ORDER = ["base", "sent", "bilstm", "mtl", "cnn", "distill", "distill4", "test", "post_teacher", "post_ens",
              "post_mtlsel", "post_freeze", "post_enstest", "xai", "xai_mtlcmp", "post_report"]
SSH = "ssh -o BatchMode=yes {host}"          # default; overridden by config launcher.ssh


def _speed(cfg, h):
    """Assumed relative throughput of host h vs the reference GPU of the timing estimates (hosts.<h>.speed).
    ASSUMPTION (not measured) -- replace with measured wall_seconds after the first jobs (see `status`)."""
    return cfg["hosts"][h].get("speed", 1.0)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cpu_safe_env() -> dict:
    """On CPUs with AVX but no AVX2, MKL can die with SIGILL (mkl_vml_kernel_sSqrt).  Pin MKL to its AVX code
    path on CPUs without AVX2; no effect elsewhere."""
    try:
        flags = open("/proc/cpuinfo").read()
    except OSError:
        return {}
    return {} if " avx2" in flags else {"MKL_ENABLE_INSTRUCTIONS": "AVX", "MKL_CBWR": "AVX"}


def stale_locks(cfg):
    """Locks whose heartbeat is older than launcher.lock_lease_seconds (REPORT ONLY; never removed
    automatically -- use `launcher unlock --job ID`, which verifies that the owner PID is dead)."""
    lease = cfg.get("launcher", {}).get("lock_lease_seconds", 1800)
    cr = os.path.join(runs_root(cfg), "_ctl")
    out = []
    for d in sorted(os.listdir(cr)) if os.path.isdir(cr) else []:
        rd = os.path.join(cr, d)
        if not os.path.isdir(os.path.join(rd, ".lock")):
            continue
        o = _owner(rd) or {}
        age = lock_age(rd)
        if age is None or age > lease:
            out.append({"job": d, "owner_host": o.get("host"), "owner_pid": o.get("pid"),
                        "heartbeat_age_s": None if age is None else round(age)})
    return out


def _prio(job, est, profile):
    """Sort key: CPU post jobs, then test jobs, then kinds_first, other training longest-first, kinds_last."""
    k = job["kind"]
    if not job["gpu"]:
        return (0, 0.0, job["id"])
    if k == "test":
        return (1, 0.0, job["id"])
    if k in (profile.get("kinds_first") or []):
        return (2, -job_minutes(job, est), job["id"])
    if k in (profile.get("kinds_last") or []):
        return (4, -job_minutes(job, est), job["id"])
    return (3, -job_minutes(job, est), job["id"])


def runnable(cfg, jobs, gpu: bool, kinds=None, profile=None, est=None, st=None):
    """Runnable jobs for a worker, in scheduling order.  A GPU worker also takes CPU post jobs."""
    st = st or {j["id"]: status(cfg, j) for j in jobs}
    est = est if est is not None else {}
    out = []
    for j in jobs:
        if st[j["id"]] not in ("todo", "retry"):
            continue
        if j["gpu"] and not gpu:
            continue
        if kinds and j["gpu"] and j["kind"] not in kinds and not (j["kind"] == "test" and j.get("train_kind") in kinds):
            continue
        if host_excludes(profile or {}, j):          # hard routing: e.g. a host that never takes mtl (or its tests)
            continue
        if all(st.get(d) == "done" for d in j["deps"]):
            out.append(j)
    out.sort(key=lambda j: _prio(j, est, profile or {}))
    return out, st


def _blocked_forever(jobs, st):
    """Jobs that can never run: stale/failed, or depending (transitively) on such a job."""
    by = {j["id"]: j for j in jobs}
    memo = {}

    def dead(jid):
        if jid not in memo:
            memo[jid] = False
            s = st[jid]
            memo[jid] = s in ("failed", "stale") or (s != "done" and any(dead(d) for d in by[jid]["deps"]))
        return memo[jid]
    return {jid for jid in by if dead(jid)}


def failure_summary(cfg, jobs, st):
    rr = runs_root(cfg)
    lines = []
    for j in jobs:
        if st[j["id"]] in ("failed", "retry", "stale"):
            rd = ctl_dir(cfg, j["id"])
            err = ""
            p = os.path.join(rd, "FAILED")
            if os.path.exists(p):
                tail = [ln for ln in open(p, errors="replace").read().strip().splitlines() if ln.strip()]
                err = tail[-1][:200] if tail else ""
            lines.append(f"  {st[j['id']]:<6} attempts={attempts(rd)}/{max_attempts(cfg)} {j['id']}: {err}")
    return lines


def worker(cfg, gpu_idx, kinds, cpu, max_jobs, config_path, host=None):
    check_campaign(cfg)
    profile = host_profile(cfg, host)
    if host and host.split(".")[0] != socket.gethostname().split(".")[0] and not cfg.get("smoke"):
        raise RuntimeError(f"--host {host} does not match this machine ({socket.gethostname()})")
    est = est_minutes(cfg)
    poll = cfg.get("launcher", {}).get("poll_seconds", 60)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    env.update(cpu_safe_env())
    if gpu_idx is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    n = 0
    while max_jobs is None or n < max_jobs:
        check_campaign(cfg)
        jobs = all_jobs(cfg)
        st = {j["id"]: status(cfg, j) for j in jobs}
        ready, st = runnable(cfg, jobs, gpu=not cpu, kinds=kinds, profile=profile, est=est)
        if not ready:
            mine = [j for j in jobs if (not j["gpu"] or not cpu) and not host_excludes(profile, j)
                    and (not j["gpu"] or not kinds or j["kind"] in kinds or j.get("train_kind") in kinds)]
            dead = _blocked_forever(jobs, st)
            waiting = [j for j in mine if st[j["id"]] not in ("done",) and j["id"] not in dead]
            if not waiting:
                fs = failure_summary(cfg, jobs, st)
                print(time.strftime("%F %T"), "nothing left for this worker" +
                      (f"; {len(fs)} failed/stale jobs and {len(dead)} jobs blocked by them:\n" + "\n".join(fs) if fs else ""),
                      flush=True)
                return
            time.sleep(poll); continue                  # dependency-aware wait: upstream work is unfinished
        j = ready[0]
        cmd = [sys.executable, "-m", "rv2.run", "--job", j["id"]] + (["--config", config_path] if config_path else [])
        print(time.strftime("%F %T"), "RUN", j["id"], flush=True)
        rd = ctl_dir(cfg, j["id"])
        os.makedirs(rd, exist_ok=True)
        with open(os.path.join(rd, "stdout.txt"), "ab") as out:      # never inherit a pipe
            child = subprocess.Popen(cmd, cwd=RERUN_DIR, env=env, stdout=out, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL)
            rc = child.wait()                                         # reaped: child.pid is verifiably not running
        r = subprocess.CompletedProcess(cmd, rc)
        after = status(cfg, j)
        if r.returncode != 0 and after in ("todo", "running"):   # died without writing FAILED (OOM kill, signal)
            record_failure(cfg, rd, f"subprocess exit code {r.returncode} without a FAILED record")
            o = _owner(rd) or {}
            if os.path.isdir(os.path.join(rd, ".lock")) and o.get("pid") == child.pid \
                    and o.get("host") == socket.gethostname():
                # Only the lock of THIS worker's own reaped child, through the same verified path as
                # `launcher unlock` (owner PID checked not running); never age/lease based.
                ok, msg = unlock_verified(rd, f"worker: own child pid {child.pid} exited {rc}")
                print(time.strftime("%F %T"), f"child lock {j['id']}: {msg}", flush=True)
            elif os.path.isdir(os.path.join(rd, ".lock")):
                print(time.strftime("%F %T"), f"lock of {j['id']} is not this worker's child; left in place "
                      f"(use `launcher unlock --job {j['id']}`)", flush=True)
        print(time.strftime("%F %T"), "EXIT", r.returncode, j["id"], status(cfg, j), flush=True)
        n += 1


def plan(cfg):
    jobs = all_jobs(cfg)
    est = est_minutes(cfg)
    by = defaultdict(lambda: [0, 0.0])
    for j in jobs:
        by[j["kind"]][0] += 1
        by[j["kind"]][1] += job_minutes(j, est) / 60.0
    tot = sum(v[1] for v in by.values())
    print(f"{'kind':<12}{'runs':>6}{'ref-GPU-h':>12}")
    for k in KIND_ORDER:
        if k in by:
            print(f"{k:<12}{by[k][0]:>6}{by[k][1]:>12.1f}")
    print(f"{'TOTAL':<12}{sum(v[0] for v in by.values()):>6}{tot:>12.1f}")
    cap = {h: _speed(cfg, h) * len(v.get("gpus", [])) for h, v in cfg["hosts"].items()}
    C = sum(cap.values()) or 1.0
    print("\nper-host share (assumed relative speeds, hosts.<h>.speed):")
    for h, c in cap.items():
        share = tot * c / C
        g = len(cfg["hosts"][h].get("gpus", []))
        print(f"  {h:<6} gpus={g} speed={_speed(cfg, h)}  work={share:.1f} ref-GPU-h  "
              f"-> {share / max(_speed(cfg, h) or 1, 1e-9):.1f} GPU-h on host, wall ~{(tot / C):.1f} h")
    return by, est


def show_status(cfg):
    jobs = all_jobs(cfg)
    try:
        check_campaign(cfg); print("campaign manifest: OK")
    except RuntimeError as e:
        print(f"campaign manifest: {e}")
    c = defaultdict(Counter)
    walls = defaultdict(list)
    st = {j["id"]: status(cfg, j) for j in jobs}
    for j in jobs:
        s = st[j["id"]]
        c[j["kind"]][s] += 1
        if s == "done" and j["gpu"]:
            p = os.path.join(runs_root(cfg), j["id"], "info.json")
            if os.path.exists(p):
                walls[j["kind"]].append(load_json(p).get("wall_seconds", 0) / 60)
    for k in KIND_ORDER:
        if k in c:
            m = f"  measured mean {sum(walls[k]) / len(walls[k]):.1f} min/run" if walls[k] else ""
            print(f"{k:<12} {dict(c[k])}{m}")
    fs = failure_summary(cfg, jobs, st)
    if fs:
        print("failures / stale:\n" + "\n".join(fs))
    sl = stale_locks(cfg)
    if sl:
        print("locks with a stale heartbeat (NOT removed; check the owner, then `launcher unlock --job ID`):")
        for x in sl:
            print(f"  {x['job']}: owner {x['owner_host']}:{x['owner_pid']} heartbeat age {x['heartbeat_age_s']} s")
    return st


def worker_command(cfg, host, g, config_path, kinds=None, max_jobs=None):
    """Remote worker command line; forwards --host, --kinds, --max-jobs and --config."""
    py = cfg["python"]
    args = f" --gpu {g} --host {host}"
    if config_path:
        args += f" --config {os.path.abspath(config_path)}"
    if kinds:
        args += f" --kinds {','.join(kinds)}"
    if max_jobs:
        args += f" --max-jobs {max_jobs}"
    tag = f"{host}_gpu{g}" + (f"_{'-'.join(kinds)}" if kinds else "")
    return f"cd {RERUN_DIR} && nohup {py} -m rv2.launcher worker{args} > logs/worker_{tag}.log 2>&1 < /dev/null &"


def start(cfg, host, gpus, config_path, kinds=None, max_jobs=None, dry=False):
    check_campaign(cfg)
    logs = os.path.join(RERUN_DIR, "logs")
    os.makedirs(logs, exist_ok=True)
    cmds = []
    for g in gpus:
        cmd = cfg.get("launcher", {}).get("ssh", SSH).format(host=host).split() + [worker_command(cfg, host, g, config_path, kinds, max_jobs)]
        print(" ".join(cmd))
        cmds.append(cmd)
        if not dry:
            subprocess.run(cmd, check=True)
    return cmds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "plan", "status", "worker", "start", "unlock"])
    ap.add_argument("--config")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--gpus")
    ap.add_argument("--host")
    ap.add_argument("--kinds")
    ap.add_argument("--cpu", action="store_true", help="worker runs the CPU post-hoc jobs")
    ap.add_argument("--max-jobs", type=int)
    ap.add_argument("--job")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    kinds = a.kinds.split(",") if a.kinds else None
    if a.cmd == "init":
        c = init_campaign(cfg)
        print(f"campaign {c['campaign_sha256']}  jobs={c['n_jobs']}  -> {os.path.join(runs_root(cfg), 'CAMPAIGN.json')}")
    elif a.cmd == "plan":
        plan(cfg)
    elif a.cmd == "status":
        show_status(cfg)
    elif a.cmd == "worker":
        worker(cfg, a.gpu, kinds, a.cpu, a.max_jobs, a.config, a.host)
    elif a.cmd == "start":
        gp = [int(x) for x in a.gpus.split(",")] if a.gpus else cfg["hosts"][a.host]["gpus"]
        start(cfg, a.host, gp, a.config, kinds, a.max_jobs, a.dry_run)
    elif a.cmd == "unlock":
        ok, msg = unlock_verified(ctl_dir(cfg, a.job), "launcher unlock", cfg.get("launcher", {}).get("ssh_check"))
        print(("UNLOCKED " if ok else "REFUSED ") + a.job + ": " + msg)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
