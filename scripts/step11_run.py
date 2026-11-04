"""Step 11 runner: extracts Olmo-3-1025-7B then Olmo-3-7B-RL-Zero-Math one at a time as a child process; polls the child's RSS every 10 s
from /proc and kills that one pid if RSS > 20 GB (never pkill). Writes data/step11_rss.json. Usage: uv run python scripts/step11_run.py"""
import json, os, signal, subprocess, sys, time, tomllib
from rtp import lineage
P = tomllib.load(open("configs/pilot.toml", "rb"))["hosts"]["olmo3_7b"]
H = "/mnt/e/ai-models/huggingface_cache/hub/models--allenai--"
BASE = ("olmo3-7b-base", H + "Olmo-3-1025-7B/snapshots/a81bae42db3975be1671e27b9c9a56da1a9f980f", "a81bae42db3975be1671e27b9c9a56da1a9f980f")
RLZ = ("olmo3-7b-rlzero", H + "Olmo-3-7B-RL-Zero-Math/snapshots/dee26a4074b88215cd1326472e1ab14d0259d892", "dee26a4074b88215cd1326472e1ab14d0259d892")
assert BASE[2].startswith(P["base"]["sha12"]) and RLZ[2].startswith(P["rl_zero_math"]["sha12"])
lineage.require_pass(BASE[2], RLZ[2])
env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"); rec = {}
for name, snap, sha in (BASE, RLZ):
    if os.path.exists(f"runs/{name}/acts.npz"):
        continue
    ch = subprocess.Popen(["nice", "-n", "10", sys.executable, "scripts/step11_extract.py", snap, sha, f"runs/{name}", "--gate", f"{BASE[2]}:{RLZ[2]}"], env=env,
                          stdout=open(f"runs/{name}.log", "w"), stderr=subprocess.STDOUT)
    peak = 0; t0 = time.time(); killed = False
    while ch.poll() is None:
        try:
            kb = [int(l.split()[1]) for l in open(f"/proc/{ch.pid}/status") if l.startswith("VmRSS")]
            # nice/uv wrappers: sum over the process and its direct children
            tot = sum(int(l.split()[1]) for pid in [ch.pid] + [int(x) for x in open(f"/proc/{ch.pid}/task/{ch.pid}/children").read().split()]
                      for l in open(f"/proc/{pid}/status") if l.startswith("VmRSS"))
        except (FileNotFoundError, ProcessLookupError):
            tot = 0
        try:
            hwm = max(int(l.split()[1]) for l in open(f"/proc/{ch.pid}/status") if l.startswith("VmHWM")) / 1e6
        except (FileNotFoundError, ProcessLookupError):
            hwm = 0
        peak = max(peak, tot / 1e6, hwm)
        if tot / 1e6 > 20:
            killed = True; os.kill(ch.pid, signal.SIGKILL); break
        time.sleep(10)
    rc = ch.wait()
    rec[name] = dict(sha=sha, rc=rc, peak_rss_gb=round(peak, 2), wall_s=round(time.time() - t0), killed_over_20gb=killed)
    json.dump(rec, open("data/step11_rss.json", "w"), indent=1); print(name, rec[name], flush=True)
    if rc:
        sys.exit(f"BROKEN: {name} extraction failed/aborted (rc {rc}, killed={killed}); see runs/{name}.log")
