"""Third queue (AMENDMENT 2): re-train SFT parents s0-2 (tag sftre, weights kept), extract them, per-seed RL entry gate, then GRPO + extraction per passing seed.
One job at a time; every job under the rule-1 cap; GRPO waits while available RAM < 4 GB. A GRPO run that fails or exceeds 4 h is logged and the next seed continues."""
import json, pathlib, subprocess, sys, time
CAP = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=16G", "-p", "MemorySwapMax=0", "--", "env", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "nice", "-n", "10", "uv", "run", "python"]
LOG = "toy/runs/queue3.log"
def say(*a):
    with open(LOG, "a") as f: print(time.strftime("%H:%M:%S"), *a, file=f, flush=True)
def job(script, *a, log):
    with open(log, "a") as f: r = subprocess.run(CAP + [script, *map(str, a)], stdout=f, stderr=subprocess.STDOUT)
    say(script, a, "rc", r.returncode); return r.returncode
def avail_gb(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1e6
for s in (0, 1, 2):
    if job("toy/scripts/t2_train.py", "sft", s, "sftre", log=f"toy/runs/sftre_s{s}.log"): say("sftre train failed"); sys.exit(1)
    if job("toy/scripts/t3_extract.py", "sftre", s, log=f"toy/runs/sftre_s{s}.log"): say("sftre extract failed"); sys.exit(1)
for s in (0, 1, 2):
    if job("toy/scripts/t5_grpo.py", "gate_sft", s, log="toy/runs/rl_gate_sft.log"): say("gate_sft failed"); sys.exit(1)
g = json.load(open("toy/data/rl_gate_sft.json")); say("gate", json.dumps(g))
for s in (0, 1, 2):
    if not g[str(s)]["run_arm"]: say(f"seed {s}: entry gate not passed, GRPO not run"); continue
    while avail_gb() < 4: time.sleep(60)
    if job("toy/scripts/t5_grpo.py", "train_sft", s, log=f"toy/runs/rlsft_s{s}.log"): say(f"seed {s}: GRPO failed or stopped (4 h budget / error)"); continue
    if job("toy/scripts/t3_extract.py", "rlsft", s, log=f"toy/runs/rlsft_s{s}.log"): say(f"seed {s}: extract failed")
say("QUEUE3 DONE")
