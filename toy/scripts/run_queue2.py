"""Second queue: waits for QUEUE DONE in toy/runs/queue.log, then RL pass-rate gate (AMENDMENT 1) and GRPO s0 (+ extraction); s1, s2 only if s0 finished within 4 h."""
import json, pathlib, subprocess, sys, time
sys.path.insert(0, "toy/scripts")
CAP = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=16G", "-p", "MemorySwapMax=0", "--", "env", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "nice", "-n", "10", "uv", "run", "python"]
while "QUEUE DONE" not in open("toy/runs/queue.log").read():
    if "job failed" in open("toy/runs/queue.log").read(): sys.exit("queue 1 failed; not starting")
    time.sleep(60)
def job(script, *a, log):
    with open(log, "a") as f: r = subprocess.run(CAP + [script, *map(str, a)], stdout=f, stderr=subprocess.STDOUT)
    print(time.strftime("%H:%M:%S"), script, a, "rc", r.returncode, flush=True); return r.returncode
if job("toy/scripts/t5_grpo.py", "gate", log="toy/runs/rl_gate.log"): sys.exit("rl gate failed")
g = json.load(open("toy/data/rl_gate.json")); print(g, flush=True)
if not g["run_arm"]: sys.exit("RL arm not run: base pass rate outside [5%,60%]")
for s in (0, 1, 2):
    if job("toy/scripts/t5_grpo.py", "train", s, log=f"toy/runs/grpo_s{s}.log"): sys.exit(f"GRPO s{s} failed or stopped (4 h budget / error); see log")
    if job("toy/scripts/t3_extract.py", "grpo", s, log=f"toy/runs/grpo_s{s}.log"): sys.exit("extract failed")
print("QUEUE2 DONE", flush=True)
