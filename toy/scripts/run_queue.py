"""Sequential job queue for the toy run: SFT s0-2 (extract after each), SFT s3-4 iff all three finished < 3 h, then DPO s0-2.
One job at a time, each under the memory cap. Usage: uv run python toy/scripts/queue.py"""
import json, subprocess, sys, time, pathlib
CAP = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=16G", "-p", "MemorySwapMax=0", "--", "env", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "nice", "-n", "10", "uv", "run", "python"]
def job(script, *a, log):
    with open(log, "a") as f:
        r = subprocess.run(CAP + [script, *map(str, a)], stdout=f, stderr=subprocess.STDOUT)
    print(time.strftime("%H:%M:%S"), script, a, "rc", r.returncode, flush=True)
    if r.returncode: sys.exit(f"job failed: {script} {a} (see {log})")
def run(stage, seed):
    if pathlib.Path(f"toy/runs/{stage}_s{seed}/acts.npz").exists(): return
    job("toy/scripts/t2_train.py", stage, seed, log=f"toy/runs/{stage}_s{seed}.log")
    job("toy/scripts/t3_extract.py", stage, seed, log=f"toy/runs/{stage}_s{seed}.log")
for s in (0, 1, 2): run("sft", s)
walls = [json.load(open(f"toy/runs/sft_s{s}.json"))["wall_s"] for s in (0, 1, 2)]
if max(walls) < 3 * 3600:
    for s in (3, 4): run("sft", s)
else: print("SFT seeds 3,4 skipped: wall", walls, flush=True)
for s in (0, 1, 2): run("dpo", s)
print("QUEUE DONE", flush=True)
