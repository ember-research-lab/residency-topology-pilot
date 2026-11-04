"""Sequential driver: fetch + closure + extract each checkpoint of a group (one job at a time), keep snapshots.
Usage: uv run python scripts/run_checkpoints.py step6|step7|step8   (appends to data/checkpoints.json)"""
import json, pathlib, subprocess, sys, time

import tomllib
BASE_SHA = tomllib.load(open("configs/pilot.toml", "rb"))["hosts"]["olmo2_1b"]["base"]["sha"]   # full SHA of the gated parent
BASE_SNAP = f"/mnt/e/ai-models/huggingface_cache/hub/models--allenai--OLMo-2-0425-1B/snapshots/{BASE_SHA}"
BASE_SPEC = tomllib.load(open("configs/pilot.toml", "rb"))["hosts"]["olmo2_1b"]["base"]["spec"]
R = "allenai/OLMo-2-0425-1B"
GROUPS = {
 "step6": [("sft", R + "-SFT", "main", "0d85a3d03787"), ("dpo", R + "-DPO", "main", "c4b0485961ab"),
           ("rlvr1", R + "-RLVR1", "main", "12cb33498d26"), ("instruct", R + "-Instruct", "main", "48d788eca847")],
 "step7": [(f"ing{i}", R, f"stage2-ingredient{i}-step23852-tokens51B", None) for i in (1, 2, 3)],
 "step8": [(f"inst_step{s}", R + "-Instruct", f"step_{s}", None) for s in (200, 800, 1600, 2600)],
}
reg = pathlib.Path("data/checkpoints.json")
db = json.loads(reg.read_text()) if reg.exists() else {}
env = dict(__import__("os").environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
for name, repo, rev, prefix in GROUPS[sys.argv[1]]:
    if name in db and pathlib.Path(f"runs/{name}/acts.npz").exists():
        continue
    t0 = time.time()
    out = subprocess.run(["nice", "-n", "10", sys.executable, "scripts/fetch_checkpoint.py", repo, rev], capture_output=True, text=True, env=env)
    line = [l for l in out.stdout.splitlines() if l.startswith(repo)][-1]
    _, _, sha, path = line.split(" ", 3)
    g = subprocess.run(["nice", "-n", "10", sys.executable, "scripts/lineage_check.py", BASE_SPEC, f"{repo}@{rev}"], capture_output=True, text=True, env=env)
    print(g.stdout.strip(), flush=True)
    if g.returncode:
        sys.exit(f"BROKEN: lineage gate FAIL for {name}; stop")
    note = "" if prefix is None or sha.startswith(prefix) else f"MISMATCH expected prefix {prefix}"
    t_fetch = time.time() - t0
    r = subprocess.run(["nice", "-n", "10", sys.executable, "scripts/step3_extract.py", path, sha, f"runs/{name}", "bf16", "--base-snap", BASE_SNAP, "--gate", f"{BASE_SHA}:{sha}"], env=env,
                       stdout=open(f"runs/{name}.log", "w"), stderr=subprocess.STDOUT)
    db[name] = dict(repo=repo, revision=rev, sha=sha, snapshot=path, sha_note=note, fetch_s=round(t_fetch), total_s=round(time.time() - t0), rc=r.returncode)
    reg.write_text(json.dumps(db, indent=1))
    print(name, db[name], flush=True)
    if r.returncode:
        sys.exit(f"BROKEN: {name} extraction/closure failed (rc {r.returncode}); see runs/{name}.log")
    if name == "sft":   # precision-null condition: ||D_prec|| < 0.1 ||D_SFT||
        import numpy as np
        a, b = np.load("runs/olmo2-1b-base/acts.npz"), np.load("runs/sft/acts.npz")
        Hb, Hs = (x["last_L12"][:4000].reshape(500, 8, -1).mean(1) for x in (a, b))
        dsft = np.linalg.norm((Hs - Hs.mean(0)) - (Hb - Hb.mean(0)))
        dprec = json.load(open("data/precision_null_step5.json"))["last_L12"]["norm_D_prec"]
        print(f"||D_SFT||={dsft:.3f} (need > 16.0)  ||D_prec||/||D_SFT||={dprec/dsft:.4f} (need < 0.1)", flush=True)
        db["precision_ratio_sft"] = dict(D_SFT=float(dsft), D_prec=dprec, ratio=float(dprec / dsft)); reg.write_text(json.dumps(db, indent=1))
        if dprec / dsft >= 0.1:
            sys.exit("BROKEN: ||D_prec|| >= 0.1 ||D_SFT||; stop and report")
