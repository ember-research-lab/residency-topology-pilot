"""t2: one full fine-tune run from the base (PREREG-toy sec.1). Usage (under the cap): uv run python toy/scripts/t2_train.py <sft|dpo> <seed>
Weights -> /mnt/e/toy-residency/<stage>_s<seed>/ (fp32 safetensors). Log -> stdout (redirect to toy/runs/<stage>_s<seed>.log). Aborts on non-finite
loss or when the projected wall time exceeds 2x plan (prereg throughput rule)."""
import json, math, os, pathlib, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "toy")
import toylib as T
from rtp import cap
cap.require_cap()
stage, seed = sys.argv[1], int(sys.argv[2]); tag = sys.argv[3] if len(sys.argv) > 3 else stage   # tag: output name (AMENDMENT 2 re-train of SFT parents = "sftre")
snap = json.load(open("toy/data/bases.json"))["HuggingFaceTB/SmolLM2-360M"]["snapshot"]
PLAN_S = 3 * 3600
torch.manual_seed(seed); rng = np.random.default_rng(seed)
model = T.load(snap); model.train(); model.config.use_cache = False
params = [p for p in model.parameters() if p.requires_grad]
assert all(p.dtype == torch.float32 for p in params)
t0 = time.time()
def vmhwm(): return int([l for l in open("/proc/self/status") if l.startswith("VmHWM")][0].split()[1])
def lr_at(step, total, peak, warm, cosine):
    if step < warm: return peak * (step + 1) / warm
    if not cosine: return peak
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1))))
def check(loss, step, total):
    if not math.isfinite(loss): sys.exit("F-T4: non-finite loss")
    el = time.time() - t0
    if step == 3 and el / (step + 1) * total > 2 * PLAN_S: sys.exit(f"THROUGHPUT RULE: projected {el / (step + 1) * total:.0f}s > 2x plan {PLAN_S}s; amend prereg")
opt = torch.optim.AdamW(params, lr=1e-6, betas=(0.9, 0.95), weight_decay=0.0)
if stage == "sft":
    d = np.load("toy/data/sft_slice.npz"); ids, lab = torch.from_numpy(d["ids"]).long(), torch.from_numpy(d["labels"]).long()
    perm = rng.permutation(len(ids)); B, MB = 8, 2; total = len(perm) // B
    for step in range(total):
        for g in opt.param_groups: g["lr"] = lr_at(step, total, 2e-5, 5, True)
        idx = perm[step * B:(step + 1) * B]; ntok = (lab[idx][:, 1:] != -100).sum().item(); tot = 0.0
        for k in range(0, B, MB):
            x, y = ids[idx[k:k + MB]], lab[idx[k:k + MB]]
            lg = model(input_ids=x).logits[:, :-1]
            loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y[:, 1:].reshape(-1), ignore_index=-100, reduction="sum") / ntok
            loss.backward(); tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item(); opt.step(); opt.zero_grad(set_to_none=True)
        check(tot, step, total)
        print(f"step {step+1}/{total} loss {tot:.4f} gnorm {gn:.3f} lr {opt.param_groups[0]['lr']:.2e} t {time.time()-t0:.0f}s", flush=True)
elif stage == "dpo":
    d = np.load("toy/data/dpo_slice.npz"); P, C, R = d["prompt"], d["chosen"], d["rejected"]; pl, cl, rl = d["plen"], d["clen"], d["rlen"]
    def seqs(i):  # (ids, response mask) for chosen and rejected
        out = []
        for resp, rln in ((C[i], cl[i]), (R[i], rl[i])):
            x = np.concatenate([P[i][:pl[i]], resp[:rln]]); m = np.zeros(len(x), bool); m[pl[i]:] = True; out.append((x, m))
        return out
    def logps(batch_idx, grad):
        S = [s for i in batch_idx for s in seqs(i)]; T_ = max(len(x) for x, _ in S)
        X = torch.zeros(len(S), T_, dtype=torch.long); Mk = torch.zeros(len(S), T_, dtype=torch.bool); A = torch.zeros(len(S), T_, dtype=torch.long)
        for j, (x, m) in enumerate(S): X[j, :len(x)] = torch.from_numpy(x); Mk[j, :len(x)] = torch.from_numpy(m); A[j, :len(x)] = 1
        with torch.set_grad_enabled(grad):
            lg = model(input_ids=X, attention_mask=A).logits[:, :-1].float()
            lp = torch.log_softmax(lg, -1).gather(-1, X[:, 1:, None])[..., 0] * Mk[:, 1:]
        return lp.sum(1).view(len(batch_idx), 2)
    ref_f = pathlib.Path("toy/data/dpo_ref_logps.npy")
    if not ref_f.exists():   # reference = the base, computed once, identical for all seeds
        model.eval(); order = np.argsort(pl + np.maximum(cl, rl)); ref = np.zeros((len(P), 2), np.float32)
        for k in range(0, len(order), 8): ref[order[k:k + 8]] = logps(order[k:k + 8], False).numpy()
        np.save(ref_f, ref); model.train(); print("ref logps done", time.time() - t0, flush=True)
    ref = torch.from_numpy(np.load(ref_f)); beta = 0.1
    perm = rng.permutation(len(P)); B, MB = 16, 2; total = len(perm) // B
    for step in range(total):
        for g in opt.param_groups: g["lr"] = lr_at(step, total, 1e-6, 3, False)
        idx = perm[step * B:(step + 1) * B]; tot = 0.0; acc = 0.0
        for k in range(0, B, MB):
            b = idx[k:k + MB]; lp = logps(b, True)
            marg = beta * ((lp[:, 0] - lp[:, 1]) - (ref[b, 0] - ref[b, 1]))
            loss = -F.logsigmoid(marg).sum() / B; loss.backward(); tot += loss.item(); acc += (marg > 0).float().sum().item() / B
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item(); opt.step(); opt.zero_grad(set_to_none=True)
        check(tot, step, total)
        print(f"step {step+1}/{total} loss {tot:.4f} acc {acc:.2f} gnorm {gn:.3f} t {time.time()-t0:.0f}s", flush=True)
else:
    sys.exit("stage?")
od = pathlib.Path(f"/mnt/e/toy-residency/{tag}_s{seed}"); od.mkdir(parents=True, exist_ok=True)
model.save_pretrained(od)
w = time.time() - t0
json.dump(dict(stage=stage, seed=seed, wall_s=round(w), vmhwm_kb=vmhwm(), steps=total), open(f"toy/runs/{tag}_s{seed}.json", "w"))
print(f"SAVED {od} wall {w:.0f}s VmHWM {vmhwm()} kB", flush=True)
