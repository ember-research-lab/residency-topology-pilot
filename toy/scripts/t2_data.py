"""t2: build the fixed training slices (PREREG-toy sec.1): SFT (500k packed tokens, tulu-3-sft-mixture shard 0), DPO (1000 pairs, tulu-3-8b
preference mixture shard 0), arithmetic (2000 train + 200 held-out). Tokenizer only; no model. -> toy/data/{sft_slice,dpo_slice}.npz, arith.json, slices.json
Usage: uv run python toy/scripts/t2_data.py"""
import hashlib, json, os, pathlib
import numpy as np, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
CACHE = "/mnt/e/ai-models/huggingface_cache/hub"
snap = json.load(open("toy/data/bases.json"))["HuggingFaceTB/SmolLM2-360M"]["snapshot"]
tok = Tokenizer.from_file(f"{snap}/tokenizer.json")
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
IM_S, IM_E = enc("<|im_start|>"), enc("<|im_end|>")
assert len(IM_S) == 1 and len(IM_E) == 1, (IM_S, IM_E)
ascii_ok = lambda s: sum(c.isascii() for c in s) >= 0.95 * max(len(s), 1)
tokn = os.environ["HF_API_KEY"]
out = {}
def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def turn(role, content, train):
    h = IM_S + enc(f"{role}\n"); b = enc(content) + IM_E + enc("\n")
    return h + b, ([-100] * len(h) + (b if train else [-100] * len(b)))
# ---- SFT
REV_SFT = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"
p = hf_hub_download("allenai/tulu-3-sft-mixture", "data/train-00000-of-00006.parquet", repo_type="dataset", revision=REV_SFT, cache_dir=CACHE, token=tokn)
t = pq.read_table(p, columns=["messages"]).column("messages").to_pylist()
order = np.random.default_rng(20261104).permutation(len(t))
ids, lab, n_ex = [], [], 0
for i in order:
    msgs = t[i]
    if not all(ascii_ok(m["content"]) for m in msgs) or not msgs or msgs[-1]["role"] != "assistant": continue
    x, y = [], []
    for m in msgs:
        a, b = turn(m["role"], m["content"], m["role"] == "assistant"); x += a; y += b
    x, y = x[:512], y[:512]
    if all(v == -100 for v in y): continue
    ids += x; lab += y; n_ex += 1
    if len(ids) >= 500_000: break
N = (len(ids) // 512) * 512
ids, lab = np.array(ids[:N], np.int32), np.array(lab[:N], np.int32)
np.savez("toy/data/sft_slice.npz", ids=ids.reshape(-1, 512), labels=lab.reshape(-1, 512))
out["sft"] = dict(dataset="allenai/tulu-3-sft-mixture", revision=REV_SFT, shard="data/train-00000-of-00006.parquet", perm_seed=20261104, examples=n_ex,
                  tokens=int(N), sequences=int(N // 512), loss_tokens=int((lab != -100).sum()), sha256_token_ids=sha(ids), sha256_labels=sha(lab))
print("sft", out["sft"], flush=True)
# ---- DPO
REV_DPO = "78a6f00785946cd24276c5dd075f83a143a3b1e6"
p = hf_hub_download("allenai/llama-3.1-tulu-3-8b-preference-mixture", "data/train-00000-of-00004.parquet", repo_type="dataset", revision=REV_DPO, cache_dir=CACHE, token=tokn)
tb = pq.read_table(p, columns=["prompt", "chosen", "rejected"]).to_pylist()
order = np.random.default_rng(20261104).permutation(len(tb))
rows = []
for i in order:
    r = tb[i]
    c, j = r["chosen"][-1], r["rejected"][-1]
    if c["role"] != "assistant" or j["role"] != "assistant" or not all(ascii_ok(s) for s in (r["prompt"], c["content"], j["content"])): continue
    pr = IM_S + enc("user\n" + r["prompt"]) + IM_E + enc("\n") + IM_S + enc("assistant\n")
    ch, rj = enc(c["content"]) + IM_E, enc(j["content"]) + IM_E
    if len(pr) + len(ch) > 256 or len(pr) + len(rj) > 256 or ch == rj: continue
    rows.append((pr, ch, rj))
    if len(rows) == 1000: break
assert len(rows) == 1000
pad = lambda xs, n: np.array([x + [0] * (n - len(x)) for x in xs], np.int32)
P, C, R = pad([r[0] for r in rows], 256), pad([r[1] for r in rows], 256), pad([r[2] for r in rows], 256)
np.savez("toy/data/dpo_slice.npz", prompt=P, chosen=C, rejected=R, plen=np.array([len(r[0]) for r in rows]), clen=np.array([len(r[1]) for r in rows]), rlen=np.array([len(r[2]) for r in rows]))
tot = sum(len(r[0]) + len(r[1]) for r in rows) + sum(len(r[0]) + len(r[2]) for r in rows)
out["dpo"] = dict(dataset="allenai/llama-3.1-tulu-3-8b-preference-mixture", revision=REV_DPO, shard="data/train-00000-of-00004.parquet", perm_seed=20261104, pairs=1000,
                  tokens_chosen_plus_rejected=int(tot), sha256=sha(np.concatenate([P.ravel(), C.ravel(), R.ravel()])))
print("dpo", out["dpo"], flush=True)
# ---- arithmetic
def gen(seed, n, avoid=()):
    rng = np.random.default_rng(seed); seen = set(avoid); items = []
    while len(items) < n:
        a, b, op = int(rng.integers(10, 1000)), int(rng.integers(10, 1000)), ["+", "-", "x"][int(rng.integers(0, 3))]
        if (a, op, b) in seen: continue
        seen.add((a, op, b)); items.append(dict(prompt=f"{a} {op} {b} =", answer=a + b if op == "+" else a - b if op == "-" else a * b))
    return items
tr = gen(12345, 2000); ho = gen(54321, 200, {(int(x["prompt"].split()[0]), x["prompt"].split()[1], int(x["prompt"].split()[2])) for x in tr})
json.dump(dict(train=tr, heldout=ho), open("toy/data/arith.json", "w"))
out["arith"] = dict(train=2000, heldout=200, gen_seeds=[12345, 54321], sha256=hashlib.sha256(json.dumps(dict(train=tr, heldout=ho)).encode()).hexdigest())
json.dump(out, open("toy/data/slices.json", "w"), indent=1); print(out["arith"])
