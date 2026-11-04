"""Base-checkpoint diagnosis (read-only): raw-file tensor comparison, no transformers. Output data/base_diagnosis_tensors.json"""
import glob, json, sys, torch, numpy as np
from safetensors import safe_open
H = "/mnt/e/ai-models/huggingface_cache/hub/"
S = {"base": H + "models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf",
     "ing1": H + "models--allenai--OLMo-2-0425-1B/snapshots/ada39260a92eb8cfc2252b9e937f790fd5b0dbc1",
     "ing2": H + "models--allenai--OLMo-2-0425-1B/snapshots/b51c5f1b7c83a96b9a80bdaf14a2077c672fbccd",
     "sft": H + "models--allenai--OLMo-2-0425-1B-SFT/snapshots/0d85a3d037876ce6ac7d4311d994400fc66ac27f"}
def load(d):
    out = {}
    fs = sorted(glob.glob(d + "/*.safetensors"))
    if fs:
        for f in fs:
            with safe_open(f, "pt") as g:
                for k in g.keys(): out[k] = g.get_tensor(k)
    else:
        out = torch.load(glob.glob(d + "/*.bin")[0], weights_only=True, map_location="cpu")
    return out
W = {k: load(v) for k, v in S.items()}
res = {"keys": {k: len(v) for k, v in W.items()}}
for k, v in W.items():
    res.setdefault("dtypes", {})[k] = sorted({str(t.dtype) for t in v.values()})
for a, b in [("base", "ing1"), ("sft", "ing1"), ("ing1", "ing2"), ("base", "sft")]:
    A, B = W[a], W[b]
    res[f"keydiff_{a}_{b}"] = dict(only_a=sorted(set(A) - set(B))[:20], only_b=sorted(set(B) - set(A))[:20],
        shape_mismatch=[k for k in set(A) & set(B) if A[k].shape != B[k].shape][:20])
    rel = {}
    for k in sorted(set(A) & set(B)):
        if A[k].shape == B[k].shape:
            x, y = A[k].float(), B[k].float()
            rel[k] = float((x - y).norm() / y.norm())
    v = np.array(list(rel.values()))
    top = sorted(rel.items(), key=lambda kv: -kv[1])[:10]
    res[f"rel_{a}_{b}"] = dict(n=len(rel), median=float(np.median(v)), min=float(v.min()), max=float(v.max()), top10=top,
                               by_type={t: float(np.median([x for k, x in rel.items() if t in k])) for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "norm", "embed_tokens", "lm_head")})
    print(a, b, res[f"rel_{a}_{b}"]["median"], res[f"rel_{a}_{b}"]["max"], flush=True)
json.dump(res, open("data/base_diagnosis_tensors.json", "w"), indent=1)
