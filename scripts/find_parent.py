"""Candidate-parent table: median relative weight distance ||A-B||/||B|| vs SFT and sampled cosines (read-only). -> data/parent_search.json"""
import glob, json, numpy as np, torch
from safetensors import safe_open
H = "/mnt/e/ai-models/huggingface_cache/hub/"
R = H + "models--allenai--OLMo-2-0425-1B/snapshots/"
C = {"main_a1847dff": R + "a1847dff35000b4271fa70afc5db10fd29fedbdf", "stage1_final_9d3e4365": R + "9d3e43659f00c17e6da23cf32333afd1fc39fa1a",
     "ing1_ada39260": R + "ada39260a92eb8cfc2252b9e937f790fd5b0dbc1", "ing2_b51c5f1b": R + "b51c5f1b7c83a96b9a80bdaf14a2077c672fbccd",
     "ing3_f83cd641": R + "f83cd641d4f00954eb25c781e0ac360debb3239e"}
SFT = H + "models--allenai--OLMo-2-0425-1B-SFT/snapshots/0d85a3d037876ce6ac7d4311d994400fc66ac27f/pytorch_model.bin"
sft = {k: v.float() for k, v in torch.load(SFT, weights_only=True, map_location="cpu").items()}
keys = sorted(sft); samp = ["model.embed_tokens.weight", "model.layers.5.mlp.up_proj.weight", "model.layers.5.self_attn.q_proj.weight", "lm_head.weight"]
def load(d):
    o = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(f, "pt") as g:
            for k in g.keys(): o[k] = g.get_tensor(k).float()
    return o
def row(W):
    rel = np.array([float((W[k] - sft[k]).norm() / sft[k].norm()) for k in keys])
    cos = {k.split("model.")[-1]: float(torch.nn.functional.cosine_similarity(W[k].flatten(), sft[k].flatten(), dim=0)) for k in samp}
    return dict(median_rel=float(np.median(rel)), min_rel=float(rel.min()), max_rel=float(rel.max()), cos=cos)
out = {}; ing = {}
for n, d in C.items():
    W = load(d); out[n] = row(W); print(n, out[n]["median_rel"], list(out[n]["cos"].values()), flush=True)
    if n.startswith("ing"): ing[n] = W
    del W
soup = {k: sum(ing[n][k] for n in ing) / 3 for k in keys}; out["uniform_soup_ing123"] = row(soup); print("soup", out["uniform_soup_ing123"]["median_rel"], flush=True)
W3 = ing["ing3_f83cd641"]; mainW = load(C["main_a1847dff"])
out["main_vs_ing3_median_rel"] = float(np.median([float((mainW[k] - W3[k]).norm() / W3[k].norm()) for k in keys]))
out["main_vs_soup_median_rel"] = float(np.median([float((mainW[k] - soup[k]).norm() / soup[k].norm()) for k in keys]))
print(out["main_vs_ing3_median_rel"], out["main_vs_soup_median_rel"])
json.dump(out, open("data/parent_search.json", "w"), indent=1)
