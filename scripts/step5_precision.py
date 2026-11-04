"""Step 5: precision null D_prec = H(base f32) - H(base bf16-rounded), geography, template-averaged, centred.
Reads runs/olmo2-1b-base{,-f32}/acts.npz; writes data/precision_null_step5.json (norm_D_prec, norm_H, rel per key)."""
import json, numpy as np
a, b = np.load("runs/olmo2-1b-base/acts.npz"), np.load("runs/olmo2-1b-base-f32/acts.npz")
out = {}
for k in a.files:
    if k == "meta": continue
    Hb, Hf = (x[k][:4000].reshape(500, 8, -1).mean(1) for x in (a, b))
    Hb, Hf = Hb - Hb.mean(0), Hf - Hf.mean(0)
    out[k] = dict(norm_D_prec=float(np.linalg.norm(Hf - Hb)), norm_H=float(np.linalg.norm(Hb)), rel=float(np.linalg.norm(Hf - Hb) / np.linalg.norm(Hb)))
json.dump(out, open("data/precision_null_step5.json", "w"), indent=1)
print({k: round(v["norm_D_prec"], 3) for k, v in out.items()})
