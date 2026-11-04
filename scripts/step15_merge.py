"""Merge data/step15_part{0,1,2}.json -> data/step15_synthetic_null.json and print the rate tables."""
import json
o = {"cells": {}}
for g in range(3):
    d = json.load(open(f"data/step15_part{g}.json")); o["design"] = d["design"]; o["cells"].update(d["cells"]); o.update({k: v for k, v in d.items() if k == "real_rlzero"})
o["design"]["readouts"] = {"raw": "(i) colour vs N4' P=499", "resid": "(ii) cross-fit KRR residual vs N4' P=499", "iii_old": "row-permuted D, fixed smoother (bias measurement)", "iii": "Freedman-Lane (Winkler 2014): D*=fhat(H_b)+E_perm, full refit with LOO lambda, P=199"}
o["design"]["note"] += "; KRR lambda per fold by closed-form LOO (step 14 used inner 5-fold CV)"
json.dump(o, open("data/step15_synthetic_null.json", "w"), indent=1)
for k, c in o["cells"].items():
    print(k, {r: (c["summary"][r]["red"], c["summary"][r]["blue"]) for r in ("raw", "resid", "iii_old", "iii")})
for sn, v in o.get("real_rlzero", {}).items(): print("real", sn, {r: (v[r]["verdict"], round(v[r]["pct"], 3), round(v[r]["enrich"], 2)) for r in ("raw", "resid", "iii_old", "iii")})
