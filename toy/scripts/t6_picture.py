"""t6: one-page picture. Usage: uv run python toy/scripts/t6_picture.py <analysis.json> <out_stem>
Rows = stage x set. Columns: displacement vs seed band (stage mean as bar, seed pairs as dots), kernel-ridge smoothness R2 (stage bar, seed-pair dots),
enrichment E with 95% CI (E=1 is the seed band). Bottom: entity-minus-control Delta per stage with CI, and calibration detection rate vs planted share."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1])); stem = sys.argv[2]
SETS = [("geo", "geography (entity)"), ("fig", "figures (entity)"), ("nouns", "C1 nouns (control)"), ("nonce", "C2 nonce (control)")]
COL = {"geo": "#1b6ca8", "fig": "#1b6ca8", "nouns": "#8a8a8a", "nonce": "#8a8a8a"}
stages = [s for s in ("sft", "dpo", "grpo") if s in d["stages"]]; rows = [(st, s, lab) for st in stages for s, lab in SETS]
fig = plt.figure(figsize=(11, 15))
gs = fig.add_gridspec(len(rows) + 6, 3, hspace=0.9, wspace=0.35)
fig.suptitle("Toy seed-replicate null, SmolLM2-360M, l*=16: stage displacement vs the seed band\n"
             f"(k seeds per stage: {', '.join(f'{s}={d['stages'][s]['k']}' for s in stages)}; T3, one 360M model; no 'no difference' claim from <=5 seeds)", fontsize=11)
titles = ["||D||/||H|| (bar: seed-mean stage, dots: seed pairs)", "KRR held-out R2 of D on H_b", "enrichment E = f_low(stage) / median f_low(seed pairs)"]
for ri, (st, s, lab) in enumerate(rows):
    p = d["stages"][st]["per_set"][s]
    ax = fig.add_subplot(gs[ri, 0]); ax.barh([0], [p["norm_D_stage"]], color=COL[s], height=0.5); ax.plot(p["norm_D_seed_pairs"], [0] * len(p["norm_D_seed_pairs"]), "o", color="k", ms=4)
    ax.set_yticks([0]); ax.set_yticklabels([f"{st.upper()} {lab}"], fontsize=7); ax.tick_params(axis="x", labelsize=7); ax.locator_params(axis="x", nbins=3)
    ax = fig.add_subplot(gs[ri, 1]); ax.barh([0], [p["krr_r2_stage"]], color=COL[s], height=0.5); ax.plot(p["krr_r2_seed_pairs"], [0] * len(p["krr_r2_seed_pairs"]), "o", color="k", ms=4); ax.set_yticks([]); ax.tick_params(axis="x", labelsize=7); ax.locator_params(axis="x", nbins=3)
    ax = fig.add_subplot(gs[ri, 2]); lo, hi = p["E_ci"]; ax.errorbar([p["E"]], [0], xerr=[[max(p["E"] - lo, 0)], [max(hi - p["E"], 0)]], fmt="o", color=COL[s], capsize=3); ax.axvline(1, color="r", lw=0.8, ls="--"); ax.set_yticks([]); ax.tick_params(axis="x", labelsize=7); ax.locator_params(axis="x", nbins=3)
    if ri == 0:
        for c, t in enumerate(titles): fig.axes[-3 + c].set_title(t, fontsize=7)
b = len(rows)
ax = fig.add_subplot(gs[b + 1:b + 3, 0:2]); xs = np.arange(len(stages))
for i, st in enumerate(stages):
    dl = d["stages"][st]["delta_geo_minus_nouns"]; ax.errorbar([i], [dl["point"]], yerr=[[max(dl["point"] - dl["ci"][0], 0)], [max(dl["ci"][1] - dl["point"], 0)]], fmt="s", color="#1b6ca8", capsize=4)
ax.axhline(0, color="r", lw=0.8, ls="--"); ax.set_xticks(xs); ax.set_xticklabels([s.upper() for s in stages]); ax.set_ylabel("Delta = E(geo) - E(C1)", fontsize=8); ax.set_title("entity-minus-control enrichment, 95% CI (paired, Zipf-matched)", fontsize=8)
ax = fig.add_subplot(gs[b + 1:b + 3, 2])
if "calibration" in d:
    sh = sorted(d["calibration"], key=float); ax.plot([float(x) for x in sh], [d["calibration"][x]["detection_rate"] for x in sh], "o-", color="#1b6ca8")
    ax.axhline(0.8, color="k", lw=0.6, ls=":"); ax.set_xlabel("planted rewiring share s", fontsize=8); ax.set_ylabel("detection rate", fontsize=8); ax.set_title("planted-rewiring calibration", fontsize=8); ax.set_ylim(-0.02, 1.02)
t = []
if "F_T2" in d: t.append(f"F-T2 {'FIRES' if d['F_T2']['fires'] else 'not fired'}: median ||D_stage||/||H|| {d['F_T2']['median_norm_D_stage']:.4f} vs ||D_seed||/||H|| {d['F_T2']['median_norm_D_seed']:.4f}")
if "F_T3" in d: t.append(f"F-T3 {'FIRES' if d['F_T3']['fires'] else 'not fired'}: power(s=0.10) {d['F_T3']['power_at_0_10']:.2f}, size(s=0) {d['F_T3']['size_at_0']:.2f}")
fig.text(0.02, 0.012, "\n".join(t), fontsize=8, va="bottom")
fig.savefig(stem + ".png", dpi=130, bbox_inches="tight"); fig.savefig(stem + ".pdf", bbox_inches="tight"); print("wrote", stem)
