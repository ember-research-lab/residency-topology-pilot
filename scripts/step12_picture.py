"""Step 12: the one-page picture (PREREG sec 4) from committed JSON. -> runs/picture-2026-11-04.png and .pdf
Rows: SFT, DPO, RLVR1, RLVR-final(Instruct), RL-Zero, then EXPLORATORY rows (AMENDMENT 3 increments and Instruct step increments)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

s9, s10, s11, inc = (json.load(open(f)) for f in ("data/step9_results.json", "data/step10_results.json", "data/step11_results.json", "data/increments_results.json"))
eps_dO = s10["epsilon"]["d_O"]["eps"]
rows = []   # (label, stats dict, verdict text, group)
for k, lab in (("sft", "SFT"), ("dpo", "DPO"), ("rlvr1", "RLVR1 (secondary)"), ("instruct", "RLVR-final (Instruct)")):
    x = s9[k]; b = s10["stages"][k]
    rows.append((lab, x, f"{x['verdict']} | stab {b['stability']:.2f}" + ("" if b["stable_claim"] else " UNSTABLE"), "pre"))
x = s11["point"]; rows.append(("RL-Zero-Math (7B)", x, f"{x['verdict']} | stab {s11['bootstrap']['stability']:.2f}", "pre"))
for k, lab in (("sft", "SFT | base"), ("dpo|sft", "DPO | SFT"), ("rlvr1|dpo", "RLVR1 | DPO"), ("instruct|rlvr1", "Instruct | RLVR1")):
    x = inc["point"][k]; sb = inc["stability"][k]
    rows.append((lab + "  [EXPL.]", x, f"{x['verdict']} | stab {sb['stability']:.2f}" + ("" if sb["stable_claim"] else " UNSTABLE"), "exp"))
for k, lab in (("step200|rlvr1", "step200 | RLVR1"), ("step800|step200", "step800 | 200"), ("step1600|step800", "step1600 | 800"), ("step2600|step1600", "step2600 | 1600")):
    rows.append(("RL " + lab + "  [EXPL.]", inc["point"][k], "non-voting", "exp"))
n = len(rows); y = np.arange(n)[::-1]
fig, ax = plt.subplots(1, 6, figsize=(17, 8.2), sharey=True, gridspec_kw=dict(width_ratios=[1.3, 1, 1.3, 1.3, 1.3, 1.6]))
col = lambda g: "#1f4e79" if g == "pre" else "#9b9b9b"
for a in ax[:-1]:
    a.grid(axis="x", alpha=.25)
# 1 f_low percentile (1 - p_red), Holm line
a = ax[0]; a.set_title("f_low(R) percentile in N4'\n(red = high; line 0.95 raw alpha)", fontsize=9)
a.barh(y, [1 - r_[1]["p_red"] for r_ in rows], color=[col(r_[3]) for r_ in rows]); a.axvline(0.95, color="k", ls="--", lw=1); a.set_xlim(0, 1.02)
for yi, r_ in zip(y, rows):
    h = r_[1].get("holm_p_red")
    a.text(1.0, yi, "Holm-red" if (h is not None and h <= .05) else "", ha="right", va="center", fontsize=7, color="w")
a.set_yticks(y); a.set_yticklabels([r_[0] for r_ in rows], fontsize=8)
# 2 lambda_lin
a = ax[1]; a.set_title("lambda_lin\n(linear share of D)", fontsize=9); a.barh(y, [r_[1]["lambda_lin"] for r_ in rows], color=[col(r_[3]) for r_ in rows])
# 3 dPR_cov with N4' band
a = ax[2]; a.set_title("Delta PR_cov\n(N4' 5-95 band)", fontsize=9)
for yi, r_ in zip(y, rows):
    q = r_[1]["null_q"]["dPR_cov"]; a.plot([q[0], q[2]], [yi, yi], color="#c9c9c9", lw=8, solid_capstyle="butt"); a.plot(r_[1]["dPR_cov"], yi, "o", color=col(r_[3]))
# 4 dclust with band
a = ax[3]; a.set_title("Delta clustering\n(N4' 5-95 band)", fontsize=9)
for yi, r_ in zip(y, rows):
    q = r_[1]["null_q"]["dclust"]; a.plot([q[0], q[2]], [yi, yi], color="#c9c9c9", lw=8, solid_capstyle="butt"); a.plot(r_[1]["dclust"], yi, "o", color=col(r_[3]))
a.axvline(0, color="k", lw=.6)
# 5 d_O with eps band (ingredient drift)
a = ax[4]; a.set_title(f"d_O\n(ribbon: eps={eps_dO:.3f})", fontsize=9)
a.axvspan(0, eps_dO, color="#c9c9c9", alpha=.8); a.barh(y, [r_[1]["d_O"] for r_ in rows], color=[col(r_[3]) for r_ in rows]); a.set_xlim(0, 1)
# 6 t_hat_k bars (k=1..5)
a = ax[5]; a.set_title("t_hat_k, k = 1..5 (low modes)", fontsize=9)
for yi, r_ in zip(y, rows):
    t = np.array(r_[1]["t_hat"])[:5]
    for k_, tv in enumerate(t):
        a.barh(yi + (k_ - 2) * .13, tv, height=.12, color=plt.cm.viridis(k_ / 5))
    a.text(a.get_xlim()[1] if False else 1.02, yi, r_[2], transform=a.get_yaxis_transform(), va="center", fontsize=7)
a.axvline(0, color="k", lw=.6)
fig.suptitle("Imitation activates, reward prunes: pilot picture 2026-11-04 (blue = preregistered rows, grey = EXPLORATORY post-hoc increments, AMENDMENT 3)\n"
             "OLMo-2-0425-1B at l*=12 on base f83cd641 (lineage-gated); RL-Zero-Math = Olmo-3 7B at l*=16. Ribbon: d_O only has an ingredient-drift band; no layer-0/control ribbon exists for these columns.",
             fontsize=9)
plt.tight_layout(rect=(0, 0.03, 0.93, 0.93))
fig.text(0.01, 0.01, "steps 13-15: controls red; colour family killed as verdict instrument; see REPORT final adjudication", fontsize=9)
fig.savefig("runs/picture-2026-11-04.png", dpi=150); fig.savefig("runs/picture-2026-11-04.pdf")
print("saved")
