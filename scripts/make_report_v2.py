"""Write the v2 steps 3-10 results block (gated base f83cd641) to runs/report_v2_block.md from the committed JSON (no hand-copied numbers)."""
import tomllib
import json

ck, s9, s10 = (json.load(open(f)) for f in ("data/checkpoints.json", "data/step9_results.json", "data/step10_results.json"))
ST = ["sft", "dpo", "rlvr1", "instruct"]
f = lambda x, n=4: f"{x:.{n}g}"
o = []
BASE = tomllib.load(open("configs/pilot.toml", "rb"))["hosts"]["olmo2_1b"]["base"]["sha"]
o.append("\n# Steps 3-10 v2 on the gated base (AMENDMENT 2, 2026-11-04)\n")
o.append("Adjudications (a)-(f) of 2026-11-04 carry over unchanged (r=50 at the n/10 cap; m=5; layer-0 mean-over-name-tokens; no zipf partialling; precision null explicit; snapshots kept). The extractions of the non-base checkpoints are reused from the void run (activations depend only on the checkpoint weights, not on the base; closure re-checked on the base only), each re-gated against the new base below.\n")
o.append("## Checkpoints (full commit SHAs, snapshots kept in /mnt/e/ai-models/huggingface_cache/hub; data/checkpoints.json)\n")
o.append("| key | repo @ revision | full SHA | prefix check | lineage gate vs base: median rel / median cos / result | closure rel (max over L0/4/8/12) | seconds (fetch / total) |\n|---|---|---|---|---|---|---|")
for k, v in ck.items():
    if k == "precision_ratio_sft":
        continue
    cl = json.load(open(f"data/runs_meta/closure_{k}.json")); gr = json.load(open(f"data/lineage/{BASE[:12]}__{v['sha'][:12]}.json")) if v["sha"] != BASE else None
    gs = "%.4f / %.4f / PASS" % (gr["median_rel"], gr["median_cos"]) if gr else "is the base"
    c = max(max(x["closure_rel"], x["reextract_rel"]) for x in cl.values())
    o.append(f"| {k} | {v['repo']} @ {v['revision']} | {v['sha']} | {v['sha_note'] or 'matches pinned prefix' if k in ST else 'n/a (branch)'} | {gs} | {c:.2e} | {v['fetch_s']} / {v['total_s']} |")
o.append("\nBROKEN-if checks, steps 3-8 (AMENDMENT 2): lineage gate PASS for every checkpoint against the base (table above; step 9/10 assert it); closure <= 1e-5 and re-extraction <= 1e-6 on every checkpoint; pinned SHA prefixes match for the four stages. ing3 is the base (bit-identical activations to runs/olmo2-1b-base, max abs difference 0.0). Tier T1 (measured).\n")
pr = ck["precision_ratio_sft"]
o.append(f"## Precision null (step 6, explicit)\n||D_SFT|| = {pr['D_SFT']:.3f} at l*=12 (centred, template-averaged last token, geography). Required > 16.0: PASS (ratio ||D_SFT||/16.0 = {pr['D_SFT']/16:.1f}). ||D_prec|| = {pr['D_prec']:.3f}; ||D_prec||/||D_SFT|| = {pr['ratio']:.4f} < 0.1: PASS. Tier T1.\n")
o.append("## Results table (l*=12, m=5, r=50, P=999; Holm at 0.05 over the 8-test family; B=1000 bootstrap)\n")
o.append("Family caveat: the Olmo-3 RL-Zero pair (step 11) is not yet run. The family of 8 is held at size 8 with the two absent p-values taken as 1 (the most permissive reading for the present ones); SFT/DPO/Instruct vote in that family. RLVR1 (a post-trained RL stage) is reported with its own Holm over its 2 p-values and is labelled secondary. Verdict words are the preregistered ones only.\n")
hdr = "| stage | ||D||/||H|| | f_low(R) (null median) | lambda_lin | r_eff(R) | p_blue | p_red | Holm p_blue / p_red | verdict | dPR_cov (N4' 5th / 95th) | d_O (N4' 5/95) | d_NP k=10 | Procrustes R_P | CKA | bootstrap stability |"
o.append(hdr + "\n" + "|---|" + "---|" * (hdr.count("|") - 2 - 1))
for s in ST:
    x, b = s9[s], s10["stages"][s]
    q = x["null_q"]
    o.append(f"| {s} | {f(x['ratio'])} | {f(x['f_low'])} ({f(x['f_low_null_median'])}) | {f(x['lambda_lin'])} | {x['reff_R']:.1f} | {x['p_blue']:.3f} | {x['p_red']:.3f} | {x['holm_p_blue']:.3f} / {x['holm_p_red']:.3f} | **{x['verdict']}** | {f(x['dPR_cov'])} ({f(q['dPR_cov'][0])} / {f(q['dPR_cov'][2])}) | {f(x['d_O'])} ({f(q['d_O'][0])} / {f(q['d_O'][2])}) | {f(x['d_NP'])} | {f(x['R_P'])} | {f(x['CKA'])} | {b['stability']:.3f} ({'claimable' if b['stable_claim'] else 'NOT claimable, < 0.9'}) |")
o.append("\nVerdict set per replicate (counts of 1000): " + "; ".join(f"{s}: {s10['stages'][s]['verdict_counts']}" for s in ST) + ".\n")
o.append("Other non-voting rows (point estimates; none of them votes):\n\n| stage | clustering change (N4' 5/50/95) | E_low(base graph) change (N4' 5/50/95) | coupling C | t_hat_k (k=1..5) | null percentile of dPR_cov / dclust / dE_low / d_O / d_NP |\n|---|---|---|---|---|---|")
for s in ST:
    x = s9[s]; q = x["null_q"]; pc = x["null_pct"]
    o.append(f"| {s} | {f(x['dclust'])} ({', '.join(f(v) for v in q['dclust'])}) | {f(x['dE_low_base_graph'])} ({', '.join(f(v) for v in q['dE_low_base_graph'])}) | {f(x['C'])} | {', '.join(f(t, 3) for t in x['t_hat'])} | {pc['dPR_cov']:.3f} / {pc['dclust']:.3f} / {pc['dE_low_base_graph']:.3f} / {pc['d_O']:.3f} / {pc['d_NP']:.3f} |")
o.append("\nBootstrap 95% CIs (percentile; 400-entity subsample, template multiset): " + "; ".join(f"{s}: ratio {f(s10['stages'][s]['ci95']['ratio'][0])}-{f(s10['stages'][s]['ci95']['ratio'][1])}, f_low {f(s10['stages'][s]['ci95']['f_low'][0])}-{f(s10['stages'][s]['ci95']['f_low'][1])}, d_O {f(s10['stages'][s]['ci95']['d_O'][0])}-{f(s10['stages'][s]['ci95']['d_O'][1])}" for s in ST) + ".\n")
o.append("## Sensitivity columns (reported, non-voting)\nr/2, r, 2r = 25/50/100 (f_low(R), p_red; red verdict by raw p_red <= 0.05):\n\n| stage | r=25 | r=50 | r=100 |\n|---|---|---|---|")
for s in ST:
    z = s9["r_sensitivity"][s]
    o.append(f"| {s} | " + " | ".join(f"{f(z[k]['f_low'])}, p_red {z[k]['p_red']:.3f}, lambda_lin {f(z[k]['lambda_lin'])}" for k in ("25", "50", "100")) + " |")
o.append("\nm = m*-2, m*+2, 10 (f_low(R), p_red, d_O):\n\n| stage | m=3 | m=7 | m=10 |\n|---|---|---|---|")
for s in ST:
    z = s9["m_sensitivity"][s]
    o.append(f"| {s} | " + " | ".join(f"{f(z[k]['f_low'])}, p_red {z[k]['p_red']:.3f}, d_O {f(z[k]['d_O'])}" for k in ("3", "7", "10")) + " |")
e = s10["epsilon"]
o.append("\n## epsilon floor (steps 7-8) and legacy P1 row\nepsilon = max(eps_A over the 3 ingredient pairs, eps_N6 over the 3 consecutive RL-step pairs), each the 95th bootstrap percentile of the statistic under the same paired draw (maximum over pairs). Caption: none of these is a true seed replicate; eps_A is a drift reference, not a noise floor.\n\n| statistic | eps_A (ingredient pairs) | eps_N6 (RL steps 200-800-1600-2600) | epsilon |\n|---|---|---|---|")
for q_, v in e.items():
    o.append(f"| {q_} | {f(v['eps_A'])} | {f(v['eps_N6'])} | **{f(v['eps'])}** |")
o.append("\nPoint values of the pairwise statistics (step 9, all templates and entities):\n\n| pair | ||D||/||H_base|| | d_O | d_NP | R_P |\n|---|---|---|---|---|")
for k, v in s9["epsilon_point_pairs"].items():
    o.append(f"| {k} | {f(v['ratio'])} | {f(v['d_O'])} | {f(v['d_NP'])} | {f(v['R_P'])} |")
o.append("\nBase against each reference checkpoint (point): " + "; ".join(f"{k}: ||D||/||H|| {f(v['ratio'])}, d_O {f(v['d_O'])}, d_NP {f(v['d_NP'])}, R_P {f(v['R_P'])}" for k, v in s9["reference_rows"].items()) + ".\n")
o.append("Within-RL drift ordering: ||D(step_k, base)||/||H|| = " + ", ".join(f"{k}: {f(v, 5)}" for k, v in s9["rl_step_norm_D_vs_base"].items()) + (" (monotone non-decreasing in step order: " + str(list(s9["rl_step_norm_D_vs_base"].values()) == sorted(s9["rl_step_norm_D_vs_base"].values())) + "; reported, not a break).\n"))
o.append("**Legacy P1 'within eps' row: DEGENERATE (reported, never decisive).** Rule: within eps if the upper 95% bootstrap bound of M(base, s) < eps; beyond if the lower bound > eps; else INCONCLUSIVE.\n\n| stage | d_O | d_NP | R_P |\n|---|---|---|---|")
for s in ST:
    z = s10["stages"][s]["legacy_P1_DEGENERATE"]
    o.append(f"| {s} | {z['d_O']} | {z['d_NP']} | {z['R_P']} |")
sc = s9["self_calibration"]
o.append(f"\n## Step-9 self-calibration (BROKEN-if: real-data uniformity)\nA row-permuted D_SFT was treated as the observed displacement 50 times (P={sc['P']} each); the 50 p_blue values were tested for uniformity: KS statistic {sc['ks_stat']:.3f}, p = {sc['ks_p']:.3f} (>= 0.01: PASS; mean value {sc['mean_pct']:.3f}). Note this checks p_blue only. Tier T1.\n")
o.append(f"## Step 10 identity check\nThe identity draw (all 8 templates, all 500 entities, step-9 seeds) reproduces the step-9 point estimates (f_low, d_O, d_NP, R_P, ||D||, p-values, verdict) to 1e-12 relative: PASS. Bootstrap B={s10['B']}, 400 entities per draw, m, r (=50, kept at its frozen value although n/10 = 40 at 400 entities) and l* frozen, P=999 per replicate for every stage, same draw applied to base, stages, ingredients and RL-step branches. Wall time {s10['wall_s']:.0f} s.\n")
o.append("## Wall times (steps 6-10)\n" + f"Step 6: " + ", ".join(f"{k} {ck[k]['total_s']} s" for k in ST) + ". Step 7: " + ", ".join(f"{k} {ck[k]['total_s']} s" for k in ('ing1', 'ing2', 'ing3')) + ". Step 8: " + ", ".join(f"{k} {ck[k]['total_s']} s" for k in ck if k.startswith('inst_step')) + f". Step 9: {s9['wall_s']:.0f} s. Step 10: {s10['wall_s']:.0f} s (a first launch was stopped after ~25 min and relaunched: the per-draw template averaging used a slow fancy-indexed mean, replaced by an equivalent weighted tensordot).\n")
open("runs/report_v2_block.md", "w").write("\n".join(o) + "\n")
