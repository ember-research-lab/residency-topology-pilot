# REPORT steps 0-10 v2 (residency-topology-pilot), 2026-11-04: gated base

**Read this first.** v1 of this report (retained verbatim at the end, section "VOID RUN") used hub `main` a1847dff as the base. That checkpoint is an unrelated network (weight cosine to SFT and to every ingredient about 0.0003; T1 measured), so every steps 3-10 verdict in v1 is VOID. AMENDMENT 2 (PREREG-2026-11-04.md, commit 2b8755d) replaces the base with `stage2-ingredient3-step23852-tokens51B` = **f83cd641d4f00954eb25c781e0ac360debb3239e** and adds a fail-closed lineage gate. Steps 3-10 below are re-run on that base. Steps 11 and 12 are not run.

## Lineage gate results (weights only; src/rtp/lineage.py, records data/lineage/*.json; PASS = median rel < 0.5 and median cos > 0.9, keys and shapes identical)

| pair (first 12 of full SHA, parent -> child) | median ||child-parent||/||parent|| | median cosine | gate |
|---|---|---|---|
| 9d3e43659f00 -> 0d85a3d03787 | 0.2311 | 0.9730 | PASS |
| 9d3e43659f00 -> f83cd641d4f0 | 0.2157 | 0.9766 | PASS |
| a1847dff3500 -> 0d85a3d03787 | 1.1101 | 0.0003 | FAIL |
| a1847dff3500 -> ada39260a92e | 1.1088 | 0.0003 | FAIL |
| a1847dff3500 -> b51c5f1b7c83 | 1.1089 | 0.0002 | FAIL |
| a1847dff3500 -> f83cd641d4f0 | 1.1088 | 0.0003 | FAIL |
| a81bae42db39 -> dee26a4074b8 | 0.0022 | 1.0000 | PASS |
| f83cd641d4f0 -> 033e02d7fe4f | 0.0884 | 0.9961 | PASS |
| f83cd641d4f0 -> 0d85a3d03787 | 0.0883 | 0.9961 | PASS |
| f83cd641d4f0 -> 0f49bb9c8bb2 | 0.0884 | 0.9961 | PASS |
| f83cd641d4f0 -> 12cb33498d26 | 0.0884 | 0.9961 | PASS |
| f83cd641d4f0 -> 48d788eca847 | 0.0884 | 0.9961 | PASS |
| f83cd641d4f0 -> 9ab374beda00 | 0.0884 | 0.9961 | PASS |
| f83cd641d4f0 -> a7b49f0c2856 | 0.0884 | 0.9961 | PASS |
| f83cd641d4f0 -> ada39260a92e | 0.1155 | 0.9933 | PASS |
| f83cd641d4f0 -> b51c5f1b7c83 | 0.1157 | 0.9933 | PASS |
| f83cd641d4f0 -> c4b0485961ab | 0.0884 | 0.9961 | PASS |

Short sha map: a1847dff3500 = hub main (regression case, must FAIL: it does, 4 of 4); f83cd641d4f0 = chosen base; 0d85a3d03787 SFT; ada39260a92e / b51c5f1b7c83 ingredients 1/2; 9d3e43659f00 = stage1 final (stage1-step1907359-tokens4001B); c4b0485961ab DPO; 12cb33498d26 RLVR1; 48d788eca847 Instruct; 0f49bb9c8bb2 / 033e02d7fe4f / 9ab374beda00 / a7b49f0c2856 Instruct steps 200/800/1600/2600; a81bae42db39 -> dee26a4074b8 the Olmo-3 7B pair (PASS, so step 11 is gated open). The ingredient1-vs-ingredient2 weight distance is 0.116 (ING3-relative values above; the coordinator's 1e-2 expectation was wrong). Base choice evidence: AMENDMENT 2 table (stage1 final -> SFT 0.231 / cos 0.973 also passes the gate but is the 4T-token stage-1 end, not the 50B-mix final that the card names; ingredients 1/2 are exploratory; SFT is closest to ingredient 3, 0.088).

## Freezes re-derived on the new base (unchanged step-4 rules; configs/pilot.toml [frozen])

| quantity | new base f83cd641 | void a1847dff |
|---|---|---|
| l* | 12 | 12 |
| m / r | 5 / 50 (uncapped r 166) | 5 / 50 (178) |
| probe R2 L0/L4/L8/L12 | 0.276 / 0.639 / 0.788 / 0.824 | 0.280 / 0.540 / 0.754 / 0.822 |
| lambda_2 | 0.429 | 0.462 |
| layer-0 overlap O (F4 threshold 0.8) | 0.200 | 0.170 |
| Spearman u_2 vs zipf / token count (F4 threshold 0.6) | -0.566 / 0.089 | -0.480 / 0.053 |
| top-3 coordinate variance share (outlier arm needed?) | 0.0099 (no) | 0.0109 (no) |
| precision null ||D_prec|| at l*=12 | 1.411 | 1.601 |

Closure on the new base: closure_rel 5.6e-7 (<= 1e-5), re-extraction 5.6e-7 (<= 1e-6); F32 run closure the same order. F4 flags: none raised (T1).


# Steps 6-10 v2 results on the gated base (AMENDMENT 2, 2026-11-04)

Adjudications (a)-(f) of 2026-11-04 carry over unchanged (r=50 at the n/10 cap; m=5; layer-0 mean-over-name-tokens; no zipf partialling; precision null explicit; snapshots kept). The extractions of the non-base checkpoints are reused from the void run (activations depend only on the checkpoint weights, not on the base; closure re-checked on the base only), each re-gated against the new base below.

## Checkpoints (full commit SHAs, snapshots kept in /mnt/e/ai-models/huggingface_cache/hub; data/checkpoints.json)

| key | repo @ revision | full SHA | prefix check | lineage gate vs base: median rel / median cos / result | closure rel (max over L0/4/8/12) | seconds (fetch / total) |
|---|---|---|---|---|---|---|
| sft | allenai/OLMo-2-0425-1B-SFT @ main | 0d85a3d037876ce6ac7d4311d994400fc66ac27f | matches pinned prefix | 0.0883 / 0.9961 / PASS | 7.12e-07 | 34 / 475 |
| dpo | allenai/OLMo-2-0425-1B-DPO @ main | c4b0485961ab24c2433b090f3b922f0913a9290f | matches pinned prefix | 0.0884 / 0.9961 / PASS | 6.97e-07 | 31 / 469 |
| rlvr1 | allenai/OLMo-2-0425-1B-RLVR1 @ main | 12cb33498d26b411c38ac3ca9df27180a1d291b8 | matches pinned prefix | 0.0884 / 0.9961 / PASS | 6.81e-07 | 44 / 491 |
| instruct | allenai/OLMo-2-0425-1B-Instruct @ main | 48d788eca847d4d7548f375ad03d3c9312f6139e | matches pinned prefix | 0.0884 / 0.9961 / PASS | 6.84e-07 | 39 / 488 |
| ing1 | allenai/OLMo-2-0425-1B @ stage2-ingredient1-step23852-tokens51B | ada39260a92eb8cfc2252b9e937f790fd5b0dbc1 | n/a (branch) | 0.1155 / 0.9933 / PASS | 6.11e-07 | 135 / 593 |
| ing2 | allenai/OLMo-2-0425-1B @ stage2-ingredient2-step23852-tokens51B | b51c5f1b7c83a96b9a80bdaf14a2077c672fbccd | n/a (branch) | 0.1157 / 0.9933 / PASS | 6.11e-07 | 90 / 564 |
| ing3 | allenai/OLMo-2-0425-1B @ stage2-ingredient3-step23852-tokens51B | f83cd641d4f00954eb25c781e0ac360debb3239e | n/a (branch) | is the base | 6.08e-07 | 63 / 513 |
| inst_step200 | allenai/OLMo-2-0425-1B-Instruct @ step_200 | 0f49bb9c8bb23d43b51b1732be137a99bfaaf9ec | n/a (branch) | 0.0884 / 0.9961 / PASS | 6.84e-07 | 45 / 519 |
| inst_step800 | allenai/OLMo-2-0425-1B-Instruct @ step_800 | 033e02d7fe4fe254c16c2c21d16bbf08bad1ad77 | n/a (branch) | 0.0884 / 0.9961 / PASS | 6.85e-07 | 110 / 564 |
| inst_step1600 | allenai/OLMo-2-0425-1B-Instruct @ step_1600 | 9ab374beda00b26265133e125046e564fb5d2f26 | n/a (branch) | 0.0884 / 0.9961 / PASS | 6.94e-07 | 136 / 595 |
| inst_step2600 | allenai/OLMo-2-0425-1B-Instruct @ step_2600 | a7b49f0c28561ccc212976a6b036ee556ee61fd7 | n/a (branch) | 0.0884 / 0.9961 / PASS | 6.86e-07 | 119 / 592 |

BROKEN-if checks, steps 3-8 (AMENDMENT 2): lineage gate PASS for every checkpoint against the base (table above; step 9/10 assert it); closure <= 1e-5 and re-extraction <= 1e-6 on every checkpoint; pinned SHA prefixes match for the four stages. ing3 is the base (bit-identical activations to runs/olmo2-1b-base, max abs difference 0.0). Tier T1 (measured).

## Precision null (step 6, explicit)
||D_SFT|| = 102.219 at l*=12 (centred, template-averaged last token, geography). Required > 16.0: PASS (ratio ||D_SFT||/16.0 = 6.4). ||D_prec|| = 1.411; ||D_prec||/||D_SFT|| = 0.0138 < 0.1: PASS. Tier T1.

## Results table (l*=12, m=5, r=50, P=999; Holm at 0.05 over the 8-test family; B=1000 bootstrap)

Family caveat: the Olmo-3 RL-Zero pair (step 11) is not yet run. The family of 8 is held at size 8 with the two absent p-values taken as 1 (the most permissive reading for the present ones); SFT/DPO/Instruct vote in that family. RLVR1 (a post-trained RL stage) is reported with its own Holm over its 2 p-values and is labelled secondary. Verdict words are the preregistered ones only.

| stage | ||D||/||H|| | f_low(R) (null median) | lambda_lin | r_eff(R) | p_blue | p_red | Holm p_blue / p_red | verdict | dPR_cov (N4' 5th / 95th) | d_O (N4' 5/95) | d_NP k=10 | Procrustes R_P | CKA | bootstrap stability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sft | 0.3993 | 0.0008937 (0.0006123) | 0.2886 | 146.2 | 1.000 | 0.001 | 1.000 / 0.008 | **red** | 7.253 (6.648 / 6.86) | 0.07484 (0.03935 / 0.05235) | 0.2732 | 0.02779 | 0.9548 | 0.763 (NOT claimable, < 0.9) |
| dpo | 0.4094 | 0.0008805 (0.0006194) | 0.2824 | 147.4 | 0.999 | 0.002 | 1.000 / 0.014 | **red** | 7.904 (6.804 / 7.014) | 0.07388 (0.04084 / 0.05369) | 0.276 | 0.02907 | 0.9524 | 0.737 (NOT claimable, < 0.9) |
| rlvr1 | 0.4102 | 0.0008827 (0.0006195) | 0.2852 | 148.1 | 1.000 | 0.001 | 1.000 / 0.002 | **red** | 8.055 (6.911 / 7.138) | 0.07513 (0.04153 / 0.05536) | 0.2766 | 0.02926 | 0.9519 | 0.944 (claimable) |
| instruct | 0.4106 | 0.0008801 (0.0006152) | 0.2851 | 147.9 | 0.999 | 0.002 | 1.000 / 0.014 | **red** | 8.067 (6.892 / 7.113) | 0.07614 (0.04213 / 0.05655) | 0.2766 | 0.02929 | 0.9519 | 0.725 (NOT claimable, < 0.9) |

Verdict set per replicate (counts of 1000): sft: {'not red': 237, 'red': 763}; dpo: {'not red': 263, 'red': 737}; rlvr1: {'not red': 56, 'red': 944}; instruct: {'not red': 275, 'red': 725}.

Other non-voting rows (point estimates; none of them votes):

| stage | clustering change (N4' 5/50/95) | E_low(base graph) change (N4' 5/50/95) | coupling C | t_hat_k (k=1..5) | null percentile of dPR_cov / dclust / dE_low / d_O / d_NP |
|---|---|---|---|---|---|
| sft | -0.03863 (-0.037, -0.03017, -0.02316) | -0.02414 (-0.02304, -0.02266, -0.02224) | 0.4808 | -0.047, -0.0606, -0.0534, -0.0191, -0.0271 | 1.000 / 0.011 / 0.001 / 1.000 / 1.000 |
| dpo | -0.04096 (-0.03781, -0.03106, -0.02383) | -0.02599 (-0.02396, -0.02358, -0.0232) | 0.4236 | -0.046, -0.0664, -0.0569, -0.00375, -0.02 | 1.000 / 0.011 / 0.001 / 1.000 / 1.000 |
| rlvr1 | -0.04211 (-0.03987, -0.03252, -0.02547) | -0.02637 (-0.02431, -0.02394, -0.02354) | 0.4273 | -0.0474, -0.069, -0.0592, -0.00403, -0.0259 | 1.000 / 0.016 / 0.001 / 1.000 / 1.000 |
| instruct | -0.04286 (-0.03914, -0.0327, -0.02565) | -0.02641 (-0.0243, -0.02389, -0.0235) | 0.4257 | -0.0467, -0.0686, -0.0584, -0.00378, -0.025 | 1.000 / 0.007 / 0.001 / 1.000 / 1.000 |

Bootstrap 95% CIs (percentile; 400-entity subsample, template multiset): sft: ratio 0.395-0.4347, f_low 0.0003185-0.00107, d_O 0.05684-0.1802; dpo: ratio 0.4052-0.4462, f_low 0.0003123-0.001044, d_O 0.05812-0.1748; rlvr1: ratio 0.4058-0.4475, f_low 0.0003131-0.001045, d_O 0.05834-0.1763; instruct: ratio 0.4061-0.4481, f_low 0.0003128-0.001045, d_O 0.05888-0.1768.

## Sensitivity columns (reported, non-voting)
r/2, r, 2r = 25/50/100 (f_low(R), p_red; red verdict by raw p_red <= 0.05):

| stage | r=25 | r=50 | r=100 |
|---|---|---|---|
| sft | 0.001419, p_red 0.001, lambda_lin 0.216 | 0.0008937, p_red 0.004, lambda_lin 0.2886 | 0.0006246, p_red 0.020, lambda_lin 0.399 |
| dpo | 0.001385, p_red 0.001, lambda_lin 0.2108 | 0.0008805, p_red 0.004, lambda_lin 0.2824 | 0.0006175, p_red 0.025, lambda_lin 0.3918 |
| rlvr1 | 0.00139, p_red 0.001, lambda_lin 0.2135 | 0.0008827, p_red 0.001, lambda_lin 0.2852 | 0.0006159, p_red 0.038, lambda_lin 0.3945 |
| instruct | 0.001389, p_red 0.001, lambda_lin 0.2134 | 0.0008801, p_red 0.001, lambda_lin 0.2851 | 0.0006149, p_red 0.040, lambda_lin 0.3944 |

m = m*-2, m*+2, 10 (f_low(R), p_red, d_O):

| stage | m=3 | m=7 | m=10 |
|---|---|---|---|
| sft | 0.0007164, p_red 0.003, d_O 0.06302 | 0.001125, p_red 0.001, d_O 0.09326 | 0.001605, p_red 0.001, d_O 0.1481 |
| dpo | 0.0007061, p_red 0.002, d_O 0.07632 | 0.001109, p_red 0.001, d_O 0.09383 | 0.001586, p_red 0.002, d_O 0.1482 |
| rlvr1 | 0.0007069, p_red 0.002, d_O 0.07925 | 0.001111, p_red 0.001, d_O 0.09458 | 0.001592, p_red 0.001, d_O 0.1492 |
| instruct | 0.0007053, p_red 0.005, d_O 0.0793 | 0.001108, p_red 0.001, d_O 0.09455 | 0.001588, p_red 0.001, d_O 0.1491 |

## epsilon floor (steps 7-8) and legacy P1 row
epsilon = max(eps_A over the 3 ingredient pairs, eps_N6 over the 3 consecutive RL-step pairs), each the 95th bootstrap percentile of the statistic under the same paired draw (maximum over pairs). Caption: none of these is a true seed replicate; eps_A is a drift reference, not a noise floor.

| statistic | eps_A (ingredient pairs) | eps_N6 (RL steps 200-800-1600-2600) | epsilon |
|---|---|---|---|
| d_O | 0.02936 | 0.000126 | **0.02936** |
| d_NP | 0.1607 | 0.008012 | **0.1607** |
| R_P | 0.01103 | 9.323e-06 | **0.01103** |

Point values of the pairwise statistics (step 9, all templates and entities):

| pair | ||D||/||H_base|| | d_O | d_NP | R_P |
|---|---|---|---|---|
| ing1|ing2 | 0.2453 | 0.01049 | 0.1446 | 0.01071 |
| ing1|ing3 | 0.2438 | 0.0133 | 0.1454 | 0.01066 |
| ing2|ing3 | 0.2452 | 0.01032 | 0.15 | 0.01074 |
| inst_step200|inst_step800 | 0.005079 | 8.39e-06 | 0.0046 | 3.596e-06 |
| inst_step800|inst_step1600 | 0.006515 | 1.353e-05 | 0.006 | 6.294e-06 |
| inst_step1600|inst_step2600 | 0.007582 | 1.434e-05 | 0.0056 | 8.237e-06 |

Base against each reference checkpoint (point): ing1: ||D||/||H|| 0.2438, d_O 0.0133, d_NP 0.1454, R_P 0.01066; ing2: ||D||/||H|| 0.2452, d_O 0.01032, d_NP 0.15, R_P 0.01074; ing3: ||D||/||H|| 0, d_O 2.22e-16, d_NP 0, R_P -2.22e-16; inst_step200: ||D||/||H|| 0.4102, d_O 0.07518, d_NP 0.2764, R_P 0.02926; inst_step800: ||D||/||H|| 0.4106, d_O 0.07577, d_NP 0.277, R_P 0.02932; inst_step1600: ||D||/||H|| 0.4105, d_O 0.07497, d_NP 0.2756, R_P 0.02927; inst_step2600: ||D||/||H|| 0.4109, d_O 0.07557, d_NP 0.2778, R_P 0.02932.

Within-RL drift ordering: ||D(step_k, base)||/||H|| = inst_step200: 0.41017, inst_step800: 0.41056, inst_step1600: 0.4105, inst_step2600: 0.41089 (monotone non-decreasing in step order: False; reported, not a break).

**Legacy P1 'within eps' row: DEGENERATE (reported, never decisive).** Rule: within eps if the upper 95% bootstrap bound of M(base, s) < eps; beyond if the lower bound > eps; else INCONCLUSIVE.

| stage | d_O | d_NP | R_P |
|---|---|---|---|
| sft | beyond eps | beyond eps | beyond eps |
| dpo | beyond eps | beyond eps | beyond eps |
| rlvr1 | beyond eps | beyond eps | beyond eps |
| instruct | beyond eps | beyond eps | beyond eps |

## Step-9 self-calibration (BROKEN-if: real-data uniformity)
A row-permuted D_SFT was treated as the observed displacement 50 times (P=199 each); the 50 p_blue values were tested for uniformity: KS statistic 0.135, p = 0.295 (>= 0.01: PASS; mean value 0.536). Note this checks p_blue only. Tier T1.

## Step 10 identity check
The identity draw (all 8 templates, all 500 entities, step-9 seeds) reproduces the step-9 point estimates (f_low, d_O, d_NP, R_P, ||D||, p-values, verdict) to 1e-12 relative: PASS. Bootstrap B=1000, 400 entities per draw, m, r (=50, kept at its frozen value although n/10 = 40 at 400 entities) and l* frozen, P=999 per replicate for every stage, same draw applied to base, stages, ingredients and RL-step branches. Wall time 3068 s.

## Wall times (steps 6-10)
Step 6: sft 475 s, dpo 469 s, rlvr1 491 s, instruct 488 s. Step 7: ing1 593 s, ing2 564 s, ing3 513 s. Step 8: inst_step200 519 s, inst_step800 564 s, inst_step1600 595 s, inst_step2600 592 s. Step 9: 1929 s. Step 10: 3068 s (a first launch was stopped after ~25 min and relaunched: the per-draw template averaging used a slow fancy-indexed mean, replaced by an equivalent weighted tensordot).


## Flags (facts; no interpretation beyond the preregistered words)
1. The void-run pathology is gone: ||D(base, stage)||/||H_base|| is 0.399 / 0.409 / 0.410 / 0.411 (SFT/DPO/RLVR1/Instruct), the ingredients 1/2 sit at 0.244/0.245 and RL-step branches at 0.410-0.411 (void run: 1.52 and 1.48). ||D_SFT|| = 102.2 > 16.0 (BROKEN-if not triggered), ||D_prec||/||D_SFT|| = 0.0138 < 0.1. Note ||D(base, ing1)|| = 0.244 is a large part of the SFT displacement 0.399: ingredient 1/2 are different runs from the final ingredient 3 and sit nearly as far from the base as post-training does (ing1 vs ing2 0.245). A displacement of this size from a different-data-order sibling is the scale to compare against, and the epsilon floor (below) is set by it.
2. Verdicts at Holm: SFT red (Holm p_red 0.008), DPO red (0.014), RLVR1 red (0.002), Instruct red (0.014); Holm p_blue 1.000 for all four. Bootstrap stability of "red" is 0.763 / 0.737 / 0.944 / 0.725: only RLVR1 reaches the F1 threshold 0.9; SFT, DPO and Instruct do not (stable_claim false). Family caveat (two absent RL-Zero p-values held at 1) as in v1.
3. Literal F1 reading, for the driver (not applied): F1 = "RLVR-final or RL-Zero-Math residual red at l*, p_red <= Holm, stability >= 0.9"; the 1B RLVR1 stage meets it (Holm p_red 0.002, stability 0.944), RL-Zero not run. The red verdict is uniform across SFT through Instruct with similar f_low (0.00088-0.00089, null median 0.0006), so it is not a reward-specific signature in this data: SFT (imitation) is equally red (T1).
4. Sensitivity: at r=100 raw p_red is 0.020 / 0.025 / 0.038 / 0.040 (SFT/DPO/RLVR1/Instruct); the r=100 Holm reading was not run (non-voting column).
5. epsilon (95th bootstrap percentile, max over pairs): d_O 0.0294, d_NP 0.1607, R_P 0.0110, all set by the ingredient pairs (eps_A), not by RL steps (eps_N6 <= 0.008). Legacy P1 DEGENERATE row: all four stages "beyond eps" on d_O, d_NP and R_P.
6. Self-calibration KS p = 0.295 (uniformity not rejected); identity bootstrap draw reproduces step 9 exactly.
7. Extractions of non-base checkpoints were reused from the void run (activations depend only on the checkpoint); the gate was re-run for each against the new base, and the base extraction was redone. Base activations are bit-identical to the ing3 extraction (max abs difference 0.0).
8. Not done: step 11 (Olmo-3 7B pair; its gate already PASSes), step 12 (picture), GitHub issues (local-only repo), a diagnosis of why hub `main` differs from every ingredient.

## Driver adjudication 2026-11-04

1. F1 is NOT met. PREREG §4 names RLVR-final (= Instruct) and RL-Zero-Math, not RLVR1 (secondary, non-voting). Instruct is red at the point estimate but stability 0.725 < 0.9, and §3 line 200 says a verdict is claimed only at stability >= 0.9. Same for SFT (0.763) and DPO (0.737): "red at point estimate, UNSTABLE, no verdict claimed". RLVR1 red/0.944 is reported in the secondary block only. So P1-S is not yet claimable either.
2. Design error, driver-owned, into the kill ledger: for a sequential chain (base->SFT->DPO->RLVR) the preregistered base-relative D for DPO/RLVR/Instruct inherits the SFT displacement (all four ||D||/||H|| ~ 0.40-0.41, near-identical p), so base-relative reward stages cannot test "reward prunes" on OLMo-2 1B. The only clean reward arm in this pilot is Olmo-3 RL-Zero-Math vs its base (RL directly from the pretrained model), i.e. step 11. State this plainly.
3. AMENDMENT 3 (post-hoc, EXPLORATORY, non-voting on P1; to be preregistered for the residency-scale run): stage increments relative to the immediate parent per the model cards, each run through the identical pipeline on the frozen base graph (see PREREG AMENDMENT 3 and the increment section below).

## Incident 2026-11-04: WSL freeze in step 11

- **Last completed action** (data/incident_step11/olmo3-7b-base.log, step11_run.log): the Olmo-3-1025-7B base was materialised (load report, 149 s, 17 blocks) and the 10-prompt closure check PASSED (closure_rel 4.2e-7 at L8, 4.1e-7 at L16; re-extraction 4.2e-7 / 4.1e-7; closure.json saved). The full 8000-prompt extraction had started (batch 64) and never logged EXTRACT_DONE. step11_run.log holds no line after the child launch (the runner prints only on child exit). Repo state intact at 45fb227.
- **Dtype and RSS.** `from_pretrained(dtype=float32, num_hidden_layers=17)` materialised the whole truncated model in **float32**: 17 blocks x 202M + 0.41B embeddings + 0.41B lm_head (the lm_head is in the checkpoint and was loaded although unused) = about 4.2B parameters = about 16.9 GB nominal. Measured VmHWM 14.49 GB at 14 min (T1), with free memory 0.2 GB and 8 GB swap behind a 24 GB WSL cap that also holds about 6 GB of non-reclaimable shared memory and the mmapped bf16 shards in page cache. Peak during the forward pass adds activations on top. An fp32 upcast of the whole model (29 GB) or two models at once would exceed the cap outright.
- **Why the 10 s RSS poll cannot help.** (a) The poll samples; an allocation or the page-cache plus swap thrash that freezes the VM happens in a few seconds, between samples. (b) Once the VM is thrashing swap, the poller itself is not scheduled, so a kill signal is never sent. (c) The threshold (20 GB RSS) was above the headroom actually available (about 12 GB after shared memory), so it could not trigger before exhaustion anyway. Only a kernel-enforced limit acts at allocation time; that is the cgroup cap below.

## Memory cap (CLAUDE.md rule 1) and streaming extraction, live from this incident

- Mechanism live: **systemd-run --user --scope (cgroup v2)**, `-p MemoryMax=16G -p MemorySwapMax=0`. From inside the scope /sys/fs/cgroup/.../memory.max reads 17179869184 and memory.swap.max reads 0. Test: a python allocating and touching 1 GB chunks was OOM-killed by the kernel (CONSTRAINT_MEMCG) at the cap with the VM unaffected: 4 GB cap, 6 GB request: killed after 3 GB; 10 GB cap, 14 GB request: killed after 9 GB. The literal 16 GB cap / 20 GB request test was not run, because only about 12 GB of RAM was free (about 6 GB of the 24 GB is non-reclaimable shared memory) and a request above that could reach the global limit before the cgroup one; the mechanism is identical at 16 GB (memory.max verified). prlimit/ulimit fallback not needed; `rtp.cap.require_cap()` accepts either (cgroup <= 16 GiB with swap 0, or RLIMIT_AS <= 20 GiB) and `step11_run.py` and `step11_extract.py` refuse to start otherwise (verified: running without the scope raises CapError); no override flag.
- Streaming extraction (scripts/step11_extract.py): meta-device skeleton with 17 blocks, only the needed tensors read per key from the safetensors shards and kept in bf16 (lm_head not loaded); each block is upcast to fp32 only for its forward (pre-hook) and returned to bf16 afterwards; embeddings, final norm and rotary are fp32. Compute is fp32 from the released bf16 weights, as preregistered. The closure numbers of the streaming path are identical to the earlier all-fp32 path (4.2127e-07 / 4.0983e-07, T1). One model per process, base then RL-Zero. **Dry run under the cap (load + 10-prompt closure): VmHWM 10.32 GB (base), 10.11 GB (RL-Zero), both < 12 GB**; data/step11_dry_rss.json.

## Step 11: Olmo-3-1025-7B base vs Olmo-3-7B-RL-Zero-Math (T1, measured; a81bae42db39 -> dee26a4074b8, lineage gate PASS)

- Extraction under the cgroup cap, streaming bf16: peak RSS (VmHWM) base 11.58 GB, RL-Zero 11.83 GB; wall 1337 s / 1228 s; closure 4.2e-7 / 4.1e-7 (10 prompts). data/step11_rss.json.
- Freezes on the 7B base (l*=16 = 1/2 L, fixed by truncation): probe R2 L8 0.783, L16 0.845; m=4, r=50, lambda_2 0.474.
- RL-Zero: ||D||/||H|| = 0.0104; f_low(R) = 0.00122 vs N4' median 0.00059, percentile 99.9 (p_red = 0.001, floor of P=999; p_blue = 1.0); Holm (family of 8) p_red = 0.008, p_blue 1.0; lambda_lin 0.396; R effective rank 224 (power gate >= 20 met); d_O 2.1e-05; dPR_cov 0.119; dclust -0.00052.
- Sensitivity: red at r=25 and r=100 (p_red 0.001 / 0.002), m=2, 6, 10 (p_red 0.001 each).
- Bootstrap B=1000: red in 1000/1000, **stability 1.0** (>= 0.9, verdict claimable). Holm family: 1B SFT/DPO/Instruct at step-9 point values (0.008/0.012/0.012 red).
- **Readings.** F1 condition (RL-Zero red, Holm-significant, stability >= 0.9): MET at the clean reward arm. F2 literal wording ("f_low(R) at or above N4' median") is also satisfied numerically (0.00122 > 0.00059), but here "red" already means f_low above the null, so F2 as worded does not discriminate from F1 on this statistic: **flagged for driver adjudication, no reading claimed** beyond the numbers. P1-R as an affirmative claim is not made here; the P2 readout (C1/C2) is not run in this step.
- Caveats (T3): single seed per checkpoint, 1B Holm entries are point values, 7B at 1/2 L only, ||D||/||H|| ~ 1% so the red signal is a small structured displacement.

## Step 12 picture
runs/picture-2026-11-04.png and .pdf (force-added; runs/ is gitignored).

## Step 13: F4 and P2 cost on the 7B pair (T1 measured; extraction under the cgroup cap)

Extraction (scripts/step13_extract.py, streaming bf16): nonce C2 (4000 prompts) and figures (2400, TEMPLATES_GEO as validated at freeze) at l*=16 on both checkpoints, plus layer-0 embeddings of all four sets on the base. C1 (nouns) and geography were already saved in step 11. Dry run VmHWM 10.25 / 11.28 GB; full run VmHWM **13.88 GB (base, includes the layer-0 pass) and 13.51 GB (RL-Zero)**, under the 16 GB cap; the dry-run gate (< 12 GB) was met by the dry runs, the full run (batch 256 on long prompts) peaked higher and is reported as such. Closure 5.7e-7. data/step13_rss.json.

### F4 (PREREG line 249): **FIRES** via (a)  [data/step13_f4_7b.json]
- (a) C1 nouns vs own N4' (P=999, r=50, m=5 by the gap rule): ||D||/||H|| 0.0110, f_low 8.01e-04 vs null median 4.51e-04, percentile 99.9, p_red 0.001, Holm (family of 2) 0.002, verdict **red**. Same pattern and percentile as geography RL-Zero (red, 99.9, ratio 0.0104). C2 nonce (secondary): also red, 99.9, ratio 0.0132. Not stability-bootstrapped (point reading).
- (b) base layer-0 vs l*=16 low band: last-token layer 0 is degenerate (134 duplicate rows, entities sharing a last subword; graph singular, not computable). Fallback mean-over-name-token embeddings (deviation, labelled): O = 0.188 (m=4) < 0.8: does not fire.
- (c) |Spearman(u2, Zipf)| = 0.378, |Spearman(u2, token count)| = 0.014: both < 0.6, does not fire.
- **Reading (as the driver specified):** F4 fires, so the RL-Zero red reading of step 11 is a token-statistics artefact, not a world-structure claim: the same displacement structure appears on matched non-entity strings. (b) and (c) alone would not have fired it.

### P2 cost readout (PREREG lines 256-257)  [data/step13_cost_7b.json]
- C1 geography 5-fold ridge R2: base 0.8454, RL-Zero 0.8453, delta -0.00013, CI95 [-0.00031, +0.00008], Holm(2) p_drop 0.208.
- C2 death-year (300 figures): base 0.6091, RL-Zero 0.6092, delta +0.00004, CI95 [-0.00066, +0.00080], Holm(2) p_drop 0.533.
- Band: eps = 0.00849 (eps_A ingredients 0.00849, eps_N6 RL steps 0.00030), **borrowed from 1B (geography ridge, l=12), T3**; no 7B ingredient or figures activations exist, so the same eps is used for C2. Bootstrap: entities resampled over out-of-fold predictions, B=1000.
- **P2 verdict: FALSIFIED** by the preregistered rule (a red reward stage, and neither C1 nor C2 drops beyond the band). Caveats: the "red" premise is itself F4-artefactual; the band is large relative to the CIs, and the drop/p_drop operationalisation is mine (stated in the JSON rule).

## Step 14 (EXPLORATORY): is red generic smoothness?  (non-voting, analysis only, T1 numbers)

**1B controls (l*=12, own N4', P=999, r=50, m by gap rule per set).** The step-3 activations contain C1 nouns and C2 nonce for the base and all four stages. Controls are red (percentile >= 0.98) at every stage, so the 1B stage reds also appear on controls, with enrichment 1.28-1.33 on controls (geography values in the table).

| set | stage | ratio | enrichment | percentile |
|---|---|---|---|---|
| geo (step 9) | sft | 0.399 | 1.46 | 0.999 |
| C1_nouns | sft | 0.363 | 1.28 | 0.982 |
| C2_nonce | sft | 0.415 | 1.33 | 0.999 |
| geo (step 9) | dpo | 0.409 | 1.42 | 0.998 |
| C1_nouns | dpo | 0.373 | 1.29 | 0.986 |
| C2_nonce | dpo | 0.427 | 1.32 | 0.998 |
| geo (step 9) | rlvr1 | 0.410 | 1.42 | 0.999 |
| C1_nouns | rlvr1 | 0.371 | 1.29 | 0.985 |
| C2_nonce | rlvr1 | 0.425 | 1.33 | 0.998 |
| geo (step 9) | instruct | 0.411 | 1.43 | 0.998 |
| C1_nouns | instruct | 0.372 | 1.29 | 0.986 |
| C2_nonce | instruct | 0.425 | 1.33 | 0.998 |

**7B smoothness diagnostic (RL-Zero vs base, l*=16).** Kernel ridge of D on H_b (RBF, bandwidth = median pairwise distance, lambda by 5-fold CV), 80/20 held-out R2, and a 5-fold cross-fit so every row has a held-out residual that is coloured against N4'.

| set | held-out R2 (80/20) | cross-fit R2 | raw enrich / pct | residual enrich / pct | residual p_blue |
|---|---|---|---|---|---|
| geo | 0.314 | 0.320 | 2.07 / 0.999 | 0.59 / 0.000 | 0.001 |
| C1_nouns | 0.218 | 0.215 | 1.78 / 0.999 | 0.95 / 0.216 | 0.217 |
| C2_nonce | 0.177 | 0.178 | 1.37 / 0.999 | 0.95 / 0.160 | 0.161 |

Geography-minus-C1 raw enrichment: point 0.29 (geo 2.07, C1 1.78, C2 1.37); paired bootstrap B=1000 (80% entity subsample without replacement, geography entity with its matched noun, null P=99; resampling with replacement makes the graph singular) mean 0.23, CI95 [-0.12, 0.55], which includes 0.

**Reading (T3, exploratory).** D is partly a smooth function of H_b (held-out R2 0.18-0.31, highest on geography). Red-ness is not specific to geography (CI includes 0). After removing the smooth part, controls lose the red (percentile 0.16-0.22, not significant), but geography flips to blue (p_blue 0.001). The blue is likely partly a construction effect (the smoother removes low-frequency energy that the permutation null retains), so the residual colour is not read as evidence for or against the hypothesis. This supports "red tracks smooth displacement shared with controls" without proving that N4' is too weak for every fine-tune.

# VOID RUN (base = hub main a1847dff; unrelated network; kept for the kill ledger; do not quote verdicts from this part)

# REPORT steps 0-10 (residency-topology-pilot), 2026-11-03/04 (steps 0-5 sections retained below, steps 6-10 appended at the end)

## Commits (repo ., branch main)
- step 0 scaffold: aead2609e2b4750f7c80c56de67aa663c69a91a9
- SEAL (PREREG-2026-11-04.md, alone): 9b8cf78799283c1cbc1ee1b8fef51cb38c028d51
- step 1 battery: 61dc090
- step 2 entities: f4a3032
- report v1: 3cce0d2
- AMENDMENTS (t^_k, stable forms, kNN-Laplacian rows, Title-case controls; appended to PREREG and mirrored to ember-review DESIGN as CORRECTION): 213e26b1a2ab7d2b22c1aadfcd6ffe422c40f033
- steps 3-4 (extraction, freezes): 244f9c2
- step 5 + this report: see git log

## Versions
python 3.12.12; torch 2.15.0+cpu (CUDA None, CPU wheel from download.pytorch.org/whl/cpu); transformers 5.22.1;
AutoConfig loads `olmo2` (Olmo2Config, 16 layers, d 2048) and `olmo3` (Olmo3Config, 32 layers, d 4096): olmo3 present.

## Battery (uv run pytest -q: 10 passed in ~15 s; tests/test_battery.py)
| assertion | value | result |
|---|---|---|
| stand-in reproduces sim3 (m=4 by gap rule, lam_2..12, r=22) | exact | PASS |
| white D: N4' p_blue uniform, KS over 200 draws (P=199 each) | KS stat 0.075, p=0.200, mean 0.526 | PASS |
| injected blue (L^2 Z): p_blue <= 0.002 (P=999) | 0.0010 (the minimum) | PASS |
| injected red (H@A, residual): p_red <= 0.002 | 0.0010 (the minimum) | PASS |
| rescale x5 and random rotation: every measure changes < 1e-12 | worst 6e-14 (x5), 2e-13 (rot), both theta_max | PASS |
| t_k mean/sd under N4 (999 perms) < 0.1 | [0.012, 0.056, -0.017, 0.007] | PASS |
| t_k witness flips sign | +1.135086 / -1.135086, others <= 2.3e-6 | PASS |
| off-span t_k < 1e-10 | 2.8e-15 | PASS |
| Gram f_low = direct, incl. permuted N4' fast form, 1e-10 | 2.7e-19 | PASS |
| Procrustes thin-SVD form = direct nuclear norm | <1e-10 | PASS |
| 2.5 null table, geometry (sim.py seed): d_O 0.9825, theta 89.78, d_NP 0.9774, R_P 0.2447; structured R_P 0.8178; x7 and rotation = 0 | match table to 2 decimals | PASS |
| 2.5 null table, graph rows (sim6): PR_cov 16.48/252.02, PR_lap 459.85/460.20, clust 0.563/0.035, deg-ent 2.348/2.421, E_low 0.370/0.017, Jaccard 1.0/0.013, chance 0.010, white f_low N4 median 0.0080 | match | PASS |

## Findings and deviations (not hidden)
1. [RESOLVED by amendment 213e26b: t_k replaced by the normalised t^_k; rows re-run, see amendment block in PREREG] t_k was NOT scale-invariant: it is linear in D, so a joint x5 rescale multiplies it by 5 (tested: covariant to 1e-12; rotation-invariant to 1e-12). DESIGN 2.5 says "every measure" is unchanged by rescale; true of all others. The invariance assertion therefore excludes t_k, and a separate test asserts covariance. Whether t_k should be normalised (e.g. by |D|) is a design decision owed to the driver/ember-lab-admin.
2. theta_max implemented as arctan2(max sin, min cos) and R_P via thin-SVD nuclear norm. Both are the exact DESIGN quantities; the literal arccos(min sigma) and sqrt(eig(G_b G_s)) forms leave 2e-6 deg and 9e-8 residue on rescale/rotation (visible in the sim json), which would fail the 1e-12 BROKEN condition.
3. Spec ambiguity: the 2.5 values of PR_lap and E_low come from the cosine-kNN (binary) Laplacian (sim6), while 2.3 calls the spectral graph the self-tuning kernel. Tests reproduce the table with the kNN Laplacian (rtp.graph.knn_laplacian); both graph types are implemented. The spectral E_low on the self-tuning graph is not in the table.
4. The sim4 single-draw f_low_resid percentiles for white D (0.96, 0.993) looked biased high; the 200-draw KS test finds no non-uniformity (p=0.20). Power of this KS at 200 draws is limited (a bias of a few percent of rank would not be detected).
5. Wall time: step 0 about 1 min compute (torch+deps install), step 1 about 15 s test run (authoring longer), step 2 about 4 min including tokenizer download.

## Step 2: entities (data/entities/, SHA256SUMS covers CSVs, survivors.json, tokenizer hashes, sources)
Sources: Gurnee & Tegmark tables fetched from raw.githubusercontent.com/wesg52/world-models/main/data/entity_datasets/{world_place,historical_figure}.csv (dataset WAS fetchable; committed under source/). GitHub raw and the one-off PyPI wn==0.0.23 fetch were outside the stated network list; accepted by the driver after the fact. WordNet 3.0 noun lemmas via PyPI wn==0.0.23 (one-off, `uv run --no-project --with`, not a project dep). wordfreq offline.
| set | survivors | notes |
|---|---|---|
| geography | **500** (no BROKEN) | populated_place/PopulatedPlace with population; names unique in the whole table; admin-unit words excluded; strata (approximate lat/lon boxes) N.Am 100, S.Am 60, Europe 100, Africa 80, Asia 120, Oceania 40, most populous first. Rejected while filling quotas: 670 unclean/admin names, 130 >4 tokens, 0 duplicates, 0 stage-tokenization failures. Token counts (space-prefixed): 1:39, 2:134, 3:213, 4:114 |
| noun controls C1 | 500 (Title-cased per amendment; re-frozen, 0 tokenization failures) | WordNet noun lemmas, zipf>=1.0, matched 1-to-1 on token count and nearest Zipf; mean abs dZipf 0.20, max 1.02 (geography names with Zipf 0 match to the floor 1.0) |
| nonce C2 | 500 | CV-syllable strings, zipf 0, not nouns, token-count histogram equal to geography |
| figures (P2) | 300 | 30 per death-year decile of the top-4000-by-Zipf pool, multiword names, unique; 4 rejected >4 tokens; death years -483..2021 |
Tokenizer check: all 5 OLMo-2 stage repos share an identical tokenizer.json (sha 73fd5254...); Olmo-3-1025-7B too; Olmo-3-7B-RL-Zero-Math has a different tokenizer.json (7137105 vs 7137177 bytes, sha a8c8767b...) but all 8 templates x every candidate tokenize identically (full prompts with special tokens) between the two Olmo-3 checkpoints. Full SHAs for 7 repos are in tokenizer_files.txt (match the DESIGN 12-char prefixes).
Quality limits (T3): geography includes states/regions and neighbourhoods besides cities (G&T populated_place); many figures are obscure (Zipf proxy for fame); nouns/nonce are Title-cased after the amendment (nonce stays lowercase); continent strata use coarse boxes, so a few places are mis-stratified.

## Not done
Steps 3+ (no model weights downloaded). Prereg addendum recording the SHA256SUMS is not written (the frozen hashes are in data/entities/SHA256SUMS). Control templates (neutral frames) are defined in rtp.entities.TEMPLATES_CTRL, not yet in an addendum. No Grok/cross-family verification of the battery (same-family, UNVERIFIED).

## Amendment battery rows (t^_k, after 213e26b)
rescale x5 and rotation: t^_k change < 1e-12 (pass); N4 null mean/sd [0.012, 0.056, -0.017, 0.007]; witness +0.145086 / -0.145086 (analytic value to 1e-12), others <= 3.7e-7; off-span 4.1e-16. 12/12 tests pass. Geography includes some states/regions and neighbourhoods; the figures set is obscure (Zipf proxy for fame); acceptable for the pilot, flagged.

## Step 3: 1B base extraction
- Checkpoint: allenai/OLMo-2-0425-1B main, full SHA a1847dff35000b4271fa70afc5db10fd29fedbdf, 5.94 GB F32 (2 safetensors). Snapshot kept on disk at /mnt/e/ai-models/huggingface_cache/hub/models--allenai--OLMo-2-0425-1B/snapshots/a1847dff... (the configured HF cache is on E:, not ext4 home; E: 9.9 TB free). `df -h /mnt/d` before: 31 GB free (97% used); D: not touched by the download.
- Extraction: layers {0,4,8,12} x 8 templates x 1500 strings (500 geo, 500 noun, 500 nonce) = 12,000 prompts, last-name-token and mean-over-name-tokens vectors; bf16-rounded weights, fp32 compute; runs/olmo2-1b-base/ (acts.npz, manifest.json with token ids; gitignored). The 300 figures were not extracted (P2 probe, owed at the stages).
- Hooked-token assertion: OK on all 12,000 prompts (prompt prefix is a strict token prefix; remainder decodes to the name; no EOS/pad follows).
- Closure (20 random prompts; truncated+hooked batch-20 forward vs unmodified output_hidden_states, one prompt at a time): relative error L0 0, L4 5.9e-7, L8 7.1e-7, L12 6.4e-7 (max abs 3.0e-6) <= 1e-5 PASS. Re-extraction (batch 20 vs singly): same values, <= 7.1e-7 <= 1e-6 PASS (margin 1.4x). Same result on the F32 weights (6.1e-7, 7.1e-7). data/closure_base_bf16.json.
- Layer 12 is a non-final block (16 blocks), so hidden_states[12] is un-normed; hooks used instead of truncated output_hidden_states.

## Step 4: base-only freezes (configs/pilot.toml [frozen]; data/freezes_base_step4.json)
| item | value |
|---|---|
| probe R2 (5-fold ridge, 3-D unit position, template-averaged last token) | L0 0.280, L4 0.540, L8 0.754, **L12 0.822** (not < 0.3 everywhere: no risk-5 stop) |
| l* | **12** (argmax of {4,8,12}) |
| m (gap rule, average over 100 bootstrap draws of templates x 80% entities, frozen) | **5** (point estimate also 5; interior of [4,20]) |
| r (90% of base variance, cap n/10) | **50** (uncapped 90% needs 178 PCs, so the cap binds) |
| lambda_2 / spectrum | 0.462, then 0.832, 0.882, 0.908, 0.916, 0.939, ... up to 0.987 at mode 22: one dominant slow mode, then a nearly flat high spectrum (T1 observation; relevant to f_low power, not a break) |
| outlier check: top-3 native coordinates' share of centred variance | 0.011 (< 0.5): robustness arm not needed |
| layer-0 overlap O (low band at l* vs layer 0, m=5) | **0.170** (theta_max 87.1 deg; F4 threshold 0.8 not met). Layer-0 uses mean-over-name-tokens vectors: last-token layer-0 vectors are raw embeddings of the final subword, only 366 of 500 are distinct, so the self-tuning kernel is undefined there (deviation from the last-token convention, forced) |
| Spearman rho(u_2, zipf) / (u_2, token count) / (u_2, log population) | **-0.480** / 0.053 / -0.161 (F4 threshold 0.6 not crossed; the zipf correlation is moderate, not negligible) |
| F4 flags | none raised |
Note: the layer-0 probe R2 is 0.28 (token embeddings alone predict position to that degree), so a token-statistics component exists in the geography signal; F4 is judged on the preregistered overlap/correlation thresholds above.

## Step 5: precision null (runs/olmo2-1b-base-f32/, data/precision_null_step5.json)
D_prec = H(base F32) - H(base bf16-rounded), centred, template-averaged last token, geography: at l*=12 **||D_prec||_F = 1.601** (0.68% of ||H|| = 236.1); L8 1.392, L4 1.154, L0 0.351. Condition ||D_prec|| < 0.1 ||D_SFT|| is checked at step 6: it requires ||D_SFT|| > 16.0 (6.8% of ||H||) at l*.

## Wall times
Step 0 ~1 min compute; step 1 15 s (tests); step 2 ~4 min; amendments + entity re-freeze ~3 min; step 3 download ~1.5 min, load 78 s, closure ~1 min, extraction 392 s (~10 min total); step 4 44 s; step 5 620 s (F32 load, closure, extraction).

## Not done (steps 0-5)
Steps 6+ (no post-trained checkpoint downloaded); figures-set extraction on base; pairwise bootstrap/N4 tests on real data. Closure margin for re-extraction is thin (7.1e-7 vs 1e-6) because it compares batch-20 to batch-1 numerics; no other checkpoint should show worse, and if one does that is a pipeline-numerics question, not a hypothesis result.

# Steps 6-10 (added 2026-11-04)

Adjudications recorded (driver, 2026-11-04): (a) r=50 at the n/10 cap stands; the r/2, r, 2r column is 25/50/100, run as written. (b) m=5 stands. The base spectrum is one slow mode then flat from 0.83, so f_low power is lower than the sim5 table assumed; no m=1 arm and no new variant was added. (c) Layer-0 mean-over-name-tokens is an accepted forced deviation. (d) rho(u_2, zipf) = -0.48 is below the 0.6 threshold; no partialling, reported only. (e) Precision null checked explicitly at step 6 (below). (f) Every snapshot is kept on disk (E:, full SHAs below).

## Checkpoints (full commit SHAs, snapshots kept in /mnt/e/ai-models/huggingface_cache/hub; data/checkpoints.json)

| key | repo @ revision | full SHA | prefix check | max rel diff vs base weights (6 tensors, layers 0/5/11 q_proj/up_proj) | closure rel (max over L0/4/8/12) | seconds (fetch / total) |
|---|---|---|---|---|---|---|
| sft | allenai/OLMo-2-0425-1B-SFT @ main | 0d85a3d037876ce6ac7d4311d994400fc66ac27f | matches pinned prefix | 1.1281 | 7.12e-07 | 34 / 475 |
| dpo | allenai/OLMo-2-0425-1B-DPO @ main | c4b0485961ab24c2433b090f3b922f0913a9290f | matches pinned prefix | 1.1281 | 6.97e-07 | 31 / 469 |
| rlvr1 | allenai/OLMo-2-0425-1B-RLVR1 @ main | 12cb33498d26b411c38ac3ca9df27180a1d291b8 | matches pinned prefix | 1.1281 | 6.81e-07 | 44 / 491 |
| instruct | allenai/OLMo-2-0425-1B-Instruct @ main | 48d788eca847d4d7548f375ad03d3c9312f6139e | matches pinned prefix | 1.1281 | 6.84e-07 | 39 / 488 |
| ing1 | allenai/OLMo-2-0425-1B @ stage2-ingredient1-step23852-tokens51B | ada39260a92eb8cfc2252b9e937f790fd5b0dbc1 | n/a (branch) | 1.1269 | 6.11e-07 | 135 / 593 |
| ing2 | allenai/OLMo-2-0425-1B @ stage2-ingredient2-step23852-tokens51B | b51c5f1b7c83a96b9a80bdaf14a2077c672fbccd | n/a (branch) | 1.1269 | 6.11e-07 | 90 / 564 |
| ing3 | allenai/OLMo-2-0425-1B @ stage2-ingredient3-step23852-tokens51B | f83cd641d4f00954eb25c781e0ac360debb3239e | n/a (branch) | 1.1269 | 6.08e-07 | 63 / 513 |
| inst_step200 | allenai/OLMo-2-0425-1B-Instruct @ step_200 | 0f49bb9c8bb23d43b51b1732be137a99bfaaf9ec | n/a (branch) | 1.1281 | 6.84e-07 | 45 / 519 |
| inst_step800 | allenai/OLMo-2-0425-1B-Instruct @ step_800 | 033e02d7fe4fe254c16c2c21d16bbf08bad1ad77 | n/a (branch) | 1.1281 | 6.85e-07 | 110 / 564 |
| inst_step1600 | allenai/OLMo-2-0425-1B-Instruct @ step_1600 | 9ab374beda00b26265133e125046e564fb5d2f26 | n/a (branch) | 1.1281 | 6.94e-07 | 136 / 595 |
| inst_step2600 | allenai/OLMo-2-0425-1B-Instruct @ step_2600 | a7b49f0c28561ccc212976a6b036ee556ee61fd7 | n/a (branch) | 1.1281 | 6.86e-07 | 119 / 592 |

BROKEN-if checks, steps 6-8: stage weights not identical to base (all differ, table above); no ingredient equals main; closure <= 1e-5 and re-extraction <= 1e-6 on every checkpoint (all pass, table above); pinned SHA prefixes match for the four stages. Tier T1 (measured).

## Precision null (step 6, explicit)
||D_SFT|| = 358.127 at l*=12 (centred, template-averaged last token, geography). Required > 16.0: PASS (ratio ||D_SFT||/16.0 = 22.4). ||D_prec|| = 1.601; ||D_prec||/||D_SFT|| = 0.0045 < 0.1: PASS. Tier T1.

## Results table (l*=12, m=5, r=50, P=999; Holm at 0.05 over the 8-test family; B=1000 bootstrap)

Family caveat: the Olmo-3 RL-Zero pair (step 11) is not yet run. The family of 8 is held at size 8 with the two absent p-values taken as 1 (the most permissive reading for the present ones); SFT/DPO/Instruct vote in that family. RLVR1 (a post-trained RL stage) is reported with its own Holm over its 2 p-values and is labelled secondary. Verdict words are the preregistered ones only.

| stage | ||D||/||H|| | f_low(R) (null median) | lambda_lin | r_eff(R) | p_blue | p_red | Holm p_blue / p_red | verdict | dPR_cov (N4' 5th / 95th) | d_O (N4' 5/95) | d_NP k=10 | Procrustes R_P | CKA | bootstrap stability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sft | 1.517 | 0.0008822 (0.0005749) | 0.5647 | 208.8 | 1.000 | 0.001 | 1.000 / 0.008 | **red** | 7.955 (56.61 / 57.31) | 0.2032 (0.2366 / 0.2805) | 0.4114 | 0.1106 | 0.8668 | 0.999 (claimable) |
| dpo | 1.523 | 0.000872 (0.0005759) | 0.5614 | 209.6 | 1.000 | 0.001 | 1.000 / 0.008 | **red** | 8.606 (56.88 / 57.57) | 0.2012 (0.2411 / 0.2829) | 0.4142 | 0.1119 | 0.8633 | 0.997 (claimable) |
| rlvr1 | 1.522 | 0.0008726 (0.000573) | 0.5609 | 209.6 | 1.000 | 0.001 | 1.000 / 0.002 | **red** | 8.757 (57.17 / 57.84) | 0.2024 (0.2425 / 0.2829) | 0.415 | 0.1121 | 0.8629 | 1.000 (claimable) |
| instruct | 1.523 | 0.0008725 (0.0005741) | 0.5608 | 209.6 | 1.000 | 0.001 | 1.000 / 0.008 | **red** | 8.769 (57.11 / 57.77) | 0.2038 (0.2424 / 0.2848) | 0.415 | 0.1121 | 0.8628 | 0.999 (claimable) |

Verdict set per replicate (counts of 1000): sft: {'red': 999, 'not red': 1}; dpo: {'red': 997, 'not red': 3}; rlvr1: {'red': 1000}; instruct: {'red': 999, 'not red': 1}.

Other non-voting rows (point estimates; none of them votes):

| stage | clustering change (N4' 5/50/95) | E_low(base graph) change (N4' 5/50/95) | coupling C | t_hat_k (k=1..5) | null percentile of dPR_cov / dclust / dE_low / d_O / d_NP |
|---|---|---|---|---|---|
| sft | -0.046 (-0.1302, -0.119, -0.108) | -0.04553 (-0.1056, -0.1052, -0.1047) | 0.5069 | -1.02, -1.03, -0.989, -1, -0.997 | 0.001 / 1.000 / 1.000 / 0.002 / 0.001 |
| dpo | -0.04833 (-0.132, -0.1207, -0.1095) | -0.04753 (-0.1063, -0.1059, -0.1054) | 0.4785 | -1.02, -1.03, -0.987, -1, -0.997 | 0.001 / 1.000 / 1.000 / 0.001 / 0.001 |
| rlvr1 | -0.04948 (-0.1332, -0.1222, -0.1115) | -0.04781 (-0.1066, -0.1061, -0.1057) | 0.481 | -1.02, -1.03, -0.987, -1, -0.997 | 0.001 / 1.000 / 1.000 / 0.001 / 0.001 |
| instruct | -0.05023 (-0.1337, -0.1226, -0.112) | -0.04789 (-0.1065, -0.1061, -0.1056) | 0.48 | -1.02, -1.03, -0.987, -1, -0.997 | 0.001 / 1.000 / 1.000 / 0.001 / 0.001 |

Bootstrap 95% CIs (percentile; 400-entity subsample, template multiset): sft: ratio 1.51-1.534, f_low 0.0002732-0.001095, d_O 0.1518-0.3117; dpo: ratio 1.517-1.541, f_low 0.0002721-0.001082, d_O 0.1519-0.3132; rlvr1: ratio 1.516-1.54, f_low 0.0002723-0.001084, d_O 0.1521-0.3139; instruct: ratio 1.516-1.54, f_low 0.0002723-0.001084, d_O 0.1539-0.3139.

## Sensitivity columns (reported, non-voting)
r/2, r, 2r = 25/50/100 (f_low(R), p_red; red verdict by raw p_red <= 0.05):

| stage | r=25 | r=50 | r=100 |
|---|---|---|---|
| sft | 0.001791, p_red 0.001, lambda_lin 0.451 | 0.0008822, p_red 0.001, lambda_lin 0.5647 | 0.0005545, p_red 0.014, lambda_lin 0.6913 |
| dpo | 0.001766, p_red 0.001, lambda_lin 0.4484 | 0.000872, p_red 0.001, lambda_lin 0.5614 | 0.0005523, p_red 0.017, lambda_lin 0.6874 |
| rlvr1 | 0.001767, p_red 0.001, lambda_lin 0.4478 | 0.0008726, p_red 0.001, lambda_lin 0.5609 | 0.0005521, p_red 0.013, lambda_lin 0.6871 |
| instruct | 0.001767, p_red 0.001, lambda_lin 0.4477 | 0.0008725, p_red 0.001, lambda_lin 0.5608 | 0.0005524, p_red 0.015, lambda_lin 0.687 |

m = m*-2, m*+2, 10 (f_low(R), p_red, d_O):

| stage | m=3 | m=7 | m=10 |
|---|---|---|---|
| sft | 0.0007035, p_red 0.001, d_O 0.1359 | 0.001185, p_red 0.001, d_O 0.1526 | 0.001727, p_red 0.001, d_O 0.222 |
| dpo | 0.0006943, p_red 0.001, d_O 0.1503 | 0.001171, p_red 0.001, d_O 0.1551 | 0.001711, p_red 0.001, d_O 0.2245 |
| rlvr1 | 0.0006948, p_red 0.001, d_O 0.1531 | 0.001172, p_red 0.001, d_O 0.1565 | 0.001712, p_red 0.001, d_O 0.2249 |
| instruct | 0.0006949, p_red 0.001, d_O 0.1534 | 0.001172, p_red 0.001, d_O 0.157 | 0.001712, p_red 0.001, d_O 0.2252 |

## epsilon floor (steps 7-8) and legacy P1 row
epsilon = max(eps_A over the 3 ingredient pairs, eps_N6 over the 3 consecutive RL-step pairs), each the 95th bootstrap percentile of the statistic under the same paired draw (maximum over pairs). Caption: none of these is a true seed replicate; eps_A is a drift reference, not a noise floor.

| statistic | eps_A (ingredient pairs) | eps_N6 (RL steps 200-800-1600-2600) | epsilon |
|---|---|---|---|
| d_O | 0.02936 | 0.000126 | **0.02936** |
| d_NP | 0.1607 | 0.008012 | **0.1607** |
| R_P | 0.01103 | 9.323e-06 | **0.01103** |

Point values of the pairwise statistics (step 9, all templates and entities):

| pair | ||D||/||H_base|| | d_O | d_NP | R_P |
|---|---|---|---|---|
| ing1|ing2 | 0.2659 | 0.01049 | 0.1446 | 0.01071 |
| ing1|ing3 | 0.2643 | 0.0133 | 0.1454 | 0.01066 |
| ing2|ing3 | 0.2658 | 0.01032 | 0.15 | 0.01074 |
| inst_step200|inst_step800 | 0.005507 | 8.39e-06 | 0.0046 | 3.596e-06 |
| inst_step800|inst_step1600 | 0.007063 | 1.353e-05 | 0.006 | 6.294e-06 |
| inst_step1600|inst_step2600 | 0.00822 | 1.434e-05 | 0.0056 | 8.237e-06 |

Base against each reference checkpoint (point): ing1: ||D||/||H|| 1.476, d_O 0.09676, d_NP 0.368, R_P 0.09807; ing2: ||D||/||H|| 1.481, d_O 0.1036, d_NP 0.3664, R_P 0.09787; ing3: ||D||/||H|| 1.474, d_O 0.1105, d_NP 0.3646, R_P 0.0975; inst_step200: ||D||/||H|| 1.522, d_O 0.2025, d_NP 0.4162, R_P 0.1121; inst_step800: ||D||/||H|| 1.522, d_O 0.2033, d_NP 0.4158, R_P 0.1122; inst_step1600: ||D||/||H|| 1.522, d_O 0.2022, d_NP 0.4146, R_P 0.1121; inst_step2600: ||D||/||H|| 1.523, d_O 0.203, d_NP 0.4154, R_P 0.1122.

Within-RL drift ordering: ||D(step_k, base)||/||H|| = inst_step200: 1.5218, inst_step800: 1.5222, inst_step1600: 1.5224, inst_step2600: 1.5227 (monotone non-decreasing in step order: True; reported, not a break).

**Legacy P1 'within eps' row: DEGENERATE (reported, never decisive).** Rule: within eps if the upper 95% bootstrap bound of M(base, s) < eps; beyond if the lower bound > eps; else INCONCLUSIVE.

| stage | d_O | d_NP | R_P |
|---|---|---|---|
| sft | beyond eps | beyond eps | beyond eps |
| dpo | beyond eps | beyond eps | beyond eps |
| rlvr1 | beyond eps | beyond eps | beyond eps |
| instruct | beyond eps | beyond eps | beyond eps |

## Step-9 self-calibration (BROKEN-if: real-data uniformity)
A row-permuted D_SFT was treated as the observed displacement 50 times (P=199 each); the 50 p_blue values were tested for uniformity: KS statistic 0.170, p = 0.099 (>= 0.01: PASS; mean value 0.588). Note this checks p_blue only. Tier T1.

## Step 10 identity check
The identity draw (all 8 templates, all 500 entities, step-9 seeds) reproduces the step-9 point estimates (f_low, d_O, d_NP, R_P, ||D||, p-values, verdict) to 1e-12 relative: PASS. Bootstrap B=1000, 400 entities per draw, m, r (=50, kept at its frozen value although n/10 = 40 at 400 entities) and l* frozen, P=999 per replicate for every stage, same draw applied to base, stages, ingredients and RL-step branches. Wall time 3115 s.

## Wall times (steps 6-10)
Step 6: sft 475 s, dpo 469 s, rlvr1 491 s, instruct 488 s. Step 7: ing1 593 s, ing2 564 s, ing3 513 s. Step 8: inst_step200 519 s, inst_step800 564 s, inst_step1600 595 s, inst_step2600 592 s. Step 9: 1685 s. Step 10: 3115 s (a first launch was stopped after ~25 min and relaunched: the per-draw template averaging used a slow fancy-indexed mean, replaced by an equivalent weighted tensordot).

## Flags (facts; no interpretation beyond the preregistered words)
1. Every stage contrast is dominated by one common displacement: ||D(base, stage)||/||H_base|| is 1.52 for all four stages and 1.48 for the three stage-2 ingredients, while stage-to-stage distances at l*=12 are small (centred, absolute: SFT-DPO 15.6, DPO-Instruct 5.5; ingredient-ingredient about 62.5; ingredient1-SFT 117.6; base-SFT 358.1; ||H_base|| = 236.1). The base checkpoint (OLMo-2-0425-1B main, a1847dff) is therefore far from every other checkpoint measured, including its own stage-2 ingredient branches. Its weights differ from every other checkpoint by a relative 1.13 in the sampled tensors. Because of this the four stage rows are nearly identical, the stage contrasts do not separate SFT from the reward stages, and the epsilon floor (0.0294 for d_O) is 7x smaller than the base-vs-stage d_O (about 0.20). Tier T1 (measured); the cause is not diagnosed here. Closure and the per-checkpoint weight comparisons passed, so this is not an extraction fault as far as the pipeline entailment checks can tell.
2. Literal reading of the falsifier text, for the driver to adjudicate (not applied): F1 names RLVR-final; RLVR1 is red with Holm p_red 0.002 (own 2-test Holm) and stability 1.000, so the F1 condition as written is met on the 1B RLVR1 stage (RL-Zero not run). F3 (SFT f_low percentile <= RLVR's through each stage's own N4'): both percentiles sit at the 0.999 floor of the test (p_red 0.001), so the inequality holds as a tie. F2 (RLVR f_low at or above its N4' median): f_low 0.000873 vs null median 0.000573, condition met. F4 flags from step 4 are unchanged (none raised).
3. dPR_cov is 8.0-8.8 observed against N4' 5th/95th of about 57; clustering and E_low(base graph) change are negative, both above the N4' 95th as a percentile 1.000 (observed values are larger than every permuted value). d_O observed (0.20) lies below the N4' 5th percentile (0.24). These are reported rows, not interpreted.
4. r=100 (2r) weakens the red p-values to 0.013-0.017 but they remain below 0.05 raw; the Holm-corrected reading at r=100 was not run (sensitivity column is non-voting).
5. Bootstrap used r=50 although n/10=40 at 400 entities (r is frozen); the self-calibration used P=199.
6. Not done: step 11 (Olmo-3 7B pair), step 12 (picture), GitHub issues (this repo is local-only); the 8-test Holm family is incomplete until step 11 (two p-values held at 1).


## Base checkpoint diagnosis (2026-11-04, after the driver's hold; read-only, nothing fixed)

**Status of the step 6-10 verdict column: DEGENERATE-pending (driver instruction) until the base question is resolved.**

1. Loading (transformers, fp32, output_loading_info): base, SFT and ingredient1 each have 0 missing_keys, 0 unexpected_keys, 0 mismatched_keys; tie_word_embeddings False. No newly-initialised weights. (data/base_diagnosis_load_loss.json)

2. Raw files (safetensors / torch.load weights_only, no transformers; data/base_diagnosis_tensors.json): all four checkpoints have 179 keys, identical key sets and shapes (no key or shape differences in any pair). Dtypes: base, ingredient1, ingredient2 float32; SFT bfloat16. Relative Frobenius difference ||A-B||/||B|| over 179 tensors:

| pair | median | min | max | median by type (q/k/v/o/gate/up/down/norm/embed/lm_head) |
|---|---|---|---|---|
| base vs ingredient1 | 2.175 | 0.113 | 2.997 | 2.318 / 2.415 / 2.198 / 2.198 / 2.112 / 2.184 / 2.195 / 0.4461 / 1.467 / 1.905 |
| base vs SFT | 2.166 | 0.112 | 2.98 | 2.306 / 2.401 / 2.189 / 2.189 / 2.104 / 2.174 / 2.185 / 0.4433 / 1.467 / 1.912 |
| SFT vs ingredient1 | 0.1478 | 0.004705 | 0.231 | 0.1996 / 0.2117 / 0.1408 / 0.1437 / 0.1786 / 0.1808 / 0.1807 / 0.00683 / 0.02093 / 0.1246 |
| ingredient1 vs ingredient2 | 0.1158 | 0.000736 | 0.2008 | 0.1653 / 0.1753 / 0.1154 / 0.1159 / 0.1555 / 0.1551 / 0.1541 / 0.003234 / 0.01895 / 0.09282 |

Ten largest base-vs-ingredient1 tensors: layers.0.self_attn.q_proj.weight 2.997, layers.0.self_attn.k_proj.weight 2.876, layers.0.mlp.gate_proj.weight 2.609, layers.1.mlp.gate_proj.weight 2.551, layers.1.self_attn.k_proj.weight 2.536, layers.2.self_attn.k_proj.weight 2.526, layers.10.self_attn.k_proj.weight 2.507, layers.1.self_attn.q_proj.weight 2.500, layers.2.self_attn.q_proj.weight 2.485, layers.4.self_attn.q_proj.weight 2.471. The difference is not concentrated: every projection matrix in every layer is at 1.9-3.0 (median 2.18); only the norm weights (0.45) and nothing else are smaller. Sampled tensors: cosine(base, ingredient1) = 0.006 (embed_tokens), -0.0001 (L5 up_proj), 0.0003 (L5 q_proj), -0.006 (lm_head), 0.956 (L5 post_attention_layernorm weight); weight-norm ratio base/ingredient1 = 1.08 (embed), 1.94 (up_proj), 2.23 (q_proj), 1.61 (lm_head). Ingredient1 vs ingredient2 and SFT vs ingredient1 are 0.12 and 0.15 (the coordinator's expected ~1e-2 does not hold for these either, but they are 15-20x closer than base is to any of them; SFT-ingredient1 pattern is concentrated in q/k projections, the usual fine-tuning signature).

3. Integrity: base shard sha256 (computed locally) c52e4ac8bf4867e698d908f9e8c2b84aa891b91dff066d46bf2a7cf453ad77fb and 5f80762572fe28aca49b01b98895d96ca529b9792fc10925a977a161ad768d30 equal the LFS sha256 from model_info(files_metadata=True) for revision main; index json lists 179 tensors (total_size 5939666944) and both shards are present with the sizes the hub reports (4983360992, 956326560). Resolved revision of main = a1847dff35000b4271fa70afc5db10fd29fedbdf (hub and fetch_base.log). fetch_checkpoint.py passes the repo and the hub-resolved sha of the requested revision (base: allenai/OLMo-2-0425-1B main); no different repo or revision was used. The ingredient shards have the same byte sizes as the base shards but different hashes (fac7173f..., 270e9ab0...).

4. config.json: base and ingredient1 are byte-identical files (same blob 0e6dffcd...): olmo2, hidden 2048, 16 layers, 16 heads, intermediate 8192, vocab 100352, rope_theta 500000, rope_scaling None, rms_norm_eps 1e-6, tie_word_embeddings False. SFT differs only in torch_dtype (bfloat16 vs float32) and transformers_version (4.50.0 vs 4.50.3).

5. Mean token loss, 20 fixed English sentences, fp32: base 2.643, ingredient1 2.638, SFT 3.022 nats (all far below the 6-nat mis-load threshold; base is a working language model).

6. ingredient1 vs ingredient2 median relative weight distance 0.116 (max 0.201); ingredient1 vs SFT 0.148 (max 0.231). See table.

**Cause (T1, measured): not a loading or mapping bug on our side.** The files on the hub for OLMo-2-0425-1B `main` (a1847dff) are intact (hash-verified), load cleanly, and give a healthy model, but their weights are essentially uncorrelated with those of the stage-2 ingredient branches (cosine ~0 in embeddings and projection matrices, uniform relative distance ~2.2 across all layers), whereas the ingredients, SFT and the later stages are mutually close (0.12-0.15). That is a different weight lineage (different initialisation or run), not a layout bug, which would concentrate in particular tensors. Why the hub `main` is not the soup of its own ingredients is **undiagnosed**: candidate explanations (main is a different run than the branches; the branches belong to the Instruct/post-training lineage rather than to main's pretraining run; hub branch naming) need a source outside this repo. Consequence for the pilot: every base-vs-stage contrast in steps 6-10 compares two unrelated networks; the ||D||/||H|| of 1.5, the identical stage rows and the red verdicts are therefore not interpretable as training-stage effects. Per instruction nothing was changed or re-run; steps 3-5 were not repeated because there is no loader fault to fix. Decision owed by the driver: choose a different base (for example ingredient soup, or one ingredient branch as the pre-post-training reference) and re-run, or abandon the 1B pair.
