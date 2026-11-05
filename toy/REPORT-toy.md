# REPORT-toy: seed-replicate null, SmolLM2-360M (2026-11-05)

Prereg: `PREREG-toy-2026-11-04.md` (sealed) + AMENDMENT 1. Numbers: `toy/data/analysis_results.json`, log `toy/runs/t4_analyze.log`, picture `toy/runs/picture-toy.png|.pdf`. Tiers: T1 = computed on this data; T3 = interpretive. One 360M model, fp32, l* = 16 (L = 32), single run of the analysis.

## Verdict (T3)
F-T1, F-T2, F-T3, F-T4 did not fire. The RL entry gate fired (base pass rate 0.04 < 0.05); the RL arm was not run (see "RL" below). Pass-line shape: SFT Delta CI [-0.46, +0.05] includes 0, so the P1-S analogue (Delta CI > 0) is NOT met; DPO Delta CI [-0.41, +0.26] includes 0 (DPO is non-voting). Per AMENDMENT 1 item 2, "CI includes 0" means "not distinguishable at this n", never "equal". No statement is made about "imitation activates" or "reward prunes".

## Setup
- Base SmolLM2-360M (sha f8027fd0, Apache-2.0); fallback pythia-410m (sha 9879c9b5) gated but not trained. Full fine-tune, fp32, AdamW(0.9,0.95), wd 0, no dropout; seed moves data order only.
- Stages (each parent = base): SFT seeds 0-4 (5 seeds; seeds 3-4 added under AMENDMENT 1 item 2), DPO seeds 0-2, GRPO none.
- Slices (`toy/data/slices.json`): SFT tulu-3-sft-mixture, 499,712 packed tokens, 976 sequences, 219,283 loss tokens, 122 steps. **Owned deviation:** 500k tokens, not the brief's ~1M (prereg sec.1). DPO: 1000 pairs, 282,770 chosen+rejected tokens (**owned deviation:** ~283k, not the ~500k planning figure), 62 steps, beta 0.1.
- Frozen (from base only): m = 4 for all four sets (geo n=464, nouns 490, nonce 466, fig 299; r = 46/19/24/29). **m = 4 is the lower edge of the allowed range [4,20].** The parent pilot treated that edge as a BROKEN-if. Here it is flagged, not waived: every enrichment number below inherits the caveat (T3).
- Paired geo/C1 set: 454 pairs. Bootstrap B = 1000 (real), B = 200 (calibration), 80% entity subsamples.
- Analysis ran under the rule-1 cap; no script fix was needed.

## Gate F-T1 (base probe: ridge R^2 of 3-D position, criterion: fires if < 0.5)
| base | R^2 | closure | verdict |
|---|---|---|---|
| SmolLM2-360M | 0.683 | 5.1e-7 | pass (T1) |
| pythia-410m | 0.594 | 6.7e-7 | pass (T1) |

## Training table (T1)
Loss = mean of last 10 steps (single-step losses are noisy). DPO acc = mean preference accuracy over last 10 steps (batch of 16 pairs, so ±0.1 per step). Closure = max relative error of hooked vs HF hidden state at l* (criterion <= 1e-5).
| run | wall s | final loss (last-10 mean; last step) | DPO acc | VmHWM GB | closure |
|---|---|---|---|---|---|
| sft_s0 | 4565 | 2.133; 2.029 | - | 11.20 | 5.2e-7 |
| sft_s1 | 5255 | 2.052; 1.929 | - | 11.34 | 6.1e-7 |
| sft_s2 | 5547 | 2.100; 2.409 | - | 11.14 | 5.2e-7 |
| sft_s3 | 4351 | 2.064; 2.197 | - | 11.22 | 5.4e-7 |
| sft_s4 | 4350 | 2.086; 2.006 | - | 11.16 | 5.3e-7 |
| dpo_s0 | 4045 | 0.673; 0.671 | 0.59 | 10.91 | 5.0e-7 |
| dpo_s1 | 3189 | 0.677; 0.672 | 0.59 | 10.90 | 5.6e-7 |
| dpo_s2 | 3204 | 0.674; 0.644 | 0.54 | 10.90 | 5.5e-7 |
No non-finite loss; all activations differ from base (F-T4 not fired). DPO loss stays near ln 2 = 0.693 (T1): a small effective update at lr 1e-6 over 62 steps.

## Readings (T1 numbers; T3 reading)
Norms are ‖.‖_F / ‖H_base (centred)‖_F. D_stage = seed-mean displacement from base; D_seed = pairwise difference of same-stage seeds (SFT 10 pairs, DPO 3 pairs). **Ratio = median ‖D_seed‖ / ‖D_stage‖ is a toy OUTPUT** (no published value, T3 status per AMENDMENT 1/prereg sec.4). For pure seed noise with k seeds the ratio would be sqrt(2k) (SFT 3.2, DPO 2.4); observed values are far below, so the seed-mean displacement is dominated by a component common to the seeds (T1 numbers, T3 reading).

| stage, set | ‖D_stage‖ | ‖D_seed‖ median (range) | ratio | excess ‖D_stage‖^2 - ‖D_seed‖^2/(2k) |
|---|---|---|---|---|
| SFT geo | 0.02035 | 0.0099 (0.0093-0.0111) | 0.488 | 4.0e-4 |
| SFT figures | 0.01990 | 0.0092 (0.0079-0.0126) | 0.464 | 3.9e-4 |
| SFT C1 nouns | 0.01431 | 0.0060 (0.0055-0.0069) | 0.416 | 2.0e-4 |
| SFT C2 nonce | 0.02096 | 0.0084 (0.0080-0.0109) | 0.401 | 4.3e-4 |
| DPO geo | 0.00102 | 0.00032 (0.00028-0.00033) | 0.314 | 1.0e-6 |
| DPO figures | 0.00084 | 0.00025 (0.00023-0.00027) | 0.302 | 7.0e-7 |
| DPO C1 nouns | 0.00071 | 0.00022 (0.00019-0.00022) | 0.304 | 5.0e-7 |
| DPO C2 nonce | 0.00099 | 0.00030 (0.00028-0.00033) | 0.304 | 9.6e-7 |

DPO displacement is about 20x smaller than SFT (geo 0.0010 vs 0.0204). The DPO seed band rests on 3 pairs from 3 seeds, i.e. about 2 degrees of freedom (AMENDMENT 1 item 2); the 3 pairs share seeds so they are dependent. SFT has 5 seeds (4 df). Kernel-ridge smoothness R^2 of D on H_b (stage vs seed-pair median): SFT geo 0.178 vs about 0.11; DPO geo 0.107 vs 0.114 (T1; per-pair values in the json).

**Enrichment E = f_low(stage)/median f_low(seed pairs), 95% bootstrap CI (E > 1: more low-band than seed noise).**
| stage | geo (entity) | figures (entity) | C1 nouns (control) | C2 nonce (control) |
|---|---|---|---|---|
| SFT | 0.976 [0.82, 1.09] | 0.919 [0.77, 1.05] | 1.239 [0.96, 1.39] | 1.156 [0.87, 1.47] |
| DPO | 0.976 [0.76, 1.21] | 0.908 [0.81, 1.05] | 0.955 [0.85, 1.33] | 0.728 [0.62, 1.48] |

Primary Delta = E(geo) - E(C1), paired: SFT -0.263 [-0.463, +0.049]; DPO +0.022 [-0.406, +0.256]. Secondary (non-voting) E(fig) - E(nonce): SFT -0.237, DPO +0.180 (no CI). All entity CIs contain 1; all Delta CIs contain 0. Bootstrap means sit below point values for DPO Delta (-0.085 vs +0.022), a sign of coarse 3-pair resampling (T3). No enrichment value is exactly 1.0 or NaN; no sanity bug found.

**Planted-rewiring calibration** (18 replicates per s: 2 stages x 3 seed pairs x 3 rewiring seeds; GRPO absent so fewer than the prereg's 27; detection = Delta CI excludes 0 from above):
| s | 0 (size) | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|
| detection rate | 0.00 | 1.00 | 1.00 | 1.00 |
T3 caveat: the planted signal is injected at the scale of seed noise (pseudo-stage = pure noise), so this shows the test separates a planted low-mode component from seed noise of that scale; it says nothing about power at the real stage amplitude, and at m = 4 the low-mode subspace is only 4-dimensional. 18 replicates are not independent (3 pairs reused over 3 rewiring seeds).

## Falsifiers
| falsifier | prereg criterion | number | fired |
|---|---|---|---|
| F-T1 | ridge R^2 < 0.5, or survivors < 300, or battery/closure fails | 0.683 / 0.594; geo 464, C1 490; closure 5e-7 | no |
| F-T2 | median over stages of ‖D_stage‖ <= ‖D_seed‖ (geo) | 0.01068 vs 0.00512 (stages SFT, DPO) | no |
| F-T3 | power(s=0.10) < 0.8 or size(s=0) > 0.10 | power 1.00, size 0.00 | no |
| F-T4 | battery failure, closure failure, non-finite loss, weights = base | max closure 6.1e-7; losses finite; acts differ from base | no |
| RL entry gate (AMENDMENT 1) | base pass rate in [5%, 60%], n = 200 | 0.04 | **fired: arm not run** |
The pilot battery (`tests/test_battery.py`) was run before the extraction jobs per commit ec4eaaa (16 passed); not re-run in this analysis session.

## RL gate (T1)
Base pass rate on 200 held-out arithmetic items (greedy, first-integer exact match) = 0.04, below the 5% floor, so GRPO was not run and contributes nothing to F-T2, the enrichment, or the Pass line. This is the gate doing its job, not a failure.

## Kill list (verbatim from AMENDMENT 1 item 3; binding)
No claim about "reward prunes" or "imitation activates" at 360M. No world-structure claim unless the base-probe gate and the planted control both pass. No scale-transfer claim to 1B/7B. No CKA-based "no change" claim. No GRPO conclusion without a shown reward gain.

## What was not done
RL arm (gate fired); no LoRA arm; no second size; no chat-template extraction arm; no layer sweep (single l*); no precision null; SFT 500k not ~1M tokens; DPO 283k tokens; pythia gated only, not trained; calibration used the first 3 pairs per stage (SFT has 10); seed-band power with 3-5 seeds is limited; the picture has no GRPO row.
