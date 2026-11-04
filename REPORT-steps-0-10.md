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

