# REPORT steps 0-2 (residency-topology-pilot), 2026-11-03/04

## Commits (repo ., branch main)
- step 0 scaffold: aead2609e2b4750f7c80c56de67aa663c69a91a9
- SEAL (PREREG-2026-11-04.md, alone): 9b8cf78799283c1cbc1ee1b8fef51cb38c028d51
- step 1 battery: 61dc090
- step 2 entities: see `git log` (last commit)

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
1. t_k is NOT scale-invariant: it is linear in D, so a joint x5 rescale multiplies it by 5 (tested: covariant to 1e-12; rotation-invariant to 1e-12). DESIGN 2.5 says "every measure" is unchanged by rescale; true of all others. The invariance assertion therefore excludes t_k, and a separate test asserts covariance. Whether t_k should be normalised (e.g. by |D|) is a design decision owed to the driver/ember-lab-admin.
2. theta_max implemented as arctan2(max sin, min cos) and R_P via thin-SVD nuclear norm. Both are the exact DESIGN quantities; the literal arccos(min sigma) and sqrt(eig(G_b G_s)) forms leave 2e-6 deg and 9e-8 residue on rescale/rotation (visible in the sim json), which would fail the 1e-12 BROKEN condition.
3. Spec ambiguity: the 2.5 values of PR_lap and E_low come from the cosine-kNN (binary) Laplacian (sim6), while 2.3 calls the spectral graph the self-tuning kernel. Tests reproduce the table with the kNN Laplacian (rtp.graph.knn_laplacian); both graph types are implemented. The spectral E_low on the self-tuning graph is not in the table.
4. The sim4 single-draw f_low_resid percentiles for white D (0.96, 0.993) looked biased high; the 200-draw KS test finds no non-uniformity (p=0.20). Power of this KS at 200 draws is limited (a bias of a few percent of rank would not be detected).
5. Wall time: step 0 about 1 min compute (torch+deps install), step 1 about 15 s test run (authoring longer), step 2 about 4 min including tokenizer download.

## Step 2: entities (data/entities/, SHA256SUMS covers CSVs, survivors.json, tokenizer hashes, sources)
Sources: Gurnee & Tegmark tables fetched from raw.githubusercontent.com/wesg52/world-models/main/data/entity_datasets/{world_place,historical_figure}.csv (dataset WAS fetchable; committed under source/; a GitHub fetch, outside the "HF + PyPI only" network list - flagged). WordNet 3.0 noun lemmas via PyPI wn==0.0.23 (one-off, `uv run --no-project --with`, not a project dep). wordfreq offline.
| set | survivors | notes |
|---|---|---|
| geography | **500** (no BROKEN) | populated_place/PopulatedPlace with population; names unique in the whole table; admin-unit words excluded; strata (approximate lat/lon boxes) N.Am 100, S.Am 60, Europe 100, Africa 80, Asia 120, Oceania 40, most populous first. Rejected while filling quotas: 670 unclean/admin names, 130 >4 tokens, 0 duplicates, 0 stage-tokenization failures. Token counts (space-prefixed): 1:39, 2:134, 3:213, 4:114 |
| noun controls C1 | 500 | WordNet noun lemmas, zipf>=1.0, matched 1-to-1 on token count and nearest Zipf; mean abs dZipf 0.20, max 1.02 (geography names with Zipf 0 match to the floor 1.0) |
| nonce C2 | 500 | CV-syllable strings, zipf 0, not nouns, token-count histogram equal to geography |
| figures (P2) | 300 | 30 per death-year decile of the top-4000-by-Zipf pool, multiword names, unique; 4 rejected >4 tokens; death years -483..2021 |
Tokenizer check: all 5 OLMo-2 stage repos share an identical tokenizer.json (sha 73fd5254...); Olmo-3-1025-7B too; Olmo-3-7B-RL-Zero-Math has a different tokenizer.json (7137105 vs 7137177 bytes, sha a8c8767b...) but all 8 templates x every candidate tokenize identically (full prompts with special tokens) between the two Olmo-3 checkpoints. Full SHAs for 7 repos are in tokenizer_files.txt (match the DESIGN 12-char prefixes).
Quality limits (T3): geography includes states/regions and neighbourhoods besides cities (G&T populated_place); many figures are obscure (Zipf proxy for fame); nouns/nonce are lowercase while places are capitalised (case confound for F4, owed: add capitalised control variants if wanted); continent strata use coarse boxes, so a few places are mis-stratified.

## Not done
Steps 3+ (no model weights downloaded). Prereg addendum recording the SHA256SUMS is not written (the frozen hashes are in data/entities/SHA256SUMS). Control templates (neutral frames) are defined in rtp.entities.TEMPLATES_CTRL, not yet in an addendum. No Grok/cross-family verification of the battery (same-family, UNVERIFIED).
