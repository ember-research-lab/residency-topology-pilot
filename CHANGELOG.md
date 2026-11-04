# Changelog

## [Unreleased]
- Step 0: uv project (python 3.12, CPU torch), layout per DESIGN 6.2, CLAUDE.md.
- Prereg sealed: `PREREG-2026-11-04.md` (DESIGN 2026-11-04, approved by ember-lab-admin).
- Step 1: synthetic battery. `rtp.graph`, `rtp.measures`, `rtp.nulls`; `tests/test_battery.py` (10 tests, all pass); driver sims copied unchanged to `tests/sim/`.
- Step 2: frozen entity sets (500 geography, 500 noun controls, 500 nonce controls, 300 figures) + SHA256SUMS; tokenizer files for 7 repos hashed; identical tokenization asserted within each family.
- Amendments before first data: t_k normalised, stable theta/R_P forms, kNN-Laplacian reference rows, Title-cased noun controls (see PREREG-2026-11-04.md).
- Steps 3-4: OLMo-2-0425-1B base extracted (4 layers x 8 templates x 1500 strings, bf16-rounded weights); closure passed; base-only freezes (l*, m, r) written to configs/pilot.toml.
- Step 5 precision null: ||D_prec|| = 1.601 at l*=12 (0.68% of ||H||). Report renamed REPORT-steps-0-5.md.
- step 6: post-trained 1B stages (SFT, DPO, RLVR1, Instruct) extracted with per-checkpoint closure check; precision-null condition checked on D_SFT (data/checkpoints.json).
- step 7: three stage-2 ingredient branches extracted (epsilon floor inputs).
- step 8: Instruct RL-step branches step_200/800/1600/2600 extracted (within-RL drift inputs).
- step 9: all measures + N4' (P=999) + Holm over the preregistered family; results in data/step9_results.json.
- step 10: bootstrap B=1000 with verdict stability, epsilon floor, legacy P1 (DEGENERATE) row; REPORT-steps-0-10.md.
- base diagnosis: hub main of OLMo-2-0425-1B is uncorrelated with its stage-2 ingredients (not a load bug); verdicts held as DEGENERATE-pending.
