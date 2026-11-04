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
- lineage gate (src/rtp/lineage.py, scripts/lineage_check.py): weights-only parent/child check, fail closed in extract/step9/step10; main a1847dff FAILs (regression case).
- steps 3-5 re-run on the gated base stage2-ingredient3 f83cd641 (AMENDMENT 2): l*=12, m=5, r=50 unchanged; void a1847dff data archived in data/void_a1847dff/.
- REPORT-steps-0-10.md v2: steps 3-10 re-run on gated base f83cd641 (all four stages red at Holm, stability 0.73-0.94, ||D||/||H|| 0.40); v1 verdicts retained as VOID.
- scripts/step_increments.py: AMENDMENT 3 exploratory parent-relative increments.
- memory cap rule: model jobs run in a systemd-run cgroup scope (MemoryMax=16G, swap 0); src/rtp/cap.py fail-closed check; step-11 7B extraction now streams bf16 weights (VmHWM ~10.3 GB). WSL freeze incident documented in REPORT v2.
- step 11 (Olmo-3 7B RL-Zero vs base) results, Holm family of 8, step-12 picture.
- step 13: F4 + P2 cost on the Olmo-3 7B pair (F4 fires via C1; P2 falsified).
- step 14 (exploratory): 1B controls colour + 7B smoothness diagnostic.
- step 15 (exploratory): synthetic positive control for the smooth null (readouts i, ii, iii-old, iii).
