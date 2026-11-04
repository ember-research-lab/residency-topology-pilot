# Changelog

## [Unreleased]
- Step 0: uv project (python 3.12, CPU torch), layout per DESIGN 6.2, CLAUDE.md.
- Prereg sealed: `PREREG-2026-11-04.md` (DESIGN 2026-11-04, approved by ember-lab-admin).
- Step 1: synthetic battery. `rtp.graph`, `rtp.measures`, `rtp.nulls`; `tests/test_battery.py` (10 tests, all pass); driver sims copied unchanged to `tests/sim/`.
- Step 2: frozen entity sets (500 geography, 500 noun controls, 500 nonce controls, 300 figures) + SHA256SUMS; tokenizer files for 7 repos hashed; identical tokenization asserted within each family.
- Amendments before first data: t_k normalised, stable theta/R_P forms, kNN-Laplacian reference rows, Title-cased noun controls (see PREREG-2026-11-04.md).
- Steps 3-4: OLMo-2-0425-1B base extracted (4 layers x 8 templates x 1500 strings, bf16-rounded weights); closure passed; base-only freezes (l*, m, r) written to configs/pilot.toml.
