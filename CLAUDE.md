# residency-topology-pilot

Preregistered CPU pilot: "imitation activates, reward prunes". Measures the graph colour of
post-training displacements D = H_stage - H_base on a frozen entity graph (OLMo-2-0425-1B chain, Olmo-3 7B pair).
Plan and prereg: `PREREG-2026-11-04.md` (sealed copy of DESIGN 2026-11-04; do not edit it, amend by addendum).

- Four-thread rule: one job at a time for 7B; at most 3 concurrent jobs; every compute command runs as
  `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 uv run ...`; the 7B extraction never runs beside a job above 6 GB RSS.
- **Step 0 of any run is the lineage gate** (`uv run python scripts/lineage_check.py <base_repo@rev> <stage_repo@rev> ...`, weights only): median relative distance < 0.5 and median cosine > 0.9 for base->first stage and base->each ingredient, else stop. Records live in `runs/lineage/` (copies in `data/lineage/`); `scripts/step3_extract.py` (mandatory `--gate PARENT:CHILD`), step 9 and step 10 refuse to run without a PASS record. No override flag. Regression case: OLMo-2-0425-1B `main` a1847dff must FAIL.
- Nothing is reported if the battery fails: any failed `tests/test_battery.py` assertion or any DESIGN 6.3 "BROKEN if" means no result is read or quoted (F5).
- Python via uv only; no GPU, no cloud; weights are downloaded one checkpoint at a time and deleted after extraction.
- Frozen inputs (entity CSVs, m, r, l*) are written once from the base and checksummed; never regenerated after a stage is opened.
