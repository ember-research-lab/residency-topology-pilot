# ISSUES (pilot backlog; the repo is local-only, so this file stands in for GitHub issues)

## OPEN-1: no calibrated null for displacement colour at ||D||/||H|| of about 0.01 (needs-decision: residency compute)

- **Symptom.** The N4' (row-permutation) colour test calls a purely smooth synthetic displacement red in 20/20 seeds (geography, C1, C2); the KRR-smoothed readout and the corrected Freedman-Lane null call it blue 20/20 on geography and C1; rewiring injected at share 0.10 is recovered only 15% of the time under the corrected null (data/step15_synthetic_null.json). Real RL-Zero colours (red raw, blue smoothed on geography) lie inside what smooth-only displacement produces.
- **Root cause (T1, synthetic; T3 for real models).** Permuting rows of D destroys the D-to-H_b link, so the null lacks the smooth part the real D has; any smoother then removes low-band energy the permutation null retains, flipping the bias.
- **Acceptance check.** A null (empirical, from seed replicates of the same recipe) under which the s=0 synthetic smooth-only displacement reads white in >= 95% of seeds on geography and on C1, and s=0.10 rewiring reads red in >= 80% of seeds: `uv run python scripts/step15_synthetic_null.py <group>` then `scripts/step15_merge.py`, read data/step15_synthetic_null.json.
- **Class guard.** The step-15 synthetic control must run (and pass) before any colour verdict is read; recorded as a BROKEN-if row in CLAUDE.md.
- **Path.** Seed replicates of the post-training recipe (open-instruct on 1B/7B/13B, >= 3 seeds): the residency compute ask.
