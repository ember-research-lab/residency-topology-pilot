# residency-topology-pilot

A small, closed pilot from Ember Research Lab (November 2026). It asked whether a post-training stage (supervised fine-tuning, preference optimization, reinforcement learning) changes how a language model's activations arrange a fixed set of entities, and whether the change is "structured" or not. It ran on a CPU, on open checkpoints only.

**Status: closed. The hypothesis is UNTESTED.** The instrument used to give a verdict failed its own control, and one run is void. Both are kept in the record on purpose. Nothing here should be cited as a result about any model.

## What happened, in order

1. **Design sealed before any run:** `PREREG-2026-11-04.md`. Amendments are appended below the sealed text, never edited into it.
2. **First run, VOID.** The 1B chain used the model hub's default checkpoint as the base. It is not the parent of the fine-tuned stages (weight cosine about 0.0003 to them and to its own listed ingredients). Every check that ran before measuring compared a model only with itself, so all of them passed. The void report is retained verbatim at the end of `REPORT-steps-0-10.md`.
3. **Fix:** a weights-only parent check (`src/rtp/lineage.py`, `scripts/lineage_check.py`) that runs first and makes every later step refuse to run without a pass. The bad checkpoint is kept as a test case that must fail.
4. **Re-run on the checked base**, then the 7B pair (base and its RL-trained descendant): `REPORT-steps-0-10.md`, top sections.
5. **Synthetic control.** A planted, purely smooth change with no rewiring in it was called a real effect 20 times out of 20 by the shuffle-based test, and the control word sets lit up like the real set. The test family is withdrawn as a verdict instrument: `ISSUES.md` (OPEN-1) and the final adjudication in the report.
6. **Toy** (`toy/`): a 360M model trained with several seeds, to see whether the spread across seeds can serve as the baseline instead. No falsifier fired; with 3 to 5 seeds nothing could be told apart from the seed spread; the RL arm never ran because its entry check stopped it. See `toy/PREREG-toy-2026-11-04.md` and `toy/REPORT-toy.md`.

## Read first

- `REPORT-steps-0-10.md`: the "Read this first" paragraph and the adjudication below it.
- `ISSUES.md`: what is still open.
- `CHANGELOG.md`: one line per step.

## Run

```
uv sync
uv run pytest
uv run python scripts/lineage_check.py <base_repo@rev> <stage_repo@rev>
```

Model downloads are large and are not part of the tests.

## Notes for readers

- The work was done with AI coding agents under human direction; commits carry co-author trailers. `ember-lab-admin` is the lab's account and stands for the human lead throughout.
- Before publication the history was rewritten once: a personal name was replaced by the lab account name, home-directory paths were made relative, and private session links were removed from commit messages. Commit order, authorship, dates and all other content are unchanged. Commit ids cited inside the reports are the pre-publication ids; `HISTORY-MAP.tsv` maps them to the published ids.
- The documents cite some internal lab notes (`DECISIONS-…`, `RECORD-…`, `LIT-…`, `PREMORTEM-…`) that are not in this repository. The documents also name some paths that are not in this repository: `runs/lineage/` (copies are in `data/lineage/`), `toy/frozen.json`, `~/ember-review/...` and `~/.claude/hooks/...`. Run records and some scripts contain the machine's model-cache path.
- `CLAUDE.md` is the instruction file the agents worked under.

## Licence

Code and text: MIT, see `LICENSE`. Third-party data under `data/entities/source/` keeps its own terms: see `data/entities/source/NOTICE.md`.
