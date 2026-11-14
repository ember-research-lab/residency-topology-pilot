# Third-party data in this directory

The files here are not covered by the repository's MIT licence for the lab's own code and text. They are redistributed under their own terms.

| File | Source | Terms |
|---|---|---|
| `world_place.csv`, `historical_figure.csv` | Entity tables from W. Gurnee and M. Tegmark, "Language Models Represent Space and Time" (arXiv:2310.02207), repository `github.com/wesg52/world-models`, path `data/entity_datasets/`, as fetched on 2026-11-03/04. | MIT, Copyright (c) 2023 Wes Gurnee. Full text in `LICENSE-world-models`. |
| `wordnet_nouns.txt` | Single-word lowercase noun lemmas dumped from WordNet 3.0 with `scripts/dump_wordnet_nouns.py` (PyPI `wn==0.0.23`). | WordNet 3.0 Copyright 2006 by Princeton University. All rights reserved. Licence: https://wordnet.princeton.edu/license-and-commercial-use |

Everything derived from these files elsewhere in `data/entities/` (frozen entity sets, controls, token tables) is built by `src/rtp/entities.py`.
