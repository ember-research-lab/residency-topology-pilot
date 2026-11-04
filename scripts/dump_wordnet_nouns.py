"""One-off: dump single-word lowercase WordNet 3.0 noun lemmas (wn==0.0.23 from PyPI) to data/entities/source/wordnet_nouns.txt.
Run: uv run --no-project --with wn==0.0.23 python scripts/dump_wordnet_nouns.py   (wn is not a project dependency)
"""
from wn import WordNet

w = WordNet()
lem = sorted({l.name() for s in w.all_synsets("n") for l in s.lemmas() if l.name().isalpha() and l.name().islower()})
open("data/entities/source/wordnet_nouns.txt", "w").write("\n".join(lem) + "\n")
print(len(lem))
