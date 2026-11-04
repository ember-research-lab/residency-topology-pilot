"""Frozen entity sets: checksums verify, counts, no duplicates (DESIGN 6.3 step 2)."""
import csv, hashlib, pathlib

D = pathlib.Path(__file__).resolve().parents[1] / "data/entities"


def rows(n): return list(csv.DictReader(open(D / n, encoding="utf-8")))


def test_sha256sums():
    for line in (D / "SHA256SUMS").read_text().splitlines():
        h, f = line.split("  ")
        assert hashlib.sha256((D / f).read_bytes()).hexdigest() == h, f


def test_counts_and_uniqueness():
    g = rows("geo_v1.csv")
    assert len(g) == 500 and len({r["name"].casefold() for r in g}) == 500
    assert len(rows("control_nouns_v1.csv")) == 500 and len(rows("control_nonce_v1.csv")) == 500
    assert len(rows("figures_v1.csv")) == 300
    assert all(-90 <= float(r["lat"]) <= 90 and -180 <= float(r["lon"]) <= 180 for r in g)
    assert not ({r["text"] for r in rows("control_nouns_v1.csv")} & {r["name"].casefold() for r in g})
