"""entities - build and freeze the entity and control sets (DESIGN 2.1); tokenization asserts across stages.

Sources: Gurnee & Tegmark world-places and historical-figures tables (github wesg52/world-models, committed under
data/entities/source/), WordNet 3.0 noun lemmas (PyPI wn==0.0.23, scripts/dump_wordnet_nouns.py), wordfreq (offline).
Tokenizers: tokenizer files only. Within a family every prompt must tokenize identically across checkpoints.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import random
import re
from dataclasses import dataclass

from tokenizers import Tokenizer
from wordfreq import zipf_frequency

FAMILY_OLMO2 = ["OLMo-2-0425-1B", "OLMo-2-0425-1B-SFT", "OLMo-2-0425-1B-DPO", "OLMo-2-0425-1B-RLVR1", "OLMo-2-0425-1B-Instruct"]
FAMILY_OLMO3 = ["Olmo-3-1025-7B", "Olmo-3-7B-RL-Zero-Math"]

TEMPLATES_GEO = ["{X}", "The city of {X}", "I recently travelled to {X}", "Here is a place: {X}", "Weather report for {X}",
                 "She grew up in {X}", "A photo taken in {X}", "The map showed {X}"]
TEMPLATES_CTRL = ["{X}", "The word {X}", "I wrote down the word {X}", "Here is a word: {X}", "A dictionary entry for {X}",
                  "She said the word {X}", "A note that says {X}", "The label read {X}"]
MAX_TOK = 4
# (stratum, quota): coarse lat/lon boxes (approximate; used only to spread the sample, not as ground truth)
STRATA = [("north_america", 100), ("south_america", 60), ("europe", 100), ("africa", 80), ("asia", 120), ("oceania", 40)]


class EntityError(RuntimeError):
    """Raised when the pilot's entity assertions fail."""


def continent(lat: float, lon: float) -> str:
    if (lon >= 110 and lat <= -10) or (lon >= 165 and lat <= 0) or (lon <= -150 and lat <= 0):
        return "oceania"
    if -170 <= lon <= -30:
        return "north_america" if lat >= 12 else "south_america"
    if -25 <= lon <= 40 and lat >= 36:
        return "europe"
    if -20 <= lon <= (34 if lat > 12 else 52) and lat < 36:
        return "africa"
    return "asia"


def load_tokenizers(tok_dir: pathlib.Path) -> dict[str, Tokenizer]:
    return {n: Tokenizer.from_file(str(tok_dir / n / "tokenizer.json")) for n in FAMILY_OLMO2 + FAMILY_OLMO3}


def ids(tok: Tokenizer, text: str, special: bool = True) -> list[int]:
    return tok.encode(text, add_special_tokens=special).ids


def prompts_for(name: str, templates: list[str]) -> list[str]:
    return [t.replace("{X}", name) for t in templates]


def check_name(name: str, toks: dict[str, Tokenizer], templates: list[str]) -> str | None:
    """Return None if the name survives, else the reason string."""
    for fam in (FAMILY_OLMO2, FAMILY_OLMO3):
        for p in prompts_for(name, templates):
            ref = ids(toks[fam[0]], p)
            if any(ids(toks[f], p) != ref for f in fam[1:]):
                return "stage_tokenization_differs"
    for t in toks.values():
        if len(ids(t, " " + name, False)) > MAX_TOK or len(ids(t, name, False)) > MAX_TOK:
            return "gt_4_tokens"
    return None


def n_tok(toks: dict[str, Tokenizer], name: str) -> int:
    return len(ids(toks[FAMILY_OLMO2[0]], " " + name, False))


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ADMIN = re.compile(r"\b(Province|District|subdivision|County|Prefecture|Region|Governorate|Municipality|Department|Greater|State of|Division|Metropolitan|Area|Urban|Territory|Parish|Canton|Commune|Capital|City of)\b", re.I)


def _clean(name: str) -> bool:
    if ADMIN.search(name):
        return False   # administrative units, not cities/places
    return bool(re.fullmatch(r"[^\W\d_]+(?:[ '\-.][^\W\d_]+)*", name)) and name == name.strip()


@dataclass
class Report:
    stats: dict


def build_geography(src: pathlib.Path, toks: dict[str, Tokenizer], rep: dict) -> list[dict]:
    rows = list(csv.DictReader(open(src / "world_place.csv", encoding="utf-8")))
    cnt: dict[str, int] = {}
    for r in rows:
        cnt[r["name"].casefold()] = cnt.get(r["name"].casefold(), 0) + 1
    rep["geo_table_rows"] = len(rows)
    pool = [r for r in rows if r["entity_type"] == "populated_place" and r["entity_subtype"] == "PopulatedPlace" and r["population"]]
    rep["geo_populated_place_with_population"] = len(pool)
    reasons = {"duplicate_name_in_table": 0, "unclean_name": 0, "stage_tokenization_differs": 0, "gt_4_tokens": 0}
    by: dict[str, list[dict]] = {s: [] for s, _ in STRATA}
    for r in sorted(pool, key=lambda r: -float(r["population"])):
        lat, lon = float(r["latitude"]), float(r["longitude"])
        by[continent(lat, lon)].append(dict(name=r["name"], country=r["country"], lat=lat, lon=lon, population=int(float(r["population"])), continent=continent(lat, lon)))
    out, seen = [], set()
    for s, q in STRATA:
        got = 0
        for e in by[s]:
            if got == q:
                break
            nm = e["name"]
            if cnt[nm.casefold()] > 1:
                reasons["duplicate_name_in_table"] += 1; continue
            if not _clean(nm) or nm.casefold() in seen:
                reasons["unclean_name"] += 1; continue
            why = check_name(nm, toks, TEMPLATES_GEO)
            if why:
                reasons[why] += 1; continue
            seen.add(nm.casefold()); got += 1
            out.append({**e, "n_tok_space": n_tok(toks, nm), "n_tok_bare": len(ids(toks[FAMILY_OLMO2[0]], nm, False))})
        rep[f"geo_stratum_{s}"] = dict(quota=q, got=got, candidates=len(by[s]))
    rep["geo_rejected_while_filling_quota"] = reasons
    rep["geo_survivors"] = len(out)
    return out


def build_noun_controls(geo: list[dict], nouns_file: pathlib.Path, toks: dict[str, Tokenizer], rep: dict) -> list[dict]:
    geo_lower = {g["name"].casefold() for g in geo}
    cands = []
    for w in nouns_file.read_text().split():
        z = zipf_frequency(w, "en")
        if z >= 1.0 and len(w) >= 3 and w not in geo_lower and check_name(w, toks, TEMPLATES_CTRL) is None:
            cands.append((w, z, n_tok(toks, w)))
    rep["noun_candidates"] = len(cands)
    used, out = set(), []
    order = sorted(range(len(geo)), key=lambda i: -zipf_frequency(geo[i]["name"], "en"))
    for i in order:
        g = geo[i]; zg = zipf_frequency(g["name"], "en")
        best = min((c for c in cands if c[0] not in used and c[2] == g["n_tok_space"]), key=lambda c: abs(c[1] - zg), default=None)
        if best is None:
            continue
        used.add(best[0])
        out.append(dict(text=best[0], zipf=round(best[1], 3), n_tok_space=best[2], matched_geo=g["name"], zipf_geo=round(zg, 3), delta_zipf=round(best[1] - zg, 3)))
    out.sort(key=lambda r: r["matched_geo"])
    rep["noun_controls"] = len(out)
    rep["noun_unmatched_geo"] = len(geo) - len(out)
    if out:
        d = [abs(r["delta_zipf"]) for r in out]
        rep["noun_mean_abs_dzipf"], rep["noun_max_abs_dzipf"] = round(sum(d) / len(d), 3), max(d)
    return out


def build_nonce(geo: list[dict], nouns: set[str], toks: dict[str, Tokenizer], rep: dict, seed: int = 20261104) -> list[dict]:
    rng = random.Random(seed)
    cons, vow = "bdfgklmnprstvz", "aeiou"
    want: dict[int, int] = {}
    for g in geo:
        want[g["n_tok_space"]] = want.get(g["n_tok_space"], 0) + 1
    out, seen, tries = [], set(), 0
    while any(want.values()) and tries < 2_000_000:
        tries += 1
        w = "".join(rng.choice(cons) + rng.choice(vow) for _ in range(rng.randint(2, 5))) + (rng.choice(cons) if rng.random() < .4 else "")
        if w in seen or w in nouns or zipf_frequency(w, "en") > 0:
            continue
        seen.add(w)
        k = n_tok(toks, w)
        if want.get(k, 0) == 0 or check_name(w, toks, TEMPLATES_CTRL):
            continue
        want[k] -= 1
        out.append(dict(text=w, n_tok_space=k))
    rep["nonce_controls"] = len(out)
    rep["nonce_unfilled_token_counts"] = {k: v for k, v in want.items() if v}
    return out


def build_figures(src: pathlib.Path, toks: dict[str, Tokenizer], rep: dict, n: int = 300) -> list[dict]:
    rows = list(csv.DictReader(open(src / "historical_figure.csv", encoding="utf-8")))
    cnt: dict[str, int] = {}
    for r in rows:
        cnt[r["name"].casefold()] = cnt.get(r["name"].casefold(), 0) + 1
    ok = []
    for r in rows:
        if not r["death_year"] or " " not in r["name"] or cnt[r["name"].casefold()] > 1 or not _clean(r["name"]):
            continue
        y = float(r["death_year"])
        if not -1000 <= y <= 2025:
            continue
        ok.append((zipf_frequency(r["name"], "en"), r["name"], int(y), r["country"], r["occupation"]))
    rep["fig_rows"], rep["fig_pool_unique_clean"] = len(rows), len(ok)
    ok.sort(key=lambda t: -t[0])
    top = ok[:4000]
    years = sorted(t[2] for t in top)
    edges = [years[int(len(years) * i / 10)] for i in range(10)] + [10**9]   # death-year deciles of the top-4000-by-zipf pool
    out, rej = [], {"stage_tokenization_differs": 0, "gt_4_tokens": 0}
    for b in range(10):
        got = 0
        for z, nm, y, c, o in top:
            if got == n // 10: break
            if not edges[b] <= y < edges[b + 1]: continue
            why = check_name(nm, toks, TEMPLATES_GEO)
            if why: rej[why] += 1; continue
            got += 1; out.append(dict(name=nm, death_year=y, country=c, occupation=o, zipf=round(z, 3), n_tok_space=n_tok(toks, nm)))
    rep["fig_rejected_while_filling"], rep["fig_survivors"] = rej, len(out)
    return out


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow(r)


def main(tok_dir: str, repo: str) -> None:
    repo_p = pathlib.Path(repo); src = repo_p / "data/entities/source"; outd = repo_p / "data/entities"
    toks = load_tokenizers(pathlib.Path(tok_dir))
    rep: dict = {}
    geo = build_geography(src, toks, rep)
    nouns = build_noun_controls(geo, src / "wordnet_nouns.txt", toks, rep)
    nonce = build_nonce(geo, set((src / "wordnet_nouns.txt").read_text().split()), toks, rep)
    figs = build_figures(src, toks, rep)
    for name, rows in (("geo_v1.csv", geo), ("control_nouns_v1.csv", nouns), ("control_nonce_v1.csv", nonce), ("figures_v1.csv", figs)):
        write_csv(outd / name, rows)
    (outd / "survivors.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    files = ["geo_v1.csv", "control_nouns_v1.csv", "control_nonce_v1.csv", "figures_v1.csv", "survivors.json", "tokenizer_files.txt",
             "source/world_place.csv", "source/historical_figure.csv", "source/wordnet_nouns.txt"]
    (outd / "SHA256SUMS").write_text("".join(f"{sha256(outd / f)}  {f}\n" for f in files))
    print(json.dumps(rep, indent=1, sort_keys=True))
    if rep["geo_survivors"] < 500:
        raise EntityError(f"BROKEN: only {rep['geo_survivors']} geography survivors (< 500); reported, not padded")


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
