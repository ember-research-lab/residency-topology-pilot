"""Lineage gate: child = parent + 1% noise passes; independent tensors fail; missing/FAIL records block downstream."""
import pathlib

import pytest
import torch
from safetensors.torch import save_file

from rtp import lineage


def _snap(tmp, name, tensors):
    d = pathlib.Path(tmp) / name; d.mkdir()
    save_file(tensors, str(d / "model.safetensors"))
    return str(d)


def _tensors(seed, shapes=((64, 32), (32, 64), (16,))):
    g = torch.Generator().manual_seed(seed)
    return {f"layer.{i}.w": torch.randn(s, generator=g) for i, s in enumerate(shapes)}


def test_child_passes_unrelated_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(lineage, "RECORD_DIR", tmp_path / "rec")
    P = _tensors(0); g = torch.Generator().manual_seed(9)
    child = {k: v + 0.01 * v.std() * torch.randn(v.shape, generator=g) for k, v in P.items()}
    a, b, c = _snap(tmp_path, "p", P), _snap(tmp_path, "c", child), _snap(tmp_path, "u", _tensors(1))
    ok, bad = lineage.pair_stats(a, b), lineage.pair_stats(a, c)
    assert ok["passed"] and ok["median_rel"] < 0.05 and ok["median_cos"] > 0.99
    assert not bad["passed"] and bad["median_cos"] < 0.5
    lineage.write_record("p" * 40, "c" * 40, ok); lineage.write_record("p" * 40, "u" * 40, bad)
    assert lineage.require_pass("p" * 40, "c" * 40)["passed"]
    with pytest.raises(lineage.LineageError):
        lineage.require_pass("p" * 40, "u" * 40)           # FAIL record blocks
    with pytest.raises(lineage.LineageError):
        lineage.require_pass("p" * 40, "z" * 40)           # missing record blocks


def test_key_mismatch_fails(tmp_path):
    P = _tensors(0); Q = dict(P); Q["extra"] = torch.zeros(2)
    st = lineage.pair_stats(_snap(tmp_path, "a", P), _snap(tmp_path, "b", Q))
    assert not st["passed"]
