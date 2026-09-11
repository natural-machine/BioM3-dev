"""With pre-written shards, Lightning must not build its all-rank Gloo barrier.

Lightning wraps any overridden prepare_data in _InfiniteBarrier, a Gloo group
over every rank. Gloo's full-mesh setup failed at 3,072 ranks (run2a, job
8816887: DistNetworkError: Connection reset by peer). When the shards are
pre-written, Pfam_DataModule stops exposing prepare_data, so Lightning's own
gate, is_overridden(), is False and the group is never created.

These tests drive Lightning's real _DataConnector.prepare_data, with
_InfiniteBarrier swapped for a recorder, so they check the actual gate rather
than a reimplementation of it.
"""
import types

import importlib

import pandas as pd
import pytest

from biom3.Stage1 import preprocess as pp
from biom3.Stage1.preprocess import Pfam_DataModule, write_pfam_splits

# preprocess imports `lightning` on XPU and `pytorch_lightning` elsewhere; test
# against the same package (the prepare_data gate is identical in both).
_PKG = "lightning.pytorch" if pp.LightningDataModule.__module__.startswith("lightning.") else "pytorch_lightning"
dc = importlib.import_module(f"{_PKG}.trainer.connectors.data_connector")
is_overridden = importlib.import_module(f"{_PKG}.utilities.model_helpers").is_overridden


def _pfam(n=60):
    return pd.DataFrame({"id": [f"A{i}" for i in range(n)], "pfam_label": [f"PF{i % 4:05d}" for i in range(n)],
                         "sequence": ["MKT"] * n, "[final]text_caption": ["cap"] * n})


def _args(src, splits, nodes=2, devices=3):
    return types.SimpleNamespace(pfam_data_path=src, pfam_splits_dir=splits, dataset_type="pfam",
                                 num_nodes=nodes, devices_per_node=devices)


@pytest.fixture
def src(tmp_path):
    p = tmp_path / "pfam.csv"
    _pfam().to_csv(p, index=False)
    return str(p)


class _Recorder:
    entered = 0
    def __enter__(self):
        type(self).entered += 1
        return self
    def __exit__(self, *exc):
        return False


def _run_connector(dm, monkeypatch):
    _Recorder.entered = 0
    monkeypatch.setattr(dc, "_InfiniteBarrier", _Recorder)
    # local_rank 1: the hook itself is not called; only the barrier decision runs
    trainer = types.SimpleNamespace(local_rank=1, node_rank=0, datamodule=dm, lightning_module=None)
    conn = dc._DataConnector.__new__(dc._DataConnector)
    conn.trainer = trainer
    conn.prepare_data()
    return _Recorder.entered


def test_prewritten_shards_skip_the_gloo_barrier(tmp_path, src, monkeypatch):
    d = str(tmp_path / "splits")
    write_pfam_splits(_pfam(), d, 6, src)                   # W = 2 nodes x 3 devices
    dm = Pfam_DataModule(_args(src, d))
    assert dm.pfam_splits_prewritten
    assert not is_overridden("prepare_data", dm)
    assert _run_connector(dm, monkeypatch) == 0


def test_without_prewritten_shards_lightning_still_uses_the_barrier(tmp_path, src, monkeypatch):
    dm = Pfam_DataModule(_args(src, str(tmp_path / "empty")))
    assert not dm.pfam_splits_prewritten
    assert is_overridden("prepare_data", dm)
    assert _run_connector(dm, monkeypatch) == 1


def test_manifest_for_another_world_size_raises(tmp_path, src):
    d = str(tmp_path / "splits")
    write_pfam_splits(_pfam(), d, 6, src)
    with pytest.raises(ValueError, match="num_shards"):
        Pfam_DataModule(_args(src, d, nodes=4, devices=3))  # W = 12, manifest says 6


def test_no_configured_world_size_keeps_old_behaviour(tmp_path, src):
    d = str(tmp_path / "splits")
    write_pfam_splits(_pfam(), d, 6, src)
    args = types.SimpleNamespace(pfam_data_path=src, pfam_splits_dir=d, dataset_type="pfam")
    dm = Pfam_DataModule(args)
    assert not dm.pfam_splits_prewritten and is_overridden("prepare_data", dm)
