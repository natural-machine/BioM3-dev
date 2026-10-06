"""Tests for --skip_facilitator in biom3_embedding_pipeline.

The option runs Stage 1 alone, for weight sets that have no Facilitator. The
stage entry points are stubbed, so these run under `pytest --quick` without
downloaded weights.
"""

import json
import os

import pytest
import torch

import biom3.data_prep.compile_stage2_data_to_hdf5 as compile_mod
import biom3.Stage1.run_PenCL_inference as stage1_mod
import biom3.Stage2.run_Facilitator_sample as stage2_mod
from biom3.pipeline.embedding_pipeline import main, parse_arguments

PENCL_CONFIG = "configs/inference/stage1_PenCL.json"
FACILITATOR_CONFIG = "configs/inference/stage2_Facilitator.json"


@pytest.fixture
def pencl_only_weight_set(tmp_path):
    path = tmp_path / "pencl_only.json"
    path.write_text(json.dumps({"pencl_weights": "weights/PenCL/only_pencl.bin"}))
    return str(path)


def _argv(tmp_path, weight_set, *extra):
    return [
        "-i", "tests/_data/stage1_inputs/sample_text_seqs1.csv",
        "-o", str(tmp_path / "out"),
        "--prefix", "run",
        "--weight_set", weight_set,
        "--pencl_config", PENCL_CONFIG,
        "--device", "cpu",
        *extra,
    ]


def test_parses_with_a_pencl_only_weight_set(tmp_path, pencl_only_weight_set):
    args = parse_arguments(_argv(tmp_path, pencl_only_weight_set, "--skip_facilitator"))

    assert args.skip_facilitator
    assert args.pencl_weights == "weights/PenCL/only_pencl.bin"
    assert args.facilitator_weights is None
    assert args.facilitator_config is None


def test_facilitator_still_required_without_the_flag(tmp_path, pencl_only_weight_set):
    with pytest.raises(SystemExit):
        parse_arguments(_argv(tmp_path, pencl_only_weight_set))
    with pytest.raises(SystemExit):
        parse_arguments(_argv(tmp_path, pencl_only_weight_set,
                              "--facilitator_config", FACILITATOR_CONFIG))


def test_rejected_together_with_generate(tmp_path, pencl_only_weight_set):
    with pytest.raises(SystemExit):
        parse_arguments(_argv(tmp_path, pencl_only_weight_set,
                              "--skip_facilitator", "--generate"))


def test_runs_stage1_only(tmp_path, pencl_only_weight_set, monkeypatch):
    stage1_calls = []

    def fake_stage1(args, _setup_logging=True):
        stage1_calls.append(args)
        torch.save({"z_t": torch.zeros(1, 2), "z_p": torch.zeros(1, 2)}, args.output_path)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("Stage 2 and the HDF5 compile must not run")

    monkeypatch.setattr(stage1_mod, "main", fake_stage1)
    monkeypatch.setattr(stage2_mod, "main", must_not_run)
    monkeypatch.setattr(compile_mod, "main", must_not_run)

    main(parse_arguments(_argv(tmp_path, pencl_only_weight_set, "--skip_facilitator")))

    out = tmp_path / "out"
    assert len(stage1_calls) == 1
    assert stage1_calls[0].model_path == "weights/PenCL/only_pencl.bin"
    assert sorted(os.listdir(out)) == [
        "run.PenCL_emb.pt", "run.build_manifest.json", "run.run.log",
    ]

    manifest = json.loads((out / "run.build_manifest.json").read_text())
    assert list(manifest["outputs"]) == ["pencl_output"]
    assert list(manifest["config_contents"]) == ["pencl"]
    assert not any("facilitator" in key for key in manifest["resolved_paths"])
