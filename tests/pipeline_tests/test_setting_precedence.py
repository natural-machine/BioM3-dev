"""The pipeline passes on settings under the same precedence as the stages.

A setting a caller states on the pipeline's command line is the setting the
stage runs; a setting the caller leaves alone comes from that stage's config.

``--text_attention_mask`` and ``--normalize_zc`` are resolved here, because the
pipeline checks them against a weight set and so has to know their value, and
forwarded as that value. ``--text_padding`` is checked against nothing, so it is
forwarded only when given and the stage reads its own config otherwise.

What each stage then does with what it is handed is covered by
tests/stage1_tests/test_inference_setting_precedence.py and
tests/stage3_tests/test_sample_setting_precedence.py.
"""
import json
import os

import pytest
import torch

import biom3.Stage1.run_PenCL_inference as stage1_mod
from biom3.pipeline.embedding_pipeline import _build_stage3_argv
from biom3.pipeline.embedding_pipeline import main as pipeline_main
from biom3.pipeline.embedding_pipeline import parse_arguments as parse_pipeline_args
from biom3.Stage3.run_ProteoScribe_sample import parse_arguments as parse_sample_args

STAGE1_BASES = [
    os.path.abspath("configs/inference/_base_runtime.json"),
    os.path.abspath("configs/inference/models/_base_PenCL.json"),
]
STAGE3_BASES = [
    os.path.abspath("configs/inference/_base_runtime.json"),
    os.path.abspath("configs/stage3_training/models/_base_ProteoScribe_1block.json"),
]


def _stage1_config(tmp_path, **values):
    path = tmp_path / "stage1.json"
    path.write_text(json.dumps({
        "_base_configs": STAGE1_BASES,
        "model_type": "pfam",
        "sequence_keyword": "protein_sequence",
        "id_keyword": "primary_Accession",
        **values,
    }))
    return str(path)


def _stage3_config(tmp_path, **values):
    path = tmp_path / "stage3.json"
    path.write_text(json.dumps({
        "_base_configs": STAGE3_BASES,
        "device": "cpu",
        "num_replicas": 1,
        "batch_size_sample": 2,
        **values,
    }))
    return str(path)


@pytest.fixture
def stage1_args(tmp_path, monkeypatch):
    """Run the pipeline with Stage 1 stubbed; return the args Stage 1 was given."""
    def _run(pencl_config, *extra):
        seen = {}

        def fake_stage1(args, _setup_logging=True):
            seen["args"] = args
            torch.save({"z_t": torch.zeros(1, 2), "sequence": ["MK"], "acc_id": ["P1"]},
                       args.output_path)

        monkeypatch.setattr(stage1_mod, "main", fake_stage1)
        pipeline_main(parse_pipeline_args([
            "-i", "tests/_data/stage1_inputs/sample_text_seqs1.csv",
            "-o", str(tmp_path / "out"), "--prefix", "run",
            "--pencl_weights", "weights/PenCL/x.bin",
            "--pencl_config", pencl_config,
            "--device", "cpu", "--skip_facilitator", *extra,
        ]))
        return seen["args"]

    return _run


@pytest.mark.parametrize("config_values, extra, expected", [
    ({}, [], False),
    ({"text_attention_mask": True}, [], True),
    ({"text_attention_mask": False}, [], False),
    ({"text_attention_mask": True}, ["--text_attention_mask", "False"], False),
    ({"text_attention_mask": False}, ["--text_attention_mask", "True"], True),
    ({}, ["--text_attention_mask", "True"], True),
])
def test_mask_is_resolved_and_forwarded(stage1_args, tmp_path, config_values, extra, expected):
    args = stage1_args(_stage1_config(tmp_path, **config_values), *extra)

    assert args.text_attention_mask is expected


@pytest.mark.parametrize("extra, expected", [
    ([], None),
    (["--text_padding", "dynamic"], "dynamic"),
    (["--text_padding", "max_padding"], "max_padding"),
])
def test_padding_is_forwarded_only_when_given(stage1_args, tmp_path, extra, expected):
    """Unset, Stage 1 is handed nothing and reads its own config."""
    args = stage1_args(_stage1_config(tmp_path, text_padding="dynamic"), *extra)

    assert args.text_padding == expected


def test_the_weight_set_check_stays_with_the_pipeline(stage1_args, tmp_path):
    """The pipeline owns the check for its own runs, so the stage is not given
    the weight set as well and does not repeat the warning."""
    weight_set = tmp_path / "weight_set.json"
    weight_set.write_text(json.dumps({"pencl_weights": "weights/PenCL/x.bin"}))

    args = stage1_args(_stage1_config(tmp_path), "--weight_set", str(weight_set))

    assert args.weight_set is None


@pytest.mark.parametrize("config_values, extra, expected", [
    ({}, [], False),
    ({"normalize_zc": True}, [], True),
    ({"normalize_zc": False}, [], False),
    ({"normalize_zc": True}, ["--normalize_zc", "False"], False),
    ({"normalize_zc": False}, ["--normalize_zc", "True"], True),
    ({}, ["--normalize_zc", "True"], True),
])
def test_normalize_zc_is_resolved_and_forwarded(tmp_path, config_values, extra, expected):
    args = parse_pipeline_args([
        "-i", "in.csv", "-o", str(tmp_path), "--prefix", "run",
        "--pencl_weights", "p.bin",
        "--pencl_config", _stage1_config(tmp_path),
        "--skip_facilitator",
        "--generate", "--proteoscribe_weights", "s.bin",
        "--proteoscribe_config", _stage3_config(tmp_path, **config_values),
        *extra,
    ])

    argv = _build_stage3_argv(args, "emb.pt", "gen.pt")

    assert parse_sample_args(argv).normalize_zc is expected


@pytest.mark.parametrize("extra", [
    ["--text_attention_mask", "yes"],
    ["--normalize_zc", "1"],
    ["--text_padding", "longest"],
])
def test_unrecognised_command_line_value_is_refused(tmp_path, extra):
    with pytest.raises(SystemExit):
        parse_pipeline_args([
            "-i", "in.csv", "-o", str(tmp_path), "--prefix", "run",
            "--pencl_weights", "p.bin",
            "--pencl_config", _stage1_config(tmp_path),
            "--skip_facilitator", *extra,
        ])


@pytest.mark.parametrize("config_values, match", [
    ({"text_attention_mask": "yes"}, "text_attention_mask must be True or False"),
    ({"text_attention_mask": 1}, "text_attention_mask must be True or False"),
])
def test_unrecognised_stage1_config_value_raises(tmp_path, config_values, match):
    with pytest.raises(ValueError, match=match):
        parse_pipeline_args([
            "-i", "in.csv", "-o", str(tmp_path), "--prefix", "run",
            "--pencl_weights", "p.bin",
            "--pencl_config", _stage1_config(tmp_path, **config_values),
            "--skip_facilitator",
        ])
