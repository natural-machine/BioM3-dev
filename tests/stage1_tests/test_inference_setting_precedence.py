"""Stage 1 inference settings resolve command line, else config, else default.

``--text_padding`` and ``--text_attention_mask`` are both expressible in a
config file and on the command line, in either direction, and the command line
wins. An unrecognised value in either place is refused rather than read as a
default or as true. The resolved padding mode is logged, and the mask is checked
against what a weight set records about the weights being loaded.

Most of these stop Stage 1 at the point the model would load, which is after the
settings are resolved, logged and checked, and need no weight files at all.
``prepare_model`` receives the namespace the dataset is built from, so what it is
handed is what the collate function will pad with (test_inference_attention_mask.py
covers what the inference collate does with each mode). The last test here runs
the whole embedding path to show the resolved mode is the one the captions are
padded to, so it needs the tokenizers and skips without them.
"""
import json
import os

import pytest
import torch
from torch import nn
from tests.conftest import check_downloads

import biom3.Stage1.run_PenCL_inference as stage1

# Only the end-to-end test at the bottom of this module reads any weight file,
# and it skips when these are missing.
REQUIRED_DOWNLOADS = [
    "weights/LLMs/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "weights/LLMs/esm2_t33_650M_UR50D.pt",
]

# Never opened: prepare_model is stubbed throughout. The path only has to be the
# one a weight set names, which is what the record check compares against.
MODEL = "weights/PenCL/stub_pencl.bin"
WARNING = "CAPTION ATTENTION MASK DOES NOT MATCH THE WEIGHTS"
# Absolute: a config written to tmp_path resolves its bases against its own
# directory, not the repo root.
BASE_CONFIGS = [
    os.path.abspath("configs/inference/_base_runtime.json"),
    os.path.abspath("configs/inference/models/_base_PenCL.json"),
]


class _Stop(Exception):
    """Raised in place of loading the model, to end the run early."""


@pytest.fixture
def run_stage1(tmp_path, monkeypatch):
    """Run Stage 1 up to the model load; return (resolved config, run log)."""
    def _run(config_values, *extra, model=MODEL):
        config = tmp_path / "stage1.json"
        config.write_text(json.dumps({
            "_base_configs": BASE_CONFIGS,
            "model_type": "pfam",
            "sequence_keyword": "protein_sequence",
            "id_keyword": "primary_Accession",
            **config_values,
        }))

        seen = {}

        def fake_prepare_model(config_args, model_path, device, load_from_checkpoint):
            seen["config_args"] = config_args
            raise _Stop

        monkeypatch.setattr(stage1, "prepare_model", fake_prepare_model)
        outdir = tmp_path / "out"
        args = stage1.parse_arguments([
            "-i", "None", "-c", str(config), "-m", model,
            "-o", str(outdir / "emb.pt"), "--device", "cpu", *extra,
        ])
        with pytest.raises(_Stop):
            stage1.main(args)
        return seen["config_args"], (outdir / "run.log").read_text()

    return _run


@pytest.mark.parametrize("config_values, extra, expected", [
    ({}, [], "max_padding"),
    ({"text_padding": "dynamic"}, [], "dynamic"),
    ({"text_padding": "max_padding"}, [], "max_padding"),
    ({"text_padding": "dynamic"}, ["--text_padding", "max_padding"], "max_padding"),
    ({"text_padding": "max_padding"}, ["--text_padding", "dynamic"], "dynamic"),
    ({}, ["--text_padding", "dynamic"], "dynamic"),
    ({"text_padding": None}, [], "max_padding"),
    ({"text_padding": None}, ["--text_padding", "dynamic"], "dynamic"),
])
def test_padding_mode_resolves_and_is_logged(run_stage1, config_values, extra, expected):
    config_args, log = run_stage1(config_values, *extra)

    assert config_args.text_padding == expected
    assert f"caption padding: {expected}" in log


@pytest.mark.parametrize("config_values, extra, expected", [
    ({}, [], False),
    ({"text_attention_mask": True}, [], True),
    ({"text_attention_mask": False}, [], False),
    ({"text_attention_mask": True}, ["--text_attention_mask", "False"], False),
    ({"text_attention_mask": False}, ["--text_attention_mask", "True"], True),
    ({}, ["--text_attention_mask", "True"], True),
    ({"text_attention_mask": None}, [], False),
    ({"text_attention_mask": None}, ["--text_attention_mask", "True"], True),
])
def test_attention_mask_resolves_and_is_logged(run_stage1, config_values, extra, expected):
    _, log = run_stage1(config_values, *extra)

    assert f"caption attention mask: {'on' if expected else 'off'}" in log


def test_config_can_turn_the_mask_off_again(run_stage1):
    """The case a store_true flag could not express: config on, run off."""
    _, log = run_stage1({"text_attention_mask": True}, "--text_attention_mask", "False")

    assert "caption attention mask: off" in log
    assert "caption attention mask: on" not in log


@pytest.mark.parametrize("config_values, match", [
    ({"text_padding": "longest"}, "text_padding must be one of"),
    ({"text_attention_mask": "yes"}, "text_attention_mask must be True or False"),
    ({"text_attention_mask": 1}, "text_attention_mask must be True or False"),
])
def test_unrecognised_config_value_raises(run_stage1, config_values, match):
    with pytest.raises(ValueError, match=match):
        run_stage1(config_values)


@pytest.mark.parametrize("extra", [
    ["--text_padding", "longest"],
    ["--text_attention_mask", "yes"],
    ["--text_attention_mask", "1"],
])
def test_unrecognised_command_line_value_is_refused(tmp_path, extra):
    with pytest.raises(SystemExit):
        stage1.parse_arguments([
            "-i", "None", "-c", "configs/inference/stage1_PenCL.json",
            "-m", MODEL, "-o", str(tmp_path / "emb.pt"), *extra,
        ])


def _weight_set(tmp_path, **record):
    path = tmp_path / "weight_set.json"
    path.write_text(json.dumps({"pencl_weights": MODEL, **record}))
    return str(path)


@pytest.mark.parametrize("recorded, extra", [
    (True, []),
    (True, ["--text_attention_mask", "False"]),
    (False, ["--text_attention_mask", "True"]),
])
def test_warns_when_the_mask_disagrees_with_the_weight_set(run_stage1, tmp_path, recorded, extra):
    weight_set = _weight_set(tmp_path, pencl_trained_with_text_attention_mask=recorded)

    _, log = run_stage1({}, "--weight_set", weight_set, *extra)

    assert WARNING in log


@pytest.mark.parametrize("record, extra", [
    ({"pencl_trained_with_text_attention_mask": True}, ["--text_attention_mask", "True"]),
    ({"pencl_trained_with_text_attention_mask": False}, []),
    ({"pencl_trained_with_text_attention_mask": False}, ["--text_attention_mask", "False"]),
    ({}, []),
    ({}, ["--text_attention_mask", "True"]),
])
def test_quiet_when_they_agree_or_nothing_is_recorded(run_stage1, tmp_path, record, extra):
    weight_set = _weight_set(tmp_path, **record)

    _, log = run_stage1({}, "--weight_set", weight_set, *extra)

    assert WARNING not in log


def test_a_config_value_is_checked_too(run_stage1, tmp_path):
    """The check reports the setting in force, wherever it came from."""
    weight_set = _weight_set(tmp_path, pencl_trained_with_text_attention_mask=False)

    _, log = run_stage1({"text_attention_mask": True}, "--weight_set", weight_set)

    assert WARNING in log
    assert "--text_attention_mask on" in log


def test_says_nothing_about_other_pencl_weights(run_stage1, tmp_path):
    weight_set = _weight_set(tmp_path, pencl_trained_with_text_attention_mask=True)

    _, log = run_stage1({}, "--weight_set", weight_set, model="weights/PenCL/other.bin")

    assert WARNING not in log
    assert "not the weights in use" in log


class _RecordingModel(nn.Module):
    """Stand-in for PenCL that keeps the width of the caption tensor it saw."""

    def __init__(self):
        super().__init__()
        self.widths = []

    def forward(self, x_t, x_p, compute_masked_logits=False, x_t_mask=None):
        self.widths.append(x_t.shape[-1])
        return {"text_joint_latent": torch.zeros(x_t.size(0), 4),
                "seq_joint_latent": torch.zeros(x_t.size(0), 4)}


@pytest.mark.slow
@pytest.mark.parametrize("config_values, extra, dynamic", [
    ({"text_padding": "dynamic"}, [], True),
    ({}, [], False),
    ({"text_padding": "dynamic"}, ["--text_padding", "max_padding"], False),
    ({}, ["--text_padding", "dynamic"], True),
])
def test_the_resolved_mode_is_what_the_captions_are_padded_to(
        tmp_path, monkeypatch, config_values, extra, dynamic):
    """End to end: a config that says 'dynamic' embeds with dynamic padding.

    With 'max_padding' the caption tensor is text_max_length wide; with
    'dynamic' it is only as wide as the batch's longest caption. This is the one
    test here that builds the real dataset, so it needs the tokenizers the
    Stage 1 base config names.
    """
    issues, skip_reason = check_downloads(REQUIRED_DOWNLOADS)
    if issues:
        pytest.skip(reason=skip_reason)

    config = tmp_path / "stage1.json"
    config.write_text(json.dumps({
        "_base_configs": BASE_CONFIGS,
        "model_type": "pfam",
        "sequence_keyword": "protein_sequence",
        "id_keyword": "primary_Accession",
        **config_values,
    }))
    csv = tmp_path / "proteins.csv"
    csv.write_text(
        "primary_Accession,protein_sequence,[final]text_caption\n"
        "P1,MKTAYIAKQR,PROTEIN NAME: Serine protease.\n"
        "P2,MVKMIVGLGN,PROTEIN NAME: Kinase.\n"
    )

    model = _RecordingModel()
    monkeypatch.setattr(
        stage1, "prepare_model",
        lambda config_args, model_path, device, load_from_checkpoint: model)
    outdir = tmp_path / "out"
    stage1.main(stage1.parse_arguments([
        "-i", str(csv), "-c", str(config), "-m", MODEL,
        "-o", str(outdir / "emb.pt"), "--device", "cpu", *extra,
    ]))

    log = (outdir / "run.log").read_text()
    expected = "dynamic" if dynamic else "max_padding"
    assert f"caption padding: {expected}" in log
    assert model.widths, "the model was never called"
    if dynamic:
        assert all(w < 512 for w in model.widths), model.widths
    else:
        assert all(w == 512 for w in model.widths), model.widths
