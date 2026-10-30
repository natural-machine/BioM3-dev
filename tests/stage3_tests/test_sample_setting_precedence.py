"""Stage 3 sampling settings resolve command line, else config, else default.

``--alpha`` and ``--normalize_zc`` are both expressible in a config file and on
the command line, in either direction, and the command line wins. An
unrecognised value in either place is refused rather than read as a default or
as true. The run's ``normalize_zc`` is checked against what a weight set
records about the ProteoScribe weights being loaded.

The sampler is stopped at the point generation would start, which is after the
settings are resolved, logged and checked.
"""
import json
import os

import pytest
import torch

import biom3.Stage3.run_ProteoScribe_sample as sample

# Never opened: prepare_model is stubbed throughout, so this module needs no
# weight files. The path only has to be the one a weight set names, which is what
# the record check compares against.
MODEL = "weights/ProteoScribe/stub_proteoscribe.bin"
WARNING = "Z_C NORMALISATION DOES NOT MATCH THE WEIGHTS"
BASE_CONFIGS = [
    os.path.abspath("configs/inference/_base_runtime.json"),
    os.path.abspath("configs/stage3_training/models/_base_ProteoScribe_1block.json"),
]

Z_C = torch.tensor([[3.0, 4.0], [0.0, 0.5]])
Z_P = torch.tensor([[1.0, 0.0], [0.0, 2.0]])


class _Stop(Exception):
    """Raised in place of generating, to end the run early."""


@pytest.fixture
def run_sampler(tmp_path, monkeypatch):
    """Run the sampler up to generation; return (resolved config, run log)."""
    def _run(config_values, *extra, model=MODEL):
        config = tmp_path / "stage3.json"
        config.write_text(json.dumps({
            "_base_configs": BASE_CONFIGS,
            "device": "cpu",
            "num_replicas": 1,
            "batch_size_sample": 2,
            "text_emb_dim": 2,
            **config_values,
        }))
        embeddings = tmp_path / "emb.pt"
        torch.save({"z_c": Z_C.clone(), "z_p": Z_P.clone()}, embeddings)

        seen = {}

        def fake_generate(args, **kwargs):
            seen["config_args"] = args
            raise _Stop

        monkeypatch.setattr(sample, "prepare_model", lambda args, config_args: None)
        monkeypatch.setattr(sample, "batch_stage3_generate_sequences", fake_generate)
        outdir = tmp_path / "out"
        args = sample.parse_arguments([
            "-i", str(embeddings), "-c", str(config), "-m", model,
            "-o", str(outdir / "gen.pt"), "--device", "cpu", "--seed", "1", *extra,
        ])
        with pytest.raises(_Stop):
            sample.main(args)
        return seen["config_args"], (outdir / "run.log").read_text()

    return _run


@pytest.mark.parametrize("config_values, extra, expected", [
    ({}, [], 0.0),
    ({"alpha": 0.25}, [], 0.25),
    ({"alpha": 0.0}, [], 0.0),
    ({"alpha": 0.25}, ["--alpha", "0.75"], 0.75),
    ({"alpha": 0.25}, ["--alpha", "0"], 0.0),
    ({}, ["--alpha", "1"], 1.0),
    ({"alpha": None}, [], 0.0),
    ({"alpha": None}, ["--alpha", "0.5"], 0.5),
])
def test_alpha_resolves_and_is_logged(run_sampler, config_values, extra, expected):
    _, log = run_sampler(config_values, *extra)

    assert f"alpha={expected:.3f}" in log


@pytest.mark.parametrize("config_values, extra, expected", [
    ({}, [], False),
    ({"normalize_zc": True}, [], True),
    ({"normalize_zc": False}, [], False),
    ({"normalize_zc": True}, ["--normalize_zc", "False"], False),
    ({"normalize_zc": False}, ["--normalize_zc", "True"], True),
    ({}, ["--normalize_zc", "True"], True),
    ({"normalize_zc": None}, [], False),
    ({"normalize_zc": None}, ["--normalize_zc", "True"], True),
])
def test_normalize_zc_resolves(run_sampler, config_values, extra, expected):
    config_args, log = run_sampler(config_values, *extra)

    assert config_args.normalize_zc is expected
    # "mean length before" is logged only when the scaling is actually applied.
    assert ("mean length before" in log) is expected
    assert (f"Conditioning vector scaled to unit length: "
            f"{'yes' if expected else 'no'}") in log


def test_config_can_turn_normalisation_off_again(run_sampler):
    """The case a store_true flag could not express: config on, run off."""
    config_args, _ = run_sampler({"normalize_zc": True}, "--normalize_zc", "False")

    assert config_args.normalize_zc is False


@pytest.mark.parametrize("config_values, match", [
    ({"normalize_zc": "yes"}, "normalize_zc must be True or False"),
    ({"normalize_zc": 1}, "normalize_zc must be True or False"),
    ({"alpha": "most"}, "could not convert string to float"),
    ({"alpha": 1.5}, r"--alpha must be in \[0, 1\]"),
])
def test_unrecognised_config_value_raises(run_sampler, config_values, match):
    with pytest.raises(ValueError, match=match):
        run_sampler(config_values)


@pytest.mark.parametrize("extra", [
    ["--normalize_zc", "yes"],
    ["--normalize_zc", "1"],
    ["--alpha", "most"],
])
def test_unrecognised_command_line_value_is_refused(tmp_path, extra):
    with pytest.raises(SystemExit):
        sample.parse_arguments([
            "-i", "emb.pt", "-c", "configs/inference/stage3_ProteoScribe_sample.json",
            "-m", MODEL, "-o", str(tmp_path / "gen.pt"), *extra,
        ])


def _weight_set(tmp_path, **record):
    path = tmp_path / "weight_set.json"
    path.write_text(json.dumps({"proteoscribe_weights": MODEL, **record}))
    return str(path)


@pytest.mark.parametrize("recorded, extra", [
    (True, []),
    (True, ["--normalize_zc", "False"]),
    (False, ["--normalize_zc", "True"]),
])
def test_warns_when_normalisation_disagrees_with_the_weight_set(
        run_sampler, tmp_path, recorded, extra):
    weight_set = _weight_set(tmp_path, proteoscribe_trained_with_normalized_zc=recorded)

    _, log = run_sampler({}, "--weight_set", weight_set, *extra)

    assert WARNING in log


@pytest.mark.parametrize("record, extra", [
    ({"proteoscribe_trained_with_normalized_zc": True}, ["--normalize_zc", "True"]),
    ({"proteoscribe_trained_with_normalized_zc": False}, []),
    ({"proteoscribe_trained_with_normalized_zc": False}, ["--normalize_zc", "False"]),
    ({}, []),
    ({}, ["--normalize_zc", "True"]),
])
def test_quiet_when_they_agree_or_nothing_is_recorded(run_sampler, tmp_path, record, extra):
    weight_set = _weight_set(tmp_path, **record)

    _, log = run_sampler({}, "--weight_set", weight_set, *extra)

    assert WARNING not in log


def test_a_config_value_is_checked_too(run_sampler, tmp_path):
    """The check reports the setting in force, wherever it came from."""
    weight_set = _weight_set(tmp_path, proteoscribe_trained_with_normalized_zc=False)

    _, log = run_sampler({"normalize_zc": True}, "--weight_set", weight_set)

    assert WARNING in log
    assert "--normalize_zc on" in log
