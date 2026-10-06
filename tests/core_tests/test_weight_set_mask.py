"""Weight sets: the record of how PenCL was trained, and filling renamed args."""

import json
from argparse import Namespace

import pytest

from biom3.core.weight_sets import (
    check_text_attention_mask,
    merge_weight_set,
    pencl_trained_with_mask,
)

WARNING = "CAPTION ATTENTION MASK DOES NOT MATCH THE WEIGHTS"


@pytest.fixture
def weight_set(tmp_path):
    def _write(**extra):
        path = tmp_path / "weight_set.json"
        path.write_text(json.dumps({
            "pencl_weights": "./weights/PenCL/p.bin",
            "facilitator_weights": "./weights/Facilitator/f.bin",
            "proteoscribe_weights": "./weights/ProteoScribe/s.bin",
            **extra,
        }))
        return str(path)
    return _write


@pytest.mark.parametrize("path", ["./weights/PenCL/p.bin", "weights/PenCL/p.bin",
                                  "weights/PenCL/../PenCL/p.bin"])
def test_same_file_under_another_spelling_is_recognised(weight_set, path):
    recorded = weight_set(pencl_trained_with_text_attention_mask=True)

    assert pencl_trained_with_mask(recorded, path) is True


def test_another_file_is_not(weight_set):
    recorded = weight_set(pencl_trained_with_text_attention_mask=True)

    assert pencl_trained_with_mask(recorded, "weights/PenCL/other.bin") is None


@pytest.mark.parametrize("recorded, flag", [(True, False), (False, True)])
def test_check_warns_on_disagreement(weight_set, recorded, flag):
    path = weight_set(pencl_trained_with_text_attention_mask=recorded)

    summary, warning = check_text_attention_mask(path, "weights/PenCL/p.bin", flag)

    assert summary.startswith("Caption attention mask: %s" % ("on" if flag else "off"))
    assert WARNING in warning
    assert any(path in line for line in warning)


@pytest.mark.parametrize("recorded, flag", [(True, True), (False, False), (None, True), (None, False)])
def test_check_is_quiet_otherwise(weight_set, recorded, flag):
    extra = {} if recorded is None else {"pencl_trained_with_text_attention_mask": recorded}

    summary, warning = check_text_attention_mask(weight_set(**extra), "weights/PenCL/p.bin", flag)

    assert warning == []
    assert ("not recorded" in summary) is (recorded is None)


def test_check_says_when_the_record_is_for_another_file(weight_set):
    path = weight_set(pencl_trained_with_text_attention_mask=True)

    summary, warning = check_text_attention_mask(path, "weights/PenCL/other.bin", False)

    assert warning == []
    assert "record is for ./weights/PenCL/p.bin, not the weights in use (weights/PenCL/other.bin)" in summary


def test_check_without_a_weight_set():
    summary, warning = check_text_attention_mask(None, "weights/PenCL/p.bin", True)

    assert warning == [] and "not recorded" in summary


def test_merge_fills_renamed_arguments(weight_set):
    args = Namespace(stage1_weights=None, stage2_weights="explicit.bin")

    merge_weight_set(
        args, weight_set(),
        keys=("pencl_weights", "facilitator_weights", "proteoscribe_weights"),
        rename={"pencl_weights": "stage1_weights",
                "facilitator_weights": "stage2_weights",
                "proteoscribe_weights": "stage3_init_weights"},
    )

    assert args.stage1_weights == "./weights/PenCL/p.bin"
    assert args.stage2_weights == "explicit.bin"
    assert args.stage3_init_weights == "./weights/ProteoScribe/s.bin"
