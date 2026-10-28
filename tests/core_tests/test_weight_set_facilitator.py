"""Weight sets: the record of where ProteoScribe's conditioning vector came from.

``proteoscribe_trained_with_facilitator`` is the only source of truth for this.
The Stage 3 ``facilitator`` config key is an HDF5 group-name prefix, so a model
trained on the raw z_t can legitimately carry ``"facilitator": "MMD"``; nothing
here reads it.
"""

import json

import pytest

from biom3.core.weight_sets import (
    check_facilitator,
    proteoscribe_trained_with_facilitator,
)

WARNING = "CONDITIONING SOURCE DOES NOT MATCH THE WEIGHTS"


@pytest.fixture
def weight_set(tmp_path):
    def _write(**extra):
        path = tmp_path / "weight_set.json"
        path.write_text(json.dumps({
            "pencl_weights": "./weights/PenCL/p.bin",
            "proteoscribe_weights": "./weights/ProteoScribe/s.bin",
            **extra,
        }))
        return str(path)
    return _write


@pytest.mark.parametrize("path", ["./weights/ProteoScribe/s.bin",
                                  "weights/ProteoScribe/s.bin",
                                  "weights/ProteoScribe/../ProteoScribe/s.bin"])
def test_same_file_under_another_spelling_is_recognised(weight_set, path):
    recorded = weight_set(proteoscribe_trained_with_facilitator=False)

    assert proteoscribe_trained_with_facilitator(recorded, path) is False


def test_another_file_is_not(weight_set):
    recorded = weight_set(proteoscribe_trained_with_facilitator=False)

    assert proteoscribe_trained_with_facilitator(
        recorded, "weights/ProteoScribe/other.bin") is None


@pytest.mark.parametrize("recorded, use_facilitator", [(True, False), (False, True)])
def test_check_warns_on_disagreement(weight_set, recorded, use_facilitator):
    path = weight_set(proteoscribe_trained_with_facilitator=recorded)

    summary, warning = check_facilitator(
        path, "weights/ProteoScribe/s.bin", use_facilitator)

    assert summary.startswith("Conditioning vector: z_c = %s" % (
        "facilitator(z_t)" if use_facilitator else "z_t (no Facilitator)"))
    assert WARNING in warning
    assert any(path in line for line in warning)


@pytest.mark.parametrize("recorded, use_facilitator",
                         [(True, True), (False, False), (None, True), (None, False)])
def test_check_is_quiet_otherwise(weight_set, recorded, use_facilitator):
    extra = ({} if recorded is None
             else {"proteoscribe_trained_with_facilitator": recorded})

    summary, warning = check_facilitator(
        weight_set(**extra), "weights/ProteoScribe/s.bin", use_facilitator)

    assert warning == []
    assert ("not recorded" in summary) is (recorded is None)


def test_check_says_when_the_record_is_for_another_file(weight_set):
    path = weight_set(proteoscribe_trained_with_facilitator=False)

    summary, warning = check_facilitator(
        path, "weights/ProteoScribe/other.bin", True)

    assert warning == []
    assert ("record is for ./weights/ProteoScribe/s.bin, not the weights in use "
            "(weights/ProteoScribe/other.bin)") in summary


def test_check_without_a_weight_set():
    summary, warning = check_facilitator(None, "weights/ProteoScribe/s.bin", False)

    assert warning == [] and "not recorded" in summary


def test_a_non_boolean_record_is_rejected(weight_set):
    path = weight_set(proteoscribe_trained_with_facilitator="zt")

    with pytest.raises(ValueError, match="must be true or false"):
        proteoscribe_trained_with_facilitator(path, "weights/ProteoScribe/s.bin")


@pytest.mark.parametrize("name", ["run2_dim16_base", "run2_dim16_sh3", "run2_dim16_cme300"])
def test_the_run2_weight_sets_record_no_facilitator(name):
    path = f"configs/weights/{name}.json"

    assert proteoscribe_trained_with_facilitator(path) is False


@pytest.mark.parametrize("name", ["run0_nm_base", "run1_base"])
def test_the_earlier_weight_sets_record_one(name):
    path = f"configs/weights/{name}.json"

    assert proteoscribe_trained_with_facilitator(path) is True
