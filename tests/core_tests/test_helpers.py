"""Tests for biom3.core.helpers."""

import pytest

from biom3.core.helpers import (
    coerce_limit_batches,
    optional_bool,
    optional_positive_int,
    resolve_bool,
    resolve_setting,
    str_to_bool,
)


@pytest.mark.parametrize("value, expected, expected_type", [
    # Fraction path: floats in (0, 1] stay float
    (0.05, 0.05, float),
    (0.5, 0.5, float),
    (1.0, 1.0, float),
    (1, 1.0, float),
    # Absolute-count path: values > 1 become int
    (2, 2, int),
    (200, 200, int),
    (200.0, 200, int),
    (5.7, 5, int),  # floor via int(); user shouldn't pass non-integer >1 anyway
    # None passes through
    (None, None, type(None)),
])
def test_coerce_limit_batches(value, expected, expected_type):
    """Cover both PL Trainer-acceptable shapes for --limit_*_batches."""
    result = coerce_limit_batches(value)
    assert result == expected
    assert type(result) is expected_type


@pytest.mark.parametrize("value, expected", [
    (None, None),
    ("None", None),
    ("none", None),
    ("", None),
    ("  ", None),
    (3, 3),
    ("3", 3),
    (" 12 ", 12),
    (5.0, 5),
])
def test_optional_positive_int(value, expected):
    result = optional_positive_int(value)
    assert result == expected
    assert result is None or type(result) is int


@pytest.mark.parametrize("value", [0, -1, "0", "-2", "abc", "2.5", 2.5, True, False, [3]])
def test_optional_positive_int_rejects(value):
    with pytest.raises(ValueError, match="num_replicas must be a positive integer"):
        optional_positive_int(value, name="num_replicas")


@pytest.mark.parametrize("value, expected", [
    (None, None),
    ("None", None),
    ("none", None),
    ("", None),
    ("  ", None),
    (True, True),
    (False, False),
    ("True", True),
    ("true", True),
    (" TRUE ", True),
    ("False", False),
    ("false", False),
    (" FALSE ", False),
])
def test_optional_bool(value, expected):
    result = optional_bool(value)
    assert result is expected


@pytest.mark.parametrize("value", [
    "yes", "no", "y", "n", "on", "off", "1", "0", 1, 0, 2, 1.0, [True], {},
])
def test_optional_bool_rejects(value):
    """An unrecognised value raises rather than being read as true."""
    with pytest.raises(ValueError, match="normalize_zc must be True or False"):
        optional_bool(value, name="normalize_zc")


@pytest.mark.parametrize("cli, config, default, expected", [
    # Command line wins, in both directions.
    (True, False, False, True),
    (False, True, True, False),
    ("True", False, False, True),
    ("False", True, True, False),
    # Config has its turn when the command line is unset.
    (None, True, False, True),
    (None, False, True, False),
    (None, "true", False, True),
    (None, "false", True, False),
    # Default only when neither is set.
    (None, None, False, False),
    (None, None, True, True),
    (None, "None", False, False),
    ("", "", True, True),
])
def test_resolve_bool(cli, config, default, expected):
    assert resolve_bool(cli, config, default) is expected


@pytest.mark.parametrize("cli, config", [("yes", None), (None, "yes"), ("yes", True)])
def test_resolve_bool_rejects(cli, config):
    with pytest.raises(ValueError, match="text_attention_mask must be True or False"):
        resolve_bool(cli, config, False, name="text_attention_mask")


@pytest.mark.parametrize("cli, config, default, expected", [
    # Command line wins.
    ("dynamic", "max_padding", "max_padding", "dynamic"),
    (0.5, 0.25, 0.0, 0.5),
    (0, 0.25, 0.9, 0),
    # Config has its turn when the command line is unset.
    (None, "dynamic", "max_padding", "dynamic"),
    (None, 0.25, 0.0, 0.25),
    ("", 0.25, 0.0, 0.25),
    # Default only when neither is set, including an explicit null or "None".
    (None, None, "max_padding", "max_padding"),
    (None, "None", "max_padding", "max_padding"),
    ("None", None, 0.0, 0.0),
    ("", "  ", "max_padding", "max_padding"),
])
def test_resolve_setting(cli, config, default, expected):
    assert resolve_setting(cli, config, default) == expected


@pytest.mark.parametrize("value, expected", [
    (True, True), (False, False),
    ("True", True), ("true", True), ("TRUE", True),
    ("False", False), ("false", False), ("FALSE", False),
])
def test_str_to_bool(value, expected):
    assert str_to_bool(value) is expected


@pytest.mark.parametrize("value", ["yes", "no", "1", "0", "", "none"])
def test_str_to_bool_rejects(value):
    """Unlike optional_bool, every value has to be stated as True or False."""
    with pytest.raises(ValueError, match="must be 'True' or 'False'"):
        str_to_bool(value)


@pytest.mark.parametrize("module_path", [
    "biom3.Stage1.run_PL_training",
    "biom3.Stage2.run_PL_training",
    "biom3.Stage3.run_PL_training",
])
def test_training_modules_share_one_str_to_bool(module_path):
    """They had a copy each; call sites use the module-local name, so it stays."""
    import importlib

    assert importlib.import_module(module_path).str_to_bool is str_to_bool
