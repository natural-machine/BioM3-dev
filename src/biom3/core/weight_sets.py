"""Named weight-set bundles.

A weight set clumps the Stage 1/2/3 trained checkpoints that belong together
into one JSON file (e.g. ``configs/weights/run1_base.json``) so a whole model
stack can be selected with a single ``--weight_set`` flag instead of three
separate path arguments.

Bundle paths follow the same convention as the rest of the repo's configs:
relative paths resolve against the working directory (run from the repo root),
absolute paths are used as-is. The bundle itself supports config composition
via ``load_json_config``.

A bundle can also record how its PenCL weights were trained, under
``pencl_trained_with_text_attention_mask`` (true or false), so a run can be
checked against it.
"""

from __future__ import annotations

import os

from biom3.core.helpers import load_json_config

WEIGHT_KEYS = ("pencl_weights", "facilitator_weights", "proteoscribe_weights")
PENCL_MASK_KEY = "pencl_trained_with_text_attention_mask"


def load_weight_set(path):
    """Load a weight-set bundle and return the recognized weight keys.

    Unknown keys in the bundle are ignored; missing keys come back absent.
    """
    cfg = load_json_config(path)
    return {k: cfg[k] for k in WEIGHT_KEYS if k in cfg and cfg[k] is not None}


def merge_weight_set(args, weight_set_path, keys, rename=None):
    """Fill ``args.<key>`` from a bundle for each key not already set on the CLI.

    Explicit CLI values (anything other than ``None`` / ``"None"``) win over the
    bundle. ``keys`` restricts which weight keys this consumer cares about
    (e.g. the embedding pipeline only needs pencl + facilitator). ``rename``
    maps a bundle key to the attribute it fills, for consumers whose argument
    names differ (e.g. ``{"pencl_weights": "stage1_weights"}``).
    """
    if not weight_set_path or str(weight_set_path) == "None":
        return
    rename = rename or {}
    bundle = load_weight_set(weight_set_path)
    for key in keys:
        attr = rename.get(key, key)
        current = getattr(args, attr, None)
        if current in (None, "None") and bundle.get(key):
            setattr(args, attr, bundle[key])


def pencl_trained_with_mask(weight_set_path, pencl_weights=None):
    """Whether a bundle records PenCL as trained with the caption attention mask.

    Returns the bundle's ``pencl_trained_with_text_attention_mask`` value, or
    None when nothing can be said: no bundle, no such key, or ``pencl_weights``
    is not the PenCL file the bundle names.
    """
    if not weight_set_path or str(weight_set_path) == "None":
        return None
    cfg = load_json_config(weight_set_path)
    value = cfg.get(PENCL_MASK_KEY)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(
            f"{PENCL_MASK_KEY} in {weight_set_path} must be true or false, "
            f"got {value!r}"
        )
    if pencl_weights is not None:
        recorded_for = cfg.get("pencl_weights")
        if recorded_for is None or (
                os.path.normpath(pencl_weights) != os.path.normpath(recorded_for)):
            return None
    return value


def check_text_attention_mask(weight_set_path, pencl_weights, text_attention_mask):
    """Compare a run's caption attention mask setting with a bundle's record.

    Returns ``(summary, warning)``: one line saying what the run does and what
    the bundle records, and the lines of a loud warning, which is empty unless
    the two disagree. The run's setting is not changed either way.
    """
    trained_with_mask = pencl_trained_with_mask(weight_set_path, pencl_weights)
    summary = "Caption attention mask: %s; %s" % (
        "on" if text_attention_mask else "off",
        {True: "the weight set records PenCL as trained with it",
         False: "the weight set records PenCL as trained without it",
         None: "how PenCL was trained is not recorded"}[trained_with_mask],
    )
    if trained_with_mask is None or trained_with_mask == bool(text_attention_mask):
        return summary, []
    bar = "!" * 72
    return summary, [
        bar,
        "CAPTION ATTENTION MASK DOES NOT MATCH THE WEIGHTS",
        f"{weight_set_path} records PenCL as trained "
        f"{'with' if trained_with_mask else 'without'} the mask "
        f"({PENCL_MASK_KEY}),",
        f"but this run has --text_attention_mask "
        f"{'on' if text_attention_mask else 'off'}.",
        "z_t will not be what these weights were trained to produce.",
        bar,
    ]
