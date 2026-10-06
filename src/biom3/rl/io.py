"""Model loading helpers for GRPO.

Wraps the existing per-stage I/O so the GRPO trainer can build the frozen
Stage 1 + Stage 2 conditioning chain and a trainable Stage 3 policy from a
single config.
"""

from argparse import Namespace
from typing import Optional

import torch.nn as nn

import biom3.Stage1.model as S1mod
from biom3.Stage3.io import prepare_model_ProteoScribe
from biom3.core.distributed import is_main_process
from biom3.core.io import load_state_dict_unwrap_pl as _load_state_dict_unwrap_pl
from biom3.core.weight_sets import WEIGHT_KEYS, check_text_attention_mask, merge_weight_set
from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

_WEIGHT_SET_ARGS = {
    "pencl_weights": "stage1_weights",
    "facilitator_weights": "stage2_weights",
    "proteoscribe_weights": "stage3_init_weights",
}


def add_conditioning_args(parser):
    """Arguments shared by the RL entry points for the prompt -> z_c chain."""
    parser.add_argument("--weight_set", type=str, default=None,
                        help="Weight-set bundle JSON (e.g. configs/weights/run1_base.json). "
                             "Fills --stage1_weights, --stage2_weights and "
                             "--stage3_init_weights when they are not given, and its "
                             "record of how PenCL was trained is checked against "
                             "--text_attention_mask.")
    parser.add_argument("--text_attention_mask", action="store_true",
                        help="Pass the caption attention mask to BERT when embedding "
                             "prompts. Set it to match how the PenCL weights were "
                             "trained; off (default) for run1_base.")
    return parser


def log_mask_warning(lines):
    """Log the mask mismatch warning, if there is one, on the main rank."""
    if is_main_process():
        for line in lines:
            logger.warning(line)


def configure_conditioning(args, cfg1):
    """Apply --weight_set and --text_attention_mask to an RL run.

    Fills the stage weights the run does not name from the weight set, tells
    the prompt encoder (through ``cfg1``) whether to pass the caption attention
    mask, and logs how that compares with the weight set's record. Returns the
    lines of the mismatch warning, empty unless the two disagree, so the caller
    can repeat them when the run ends.
    """
    merge_weight_set(args, args.weight_set, keys=WEIGHT_KEYS, rename=_WEIGHT_SET_ARGS)
    cfg1.text_attention_mask = bool(args.text_attention_mask)
    summary, warning = check_text_attention_mask(
        args.weight_set, args.stage1_weights, args.text_attention_mask)
    if is_main_process():
        logger.info(summary)
    log_mask_warning(warning)
    return warning


def _attach(model: nn.Module, sd: Optional[dict], device, eval_mode: bool):
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning("missing keys (%d): %s ...", len(missing), missing[:3])
        if unexpected:
            logger.warning("unexpected keys (%d): %s ...", len(unexpected), unexpected[:3])
    if device is not None:
        model.to(device)
    if eval_mode:
        model.eval()
    return model


def load_pencl_frozen(
    cfg: Namespace,
    weights_path: Optional[str],
    device: Optional[str] = None,
) -> nn.Module:
    model = S1mod.pfam_PEN_CL(args=cfg)
    sd = _load_state_dict_unwrap_pl(weights_path, device=device) if weights_path else None
    model = _attach(model, sd, device=device, eval_mode=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_facilitator_frozen(
    cfg: Namespace,
    weights_path: Optional[str],
    device: Optional[str] = None,
) -> nn.Module:
    model = S1mod.Facilitator(
        in_dim=cfg.emb_dim,
        hid_dim=cfg.hid_dim,
        out_dim=cfg.emb_dim,
        dropout=cfg.dropout,
    )
    sd = _load_state_dict_unwrap_pl(weights_path, device=device) if weights_path else None
    model = _attach(model, sd, device=device, eval_mode=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_proteoscribe_trainable(
    cfg: Namespace,
    weights_path: Optional[str],
    device: Optional[str] = None,
) -> nn.Module:
    model = prepare_model_ProteoScribe(
        config_args=cfg,
        model_fpath=weights_path,
        device=device,
        strict=True,
        eval=False,
        attempt_correction=True,
        verbosity=2
    )
    return model
