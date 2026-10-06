# Session: Stage 1 inference weight loading, `--skip_facilitator`, caption attention mask

**Date:** 2026-10-06
**Branch:** `stage1-inference-loading` (from `dev` at `b1a8363`)

## Goal

Act on a handoff from an agent working in the `embedding-space-perturbations` workspace
(`BIOM3_DEV_HANDOFF.md` there), which reported four problems with Stage 1 inference and the
embedding pipeline. All four were checked against `dev` and confirmed before any change.

## Summary

1. **Stage 1 inference could embed with weights that never loaded.** Fixed.
2. **The pipeline could not run a PenCL weight set that has no Facilitator.** Added
   `--skip_facilitator`.
3. **Inference never passed the caption attention mask to BERT**, although Stage 1 pfam
   training has since `ef43803` (2026-09-09). Added `--text_attention_mask`, off by default.
4. **Two weight files in the main Aurora checkout are Lightning checkpoints under a `.bin`
   name.** Not changed: the files are read-only and replacing them is the user's decision.
   With fix 1 the PenCL one now loads correctly anyway.

## The loading bug

`biom3_PenCL_inference` chose its loader from the file extension. Anything not named `.ckpt`
went through `load_and_prepare_model(strict=False)`, which unwraps a Lightning checkpoint
but keeps its `model.` prefix. For a Lightning checkpoint named `.bin`, no key matched, the
non-strict load reported nothing, and the run embedded with an untrained projection head.
The sign is every `z_t` row norm equal to 22.63 (√512, from the untrained LayerNorm).

This is the same failure as the finetune-embedder bug fixed in July
(`2026-07-24_pencl_finetune_embedder_load_fix.md`), in a path that fix did not cover.

Both loader paths now go through `Stage1.io.load_pencl_weights`:

- The format is read from the file (`core.io.load_state_dict_unwrap_pl`), not its name.
- The load is non-strict but checked: it raises unless every parameter is populated.
  Parameters are tracked by identity, so a tied parameter (the ESM-2 and BERT output heads)
  counts as loaded when any of its names is in the file.
- Missing buffers and unused keys only warn. This is what the old `strict=False` was for:
  `BioM3_PenCL_epoch20.bin` and `PenCL_V09152023_last.ckpt` carry a
  `text_encoder.model.bert.embeddings.position_ids` buffer that current transformers no
  longer keeps.
- Checkpoints no longer go through the PL wrapper. The wrappers hold the network as
  `self.model` and have no parameters or load hooks of their own.

`--load_from_checkpoint` is kept for compatibility. It, or a `.ckpt` path, still selects the
network class from the config's `model_type`; it no longer changes how weights load.

## Changes

| Commit | What |
| ------ | ---- |
| `da524c6` | `fix:` `load_pencl_weights` in `Stage1/io.py`; both paths in `run_PenCL_inference.py` use it; 14 tests |
| `458897c` | `feat:` `--skip_facilitator` in `pipeline/embedding_pipeline.py`; 4 tests |
| `a1a43ec` | `feat:` `--text_attention_mask` through `preprocess.collate_fn(include_mask=...)`, Stage 1 inference and the pipeline; 10 tests |
| this commit | `docs:` `emb` pass criterion and open item 17 in `container_validation.md`; this note |

## Verification

Run bare metal on an Aurora compute node (`debug` queue), worktree source over the main
checkout's venv, torch 2.13.0a0, transformers 5.14.1.

Job 8906615, weight loading:

| File | Format in the file | Result |
| ---- | ------------------ | ------ |
| published `run1_base_pencl.bin` | raw, 788 keys | 752/752 parameters |
| main checkout's `run1_base_pencl.bin` | Lightning, 788 `model.` keys | 752/752 parameters |
| `run1_base_pencl.ckpt` | Lightning | 752/752 parameters |
| `BioM3_PenCL_epoch20.bin`, `PenCL_V09152023_last.ckpt` | raw / Lightning, 789 keys | loaded, one unused key warned |
| Facilitator weights (negative control) | raw, 6 keys | refused |

Through the entry point, the mislabelled file and the published one give the same
embeddings: cosine 1.000000, largest difference 2e-6, `|z_t|` 12.93 to 14.26.

Job 8906650, mask and pipeline, 64 rows, fp32 (`--no_amp --float32_matmul_precision highest`):

| Comparison | `z_t` |
| ---------- | ----- |
| mask off, old code vs new code, CPU | bit-identical |
| mask off, old code vs new code, XPU | largest difference 3e-6 |
| mask on, `max_padding` vs `dynamic` | largest difference 1.4e-5 |
| mask off, `max_padding` vs `dynamic` | cosine mean 0.978, min 0.805 |
| `run1_base`, mask on vs off | cosine mean 0.872, min 0.695 |

`z_p` is unchanged in every comparison (differences at 2e-6). The last row is why the option
is off by default: `run1_base` was trained without the mask, and turning it on moves `z_t` a
long way.

The same job ran the pipeline with `--skip_facilitator` on a weight set holding only
`pencl_weights` (three files written, no Stage 2, manifest lists only `pencl_output`),
confirmed that weight set is refused without the flag, ran the full pipeline, and ran the
tests: Stage 1 inference and pipeline tests 52 passed, 4 skipped (no CUDA);
`pytest tests --quick` 1465 passed, 180 skipped.

The loader tests fail on the old code (8 of the 9 `prepare_model` cases). `da524c6` and
`458897c` were each checked on their own tree with the stub-based tests.

## Open items

1. **Replace the mislabelled weights in the main Aurora checkout?**
   `weights/PenCL/run1_base_pencl.bin` (3,918,095,585 bytes; published is 3,048,812,529)
   and `weights/Facilitator/run1_base_facilitator.bin` (12,618,834; published 4,203,629).
   Both are read-only. `scripts/weights_bundle/bundle_specs/run1_base.json` reads them, so
   a bundle built from this checkout would publish Lightning checkpoints as `.bin`.
2. **Rerun the invalid `emb` outputs**: `outputs/validation/au1-emb`, `au2-emb` and
   `au2-emb.pre-4ea2435` on Aurora, and anything downstream of them.
3. **Rebuild the images.** The changes are in `src/`, so `xpu-oneapi-abd9941` and the other
   published images still have the loading bug.
4. **Record the mask setting with the weights?** It is a property of how a weight set was
   trained, but today it is a flag the caller has to know to pass. A key in the weight-set
   JSON would remove that. Not done: it is a design decision.
5. **The mask in the other caption paths.** `Stage3/finetune_embedder.py:52` and
   `rl/grpo.py:138` also embed captions without the mask. That is right for `run1_base`
   and wrong for weights trained with the mask. Left alone: they need the same option
   before they are used with such weights.

## Reverting

```bash
git revert a1a43ec 458897c da524c6
```

The three commits are independent in content, except that one test in `a1a43ec` uses
`--skip_facilitator` from `458897c`.
