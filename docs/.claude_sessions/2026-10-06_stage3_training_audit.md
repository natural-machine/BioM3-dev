# Session: Stage 3 training audit — schedule length, attention mask, pretrained weights, padding in the loss

**Date:** 2026-10-06
**Branch:** `stage3-training-fixes` (from `dev` at `17db3ec`)

## Goal

After the Stage 1 inference fixes of the same day
(`2026-10-06_stage1_inference_weight_loading.md`), look through the Stage 3 training
scripts for the same kind of lingering issue: caption masking and padding first, then
anything else that would silently train the wrong thing. The user chose three of the
findings to fix in this round.

## What the audit found

Masking and padding:

- **Padding is fine.** Generalized finetuning pads captions to the fixed `text_max_length`
  (`Stage3/preprocess.py::make_seq_caption_collate_fn`), so `z_c` does not depend on batch
  composition.
- **The mask could not be passed.** That collate dropped the attention mask and the frozen
  embedder called BERT with `input_ids` only; the RL prompt encoder likewise. Right for
  `run1_base`, wrong for PenCL trained with the mask, and silent. **Fixed.**
- **The HDF5 path trusts `z_c`.** `biom3_train_stage3` reads `text_to_protein_embedding`
  from a file that records nothing about how it was made. Not changed.

Other:

- **The cosine schedule ignored `num_nodes`.** **Fixed.**
- **Finetuning without pretrained weights started from random weights.** **Fixed.**
- **Validation captions are re-randomised on every pass.** Train and validation share one
  `GeneralizedRecordDataset`, so per-field dropout and shuffle apply to validation too. Not
  changed (the user left it out of this round).
- **The embedder runs in fp32** while Stage 1 inference defaults to autocast, so the two
  match exactly only against `--no_amp`. Docstring corrected; behaviour unchanged.

## Changes

| Commit | What |
| ------ | ---- |
| `94f6c3f` | `fix:` `optimizer_steps_per_epoch` / `set_traindata_len` in `Stage3/run_PL_training.py`, used by both training scripts |
| `1e97d6c` | `fix:` `require_finetune_weights` in `biom3_train_stage3`; the same check in `biom3_finetune_stage3` |
| `3e0e288` | `feat:` `--text_attention_mask` and `--weight_set` in `biom3_finetune_stage3` and the GRPO, GDPO and DPO entry points |

### Schedule length

`args.traindata_len` is the number of optimizer steps in an epoch; `coswarmup` warms up
over that many steps and decays over `traindata_len * epochs`. It was computed as the
single-process loader length divided by `devices_per_node`, before `torch.distributed` was
initialized. So it ignored `num_nodes` (N times too long on N nodes), ignored that each
rank's shard is truncated to `num_samples // world_size`, and could be 0, which made the
rate alternate between full and zero.

It is now counted from the dataset size and `num_nodes * devices_per_node`. The default
`log_every_n_steps` follows it, so a multi-node run logs once per real epoch again.

### Attention mask

`--text_attention_mask` (default off) decides whether BERT gets the mask. With it off, the
batch contents and the calls into the embedder are exactly what they were. With it on, the
finetune collate emits the mask after `input_ids`, `PL_ProtARDM_Finetune` hands it to
`TextToZcEmbedder`, and the RL `_PromptEncoder` takes it from the tokenizer.

Both paths accept `--weight_set`. It fills the weights the run does not name and its
`pencl_trained_with_text_attention_mask` record is compared with the flag by the shared
`core.weight_sets.check_text_attention_mask`, which the embedding pipeline now uses too. A
mismatch logs the `CAPTION ATTENTION MASK DOES NOT MATCH THE WEIGHTS` banner at the start
and end of the run; the run follows the flag. Weight paths are compared after
normalisation, so `weights/x.bin` and `./weights/x.bin` are the same file. The two
generalized-finetune configs now name `configs/weights/run1_base.json`.

## Verification

Bare metal on Aurora compute nodes, worktree source over the main checkout's venv.

Job 8907033, 2 nodes:

| Check | Result |
| ----- | ------ |
| Schedule on 2 nodes x 12 ranks, 800 training samples, batch 8, 3 epochs | built for 5 steps per epoch and 15 in total; the run took 5, 5, 5 |
| Finetune embedder vs pipeline `z_c`, same mask setting | 4e-7 (off), 3e-7 (on) |
| RL prompt encoder vs pipeline `z_c`, same mask setting | 2e-6 (off), 1.5e-6 (on) |
| Either path against the pipeline with the other mask setting | 0.68 |
| Generalized finetune, flag off | trains one step; no banner |
| Generalized finetune, flag on, `run1_base` weight set | trains one step; banner at the start and at the end |
| Generalized finetune without pretrained weights | refused |

Job 8907034, 1 node: `pytest tests`, 1625 passed, 94 skipped.

Earlier runs that show the old schedule length: a 4-node run built for 8 steps per epoch
took 2 (`Num_warmup_steps=8`, 10 steps in the whole run); a 2-node run came out at 0.

## Learning-rate history: run2d against the Stage 3 schedule

The user asked whether the rate still decays enough, with `run2d` as the reference. `run2d`
is a Stage 1 run (`pencl_sanity` workspace, `run2d_V20260926_161806`), and Stage 1 has no
scheduler: its metrics history logs one value per parameter group for all 2,680 steps
(2.3e-4, 2.3e-6, 2.3e-3). These changes do not touch Stage 1.

For a Stage 3 run of that shape (3,072 ranks, 134 steps per epoch, 20 epochs), stepping the
real scheduler gives, as a fraction of the peak rate at the start of each epoch:

| Epoch | 0 | 1 | 5 | 10 | 15 | 19 | end | mean |
| ----- | - | - | - | -- | -- | -- | --- | ---- |
| old length | 0.000 | 0.004 | 0.020 | 0.039 | 0.059 | 0.074 | 0.078 | 0.039 |
| new length | 0.000 | 1.000 | 0.895 | 0.541 | 0.161 | 0.007 | 0.000 | 0.500 |

With the old length the warmup alone was 256 epochs, so the run would have ended at 7.8% of
the peak without ever decaying. The zero at epoch 0 is the first step only: the rate climbs
linearly through the first epoch.

### What past multi-node Stage 3 runs did

Lightning's `LearningRateMonitor` logged the rate in each run's `logs/lightning_logs`. For
three generalized finetunes under `outputs/Stage3/` in the main Aurora checkout:

| Run | Warmup built for | Run length | Logged mid-run | Logged at the last step |
| --- | ---------------- | ---------- | -------------- | ----------------------- |
| 64 nodes x 12, 50 epochs (`..._n64_d12_e50_V20260702_232126`) | 289 steps | 250 steps | 5.2e-5, still rising | 8.6e-5; never reached the 1e-4 peak |
| 16 nodes x 12, 300 epochs (`..._sh3_prod_n16_d12_e300_V20260703_213155`) | 126 steps | 2,400 steps | 9.98e-5 | 9.91e-5 (99.1% of peak) |
| 16 nodes x 12, 3000 epochs (`..._sh3_prod_n16_d12_e3000_V20260703_224952`) | 126 steps | 24,000 steps | 1.382e-3 | 1.372e-3 (99.0% of peak) |

Stepping the real scheduler with those old lengths reproduces every logged value. So
multi-node runs trained with a long warmup and then an almost constant rate; single-node
runs, where the old length was right, did decay. With the new length the same runs warm up
over one epoch (5 to 8 steps) and decay to zero by the last step. That is a change in how
multi-node runs train, not only in what the log says. The user reviewed this comparison and
kept the fix as it is.

## Second round: RL loading and padding in the loss

After merging the first three fixes the user asked for two more changes, on the same
branch (from `dev` at `6a8b9a6`).

| Commit | What |
| ------ | ---- |
| `cc8244b` | `fix:` RL loads PenCL through `load_pencl_weights`, requires every Facilitator tensor, and requires all three weight paths |
| `a43e324` | `feat:` `loss_all`, `loss_non_pad` and `loss_pad` computed and logged in Stage 3 training; `--loss_positions all\|non_pad` chooses the gradients |

### The loss terms

The Stage 3 loss sums the log-probability of the true token over the unsampled positions
of a fixed-length canvas, and most of that canvas is tail padding. The suggestion was to
ignore the padded positions. Training now computes three terms on every step:

- `loss_all`: every unsampled position, exactly as before.
- `loss_non_pad`: unsampled positions whose true token is `<START>`, a residue or `<END>`.
- `loss_pad`: unsampled positions whose true token is the pad (`-`, id 23).

Padding is decided by the true token and never by the mask token (id 0), which marks what
the model has not been shown. A pad that has been sampled is context and is in no term.
Each term divides its summed log-probability by its own count plus one, the convention
`weight_log_prob` already uses, so with no padding `loss_non_pad` equals `loss_all`.

`--loss_positions` picks `all` or `non_pad` for the gradients, and `train_loss` /
`val_loss` report the chosen one. The user leaned towards making `non_pad` the default;
it was left at `all` because of the experiment below.

### Why the default stayed `all`

The sampler decodes by deleting the special tokens from all positions
(`run_ProteoScribe_sample.py`); it does not cut at the first `<END>`. So the length of a
generated protein is set by where the model emits pads, and `non_pad` removes the only
training signal for that.

Job 8907188: three runs from `run1_base`, 16 epochs (832 steps), single rank, same seed, on
1,024 Swiss-Prot rows embedded by `run1_base`'s own PenCL and Facilitator. Validation at
the last epoch, and sequences generated by each final model for five prompts:

| Gradients from | `loss_all` | `loss_non_pad` | `loss_pad` | Training gradient norm |
| -------------- | ---------- | -------------- | ---------- | ---------------------- |
| nothing (learning rate 0) | 0.346 | 1.596 | 0.0015 | |
| all positions | 0.366 | 1.694 | 0.0014 | 0.50 |
| non-pad only | 0.375 | 1.716 | 0.0130 | 2.29 |

| Prompt's own protein length | 72 | 189 | 380 | 548 | 632 |
| --------------------------- | -- | --- | --- | --- | --- |
| `run1_base`, mean generated length | 122 | 189 | 380 | 547 | 625 |
| after `all` | 81 | 189 | 371 | 543 | 608 |
| after `non_pad` | 326 | 215 | 382 | 555 | 660 |

- With `non_pad`, `loss_pad` rose about tenfold in training and validation, and generated
  sequences got longer, most for the shortest protein (one replica reached 723 residues).
  With `all` neither moved.
- `non_pad` did not improve the validation sequence loss (1.716 against 1.694). With
  matched conditioning the padding term's gradient is already close to zero, so the two
  objectives pull the sequence positions the same way.
- The non-pad term is about 4.6 times larger, in value and in gradient norm, so `non_pad`
  also acts like a larger learning rate.

Job 8907128 was the same experiment run by mistake on the baked test HDF5, whose `z_c`
predates `run1_base`. There the model starts out unable to end a sequence (`loss_pad` 7.3,
generated lengths 1,015 to 1,024). With `all` it learned to within 400 steps (lengths 289
to 496); with `non_pad` it never did (`loss_pad` 8.9, lengths 1,014 to 1,023).

What does answer the original concern without changing the gradients: `loss_non_pad` is the
undiluted per-residue number, and `checkpoint_monitors` can select on `val_loss_non_pad`.

Job 8907129, full test suite on these changes: 1650 passed, 94 skipped.

## Open items

1. **Multidomain finetuning and sampling** (`Stage3/multidomain/`) embed captions through
   the same frozen embedder with their own collate, and cannot pass the mask.
2. ~~RL weight loading is non-strict.~~ Fixed in `cc8244b`.
3. **`--warmup_steps` is not used.** It is documented as the cosine warmup length, but the
   schedule always warms up over one epoch. On many ranks one epoch is only a few steps (5
   on the 64-node run above), so wiring this argument in would make the warmup explicit.
4. **Step-based training** (`training_strategy=combine`) runs to `max_steps` while the
   schedule is still sized from `epochs`.
5. **Stage 1 has no learning-rate schedule.** If one is wanted for the run2 series, it is a
   new feature there.
6. **Validation captions** are re-randomised every pass (see above).
7. **The HDF5 path** has no record of how its `z_c` was made.
8. The Stage 3 cosine ends at zero, so the last epoch runs below 1% of the peak. That is
   how the schedule is written, not part of this fix.
9. `weight_log_prob` divides by the number of unsampled positions plus one, where the
   count itself would be exact. Left as it is, and mirrored in the new terms.
10. The baked `Stage2_MMD_swissprot_embedding_subset_1000.hdf5` and
    `test_Facilitator_embeddings.pt` carry `z_c` that predates `run1_base`; with them
    `run1_base` generates full-length sequences. Fine for smoke tests, misleading for
    anything that looks at the model's behaviour.

## Reverting

```bash
git revert a43e324 cc8244b 3e0e288 1e97d6c 94f6c3f
```

The three are independent in behaviour; `94f6c3f` can be reverted on its own.
