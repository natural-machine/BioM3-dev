# BioM3 run2 weights — what these are and how to condition them

Stage 1 PenCL `run2g_cont2` (z=16) and the Stage 3 ProteoScribe models trained
against it. Trained on Aurora, full Swiss-Prot + Pfam, fp32.

## There is no Facilitator in this weight set

ProteoScribe was conditioned on the **raw `z_t`** out of PenCL's text
projection, not on `facilitator(z_t)`. No Facilitator was trained for run2 and
none is needed. Run the embedding pipeline with `--skip_facilitator`.

`proteoscribe_trained_with_facilitator: false` in the weight set is the only
source of truth for this. The `facilitator` key in the ProteoScribe config is an
HDF5 group-name prefix naming the group the training data was read from
(`zt_data`), not a mode — do not read it as provenance either way.

## Two settings must match, or z_t is silently wrong

| setting | value | recorded in the weight set as |
|---|---|---|
| caption attention mask | **on** | `pencl_trained_with_text_attention_mask: true` |
| conditioning vector scaled to unit length | **on** | `proteoscribe_trained_with_normalized_zc: true` |

Both are off by default, and both were on for run2. A mismatch runs without
error and gives a model conditioned on vectors it was never trained on. Passing
`--weight_set` makes the tools check the run against these records and warn
loudly on a disagreement.

With the mask on, `z_t` does not depend on padding, so either `--text_padding`
mode reproduces the training-time embedding.

## Dimensions

`proj_embedding_dim: 16` (PenCL) feeds `text_emb_dim: 16` (ProteoScribe). PenCL
also exists at z=32 and z=512 from the same run family, but every ProteoScribe
here was trained at 16 and accepts nothing else.

## Which ProteoScribe checkpoint this is

`run2_dim16_base_proteoscribe.bin` is phase 2 (Swiss-Prot + Pfam) at step 58,951 of
62,500 — the run's **last** weights, not the file its checkpoint monitor labelled
"best". That monitor watched Swiss-Prot validation loss while phase 2 trained on
Swiss-Prot + Pfam, so the metric worsened from the first validation and "best"
never moved off step 1,000. On held-out Pfam (22,363 rows, never trained on):

| checkpoint | Swiss-Prot val | held-out Pfam | Pfam current-pos accuracy |
|---|---|---|---|
| phase 1 best | 2.2673 | 2.8455 | 0.0926 |
| phase 2 step 1,000 ("best") | 2.3065 | 2.7500 | 0.0966 |
| **phase 2 step 58,951 (shipped)** | 2.3226 | **2.1423** | **0.2436** |

Figures are nats per residue. Phase 2 bought 0.70 nats/residue on Pfam for 0.055
on Swiss-Prot. All seven family finetunes are seeded from the shipped file.

## Family finetunes

Published separately as `run2_dim16_<family>`: `sh3`, `pdz`, `cm`, `ww`, `amp501`,
`s1a`, `cme300`. Each holds only its ProteoScribe weights — pull `run2_dim16_base`
first for the encoders and PenCL, then layer the family on top.

`cme300` is a 300-epoch control for `cm`, not a better `cm`; it reaches the same
validation minimum at the same epoch. Prefer `cm`.

Per-family `val_loss` is **not comparable across families**: the loss divides by
the fixed 1024-slot window rather than by protein length, so it mostly tracks
mean sequence length. Compare on `val_loss_non_pad`.

## Not included

ESM-2 is shipped because PenCL's protein branch is constructed from it before
the checkpoint's own fine-tuned copy is loaded over the top; its MIT notice is in
`weights/LLMs/ESM2_LICENSE.md`. The conditioning banks each finetune was trained
against are not here — generation embeds new captions rather than reusing them.
