# Weights Bundle (GHCR)

The BioM3 base weights are published to GHCR as OCI artifacts, alongside the
software images. One `oras pull` gets a consistent, checksummed set of weights along with
the architecture configs those weights require in order to load successfully.

| | |
|---|---|
| Package | `ghcr.io/natural-machine/biom3-weights` |
| Tags | `run1_base`, `run2_dim16_base` (whatever the publisher pushed them as) |
| Size | ~6.44 GB for `run1_base`, ~6.43 GB for `run2_dim16_base`, ~0.34 GB per family add-on |
| Prereq | [`oras`](https://oras.land/docs/installation) — a single static binary |

## Which bundle

| tag | stages | conditioning | notes |
|---|---|---|---|
| `run1_base` | PenCL + Facilitator + ProteoScribe | `z_c = facilitator(z_t)`, z=512 | published; the NeurIPS-era stack |
| `run2_dim16_base` | PenCL + ProteoScribe (**no Facilitator**) | `z_c = z_t`, z=16 | published; Swiss-Prot + Pfam, trained on Aurora |
| `run2_dim16_<family>` | ProteoScribe only | as `run2_dim16_base` | **not yet published** — specs exist for `sh3`, `pdz`, `cm`, `ww`, `amp501`, `s1a`, `cme300` |

A `*_base` tag is self-sufficient: it carries the frozen backbones plus one trained model
per stage, so it generates on its own. A family tag carries only its ProteoScribe finetune
and is an add-on to its `*_base`, never a substitute for it.

The `dim16` in `run2_dim16_*` is the latent width. PenCL was also trained at z=32 and
z=512 in the same run family, but every ProteoScribe so far was trained at `text_emb_dim:
16` and accepts nothing else, so the width is part of the name to keep the pairings
unambiguous.

The two runs are not interchangeable: they differ in latent width, in whether a Facilitator
sits in the chain, and in two settings that fail *silently* when mismatched. Each bundle
ships a weight set recording all of it, so pass `--weight_set` and the tools check the run
against the weights:

| record | `run1_base` | `run2_dim16_base` |
|---|---|---|
| `pencl_trained_with_text_attention_mask` | `false` | `true` |
| `proteoscribe_trained_with_normalized_zc` | `false` | `true` |
| `proteoscribe_trained_with_facilitator` | `true` | `false` |

`proteoscribe_trained_with_facilitator` is the only source of truth for whether a
Facilitator was in the chain. The Stage 3 `facilitator` config key is an HDF5 group-name
prefix naming the group training data was read from — not a mode — so it says nothing
either way, and a model trained on raw `z_t` can legitimately carry `"facilitator": "MMD"`.

## Contents

The published artifact unpacks to a self-contained directory:

```txt
biom3-weights-run1_base/
├── MANIFEST.json        sha256 and size for every file
├── configs/             one architecture config fragment per stage
│   ├── _base_PenCL.json
│   ├── _base_Facilitator.json
│   └── _base_ProteoScribe.json
└── weights/             weight files, one subdirectory per component
    ├── LLMs/
    ├── PenCL/
    ├── Facilitator/
    └── ProteoScribe/
```

`run2_dim16_base` has the same shape with no `Facilitator/`, and adds its weight set and a
`NOTICE.md` covering how the weights must be conditioned:

```txt
biom3-weights-run2_dim16_base/
├── MANIFEST.json
├── configs/
│   ├── _base_PenCL.json
│   ├── _base_ProteoScribe.json
│   ├── run2_dim16_base.json       the weight set, with the three records
│   └── NOTICE.md
└── weights/
    ├── LLMs/                ESM-2 + BiomedBERT, as run1_base
    ├── PenCL/run2_dim16_base_pencl.bin
    └── ProteoScribe/run2_dim16_base_proteoscribe.bin
```

A **family bundle** carries only its own ProteoScribe weights, so layer it on `run2_dim16_base`:
pull `run2_dim16_base` first for the encoders and PenCL, then the family into the same weights
root. Its `configs/run2_dim16_<family>.json` weight set inherits the records from the
`run2_dim16_base.json` shipped beside it, overriding only `proteoscribe_weights`.

```bash
biom3_fetch_weights run2_dim16_base -o ./weights
biom3_fetch_weights run2_dim16_sh3  -o ./weights   # merges in; shared files are skipped
```

The family tags are not published yet, so the second command only works once someone has
pushed that tag; their specs are in `scripts/weights_bundle/bundle_specs/`.

- **`configs/`** holds a small JSON fragment per stage carrying only the settings needed
  to construct and load that stage. Shipping them with the weights keeps configuration and
  weights from drifting apart — a fragment and its weights are always a matched set.
- **`weights/`** holds the weight files, grouped into one subdirectory per pipeline
  component. The layout mirrors the repo's own `weights/`, so it links straight into a
  checkout (see [Pulling](#pulling)).
- **`MANIFEST.json`** records a sha256 and byte size for every file, and `oras pull`
  verifies each blob's digest against the manifest as it downloads.

## Pulling

### Standalone — nothing but `oras`

The weights need neither the `BioM3-dev` repo, the software image, nor a login (once the
package is public). Pull them, with their configs, straight to a directory:

```bash
oras pull ghcr.io/natural-machine/biom3-weights:run1_base -o biom3-weights-run1_base
```

`oras pull` checks each blob's digest against the manifest as it downloads, so a clean
pull is already integrity-verified. You get a self-contained
`biom3-weights-run1_base/{configs,weights,MANIFEST.json}` to point any `biom3` install at.

The software image pulls the same standalone way (see
[../../cloud/README.md](../../cloud/README.md) for running it):

```bash
docker pull ghcr.io/natural-machine/biom3:cuda-dev
```

Each artifact stands on its own — pull either without the other, and neither needs a
checkout.

### Into a checkout

By default `fetch_bundle.sh` pulls and verifies — it does **not** touch your checkout:

```bash
cd /path/to/BioM3-dev
./scripts/weights_bundle/fetch_bundle.sh ~/biom3-bundles --tag run1_base
```

1. `oras pull` into `~/biom3-bundles/run1_base/`
2. verifies every file's sha256 against the bundle's `MANIFEST.json`

Wiring it into the checkout is opt-in. Either pass `--link` to have fetch do it after
verifying, or run the two commands yourself from the repo root (fetch prints them on
success):

```bash
./scripts/weights_bundle/fetch_bundle.sh ~/biom3-bundles --tag run1_base --link
# equivalently, by hand:
./scripts/link_weights.sh ~/biom3-bundles/run1_base/weights weights
ln -s ~/biom3-bundles/run1_base/configs configs/bundles/run1_base
```

[`link_weights.sh`](../../scripts/link_weights.sh) never overwrites: it symlinks only the
files that are absent and reports `MATCH`/`MISMATCH` for anything already present, so review
its output before relying on the result. The bundle's `weights/` subtree mirrors the repo's
own layout, so once linked, a path like `weights/PenCL/run1_base_pencl.bin` resolves
through the symlink. Keep the pulled bundle dir around — the symlinks point into it.

`--tag` is required — it names the bundle to pull; list published tags with `oras repo tags
ghcr.io/natural-machine/biom3-weights`. `--link` wires the bundle into the checkout;
`--quick-verify` checks sizes without hashing.

### Into an image

The published image does not bake in the weights. Pull the bundle on the host and bind its
`weights/` into the container at run time:

```bash
oras pull ghcr.io/natural-machine/biom3-weights:run1_base -o ~/biom3-bundles/run1_base
# then at container run time, bind:
#   ~/biom3-bundles/run1_base/weights -> /app/weights
```

You bind only the weights: run1_base's architecture is already the default baked into the
image's `configs/inference/` configs, so you point `--config_path` at those and don't need
to bind the bundle's configs at run time. See [setup_polaris_container.md](./setup_polaris_container.md)
or [setup_aurora_container.md](./setup_aurora_container.md) for a worked container invocation,
including the `--bind` flags and Apptainer environment setup.

The bundle's `weights/` tree uses the **same filenames as `configs/weights/<name>.json`**,
so once it is bound (or pulled) into `/app/weights`, `--weight_set configs/weights/run1_base.json`
resolves against it — no per-stage `--model_path` needed for the tools that accept a weight
set (`biom3_embedding_pipeline`, Stage 3 finetuning). The individual `biom3_*_inference` /
`biom3_*_sample` entry points do not read `--weight_set`; pass `--model_path` explicitly there.

### Pulled by the container itself

The image ships `oras`, so a container can fetch its own weights with no object store and no
credentials. Set `BIOM3_WEIGHTS_BUNDLE` and the entrypoint pulls into `/app/weights` before
running your command:

```bash
docker run --rm --gpus all \
    -e BIOM3_WEIGHTS_BUNDLE=run1_base \
    ghcr.io/natural-machine/biom3:cuda-dev \
    biom3_embedding_pipeline --generate -i my.csv -o /app/outputs/gen --prefix gen \
        --weight_set configs/weights/run1_base.json
```

`BIOM3_WEIGHTS_BUNDLE_REPO` overrides the repo. The pull honours `BIOM3_SYNC_MODE`, so under
the default `auto` it is skipped when `/app/weights` is already populated (e.g. bind-mounted).
A private bundle additionally needs `GHCR_TOKEN` (and `GHCR_USER`).

### Verifying an existing bundle

`verify_bundle.py` is stdlib-only and needs neither torch nor a `biom3` install, so it
works on a bare machine:

```bash
python scripts/weights_bundle/verify_bundle.py ~/biom3-bundles/run1_base
```

## Running against the bundle's weights

Each bundle ships a per-stage architecture fragment (`configs/_base_*.json`) alongside its
weights, so the settings the weights were built with are published *with* them — you can
always see exactly what a bundle expects.

For **run1_base**, that architecture is already the default the repo's `configs/inference/`
configs encode, so there is nothing to compose: point `--config_path` straight at the stock
inference config and pass the bundle's weight file. (Weight paths are relative to the run
dir, so run from the repo root — or `/app` in the container — with the bundle's `weights/`
linked or bound there.)

```bash
biom3_PenCL_inference \
    --config_path configs/inference/stage1_PenCL.json \
    --model_path weights/PenCL/run1_base_pencl.bin \
    --input_data_path None --output_path outputs/pencl_embeddings.pt

biom3_Facilitator_sample \
    --config_path configs/inference/stage2_Facilitator.json \
    --model_path weights/Facilitator/run1_base_facilitator.bin \
    --input_data_path outputs/pencl_embeddings.pt \
    --output_data_path outputs/facilitator_embeddings.pt

biom3_ProteoScribe_sample \
    --config_path configs/inference/stage3_ProteoScribe_sample.json \
    --model_path weights/ProteoScribe/run1_base_proteoscribe.bin \
    --input_path outputs/facilitator_embeddings.pt \
    --output_path outputs/generated_sequences.csv
```

### run2_dim16_base

run2_dim16_base is the first weight set whose architecture is *not* the repo default, so it ships
with its own committed inference configs under
[../../configs/inference/run2_dim16_base/](../../configs/inference/run2_dim16_base/). They compose the
stock bases and state only the run2 deltas (`proj_embedding_dim: 16`, `text_emb_dim: 16`,
`text_attention_mask: true`, `normalize_zc: true`), so they work in a fresh checkout without
the bundle's `configs/` linked.

There is no Facilitator stage, so the chain is Stage 1 → Stage 3 and `z_c = z_t`:

```bash
biom3_embedding_pipeline --generate \
    -i my.csv -o outputs/run2 --prefix run2 \
    --weight_set configs/weights/run2_dim16_base.json \
    --pencl_config configs/inference/run2_dim16_base/stage1_PenCL.json \
    --proteoscribe_config configs/inference/run2_dim16_base/stage3_ProteoScribe_sample.json \
    --skip_facilitator --text_attention_mask --normalize_zc
```

`--skip_facilitator` is required; without it the pipeline asks for Facilitator weights that
do not exist. The `--weight_set` makes the run check itself against the three records above
and warn loudly on any mismatch. The two boolean settings are also carried by the configs,
but passing them explicitly is harmless and makes the invocation self-describing.

Swap in a family by changing one flag — the family weight set names the same PenCL:

```bash
    --weight_set configs/weights/run2_dim16_sh3.json
```

A weight set whose architecture differs from the repo defaults ships a fragment that no
longer matches `configs/inference/`; compose the shipped fragment into a config via
`_overwrite_configs`, or follow the `configs/inference/run2_dim16_base/` pattern. run1_base does
not need it.

Image: bind the pulled bundle's `configs/` to `/app/configs/bundles/run1_base` and your own
`configs/` to `/app/configs` (see [Into an image](#into-an-image)); the same
`../bundles/run1_base/...` reference then resolves inside the container.

Until the bundle is linked or bound, these configs raise
`FileNotFoundError: .../configs/bundles/run1_base/_base_PenCL.json` — that means fetch the
bundle first, not that the config is broken. `--input_data_path None` uses Stage 1's
built-in 5-protein test set, so the whole chain runs with no external data.

## Publishing (maintainers)

The publish runbook — `oras` login, build from a spec, push, and the visibility flip —
lives alongside the image's runbook in
[../../cloud/README.md](../../cloud/README.md) under **Publishing the weights bundle
(GHCR)**, so both GHCR publish flows sit in one place. A bundle's contents are declared by a
spec under `scripts/weights_bundle/bundle_specs/`.
