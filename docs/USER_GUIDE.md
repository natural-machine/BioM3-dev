# BioM3 User Guide

## About

BioM3 is a multimodal biological model that is capable of generating functional protein sequences from text prompts.
It works in three stages, each a distinct module: PenCL (Stage 1), Facilitator (Stage 2), and ProteoScribe (Stage 3).
The method is described in [Natural Language Prompts Guide the Design of Novel Functional
Protein Sequences](https://www.biorxiv.org/content/10.1101/2024.11.11.622734v1).

This guide covers basic setup and usage of BioM3.

## Installation and Setup

Setup instructions vary depending on the particular use case and computing environment. For basic workflows, we recommend using the published BioM3 container images via Docker. Alternatively, one can clone the repository, create a working conda environment, and run BioM3 commands with the installed package. Finally, for use of BioM3 on HPC environments such as Midway (UChicago) and Aurora (ALCF) we provide additional instructions for the use of Apptainer in place of Docker.

### Case 1: Using BioM3 through a repo checkout

Start by cloning the repository. 

```bash
git clone https://github.com/ranganathanlab/BioM3-dev.git && cd BioM3-dev
```

We recommend the use of conda for managing python environments, however, venv may be required in some cases, such as on HPC clusters like Aurora.
A conda environment named `biom3-env` stored under `venvs/` is the convention used here.
Refer to the per-machine setup guides under `docs/setup/` for machine-specific installation commands.

Once you have installed `biom3` into your `biom3-env` environment, verify that the installation is working by running a short suite of tests. Note that you must first source the `environment.sh` file.

```bash
source environment.sh
python -m pytest tests --quick
```

Next, fetch the current set of model weights with the following command.

```bash
biom3_fetch_weights run1_base -o weights
```

This command may take some time, as it downloads multiple gigabytes of model weights.
This will populate the `weights` directory with the necessary weights for each module.
With these weights, one can use BioM3 to embed protein sequences and text prompts, and then generate new sequences from those embeddings.

#### Embedding and generation workflow

To begin, create a csv file containing the text prompts and sequences that you want to embed. The csv should contain the header: `primary_Accession,protein_sequence,[final]text_caption`. You may use any value for the ID/accession field. Take care to wrap text captions in double quotes, in case the caption includes commas.

Note that the embeddings of sequence and text are independent processes. However, BioM3 currently requires both fields to be given. If you wish to only embed a sequence, or only embed a text caption, you may enter a dummy string (e.g. `"AAAA"` or `"Dummy text"` in the corresponding entry. Presently, an empty string input will raise an error. In the case of a dummy input, an embedding of the nonsense string will still be produced, and one should take care not to make use of those embeddings in downstream tasks.

```bash
cat > data/prompts.csv <<'EOF'
primary_Accession,protein_sequence,[final]text_caption
P69222,MAKEDNIEMQGTVLETLPNTMFRVELENGHVVTAHISGKMRKNYIRILTGDKVTVELTPYDLSKGRIVFRSR,"PROTEIN NAME: Translation initiation factor IF-1. FUNCTION: One of the essential components for the initiation of protein synthesis. Binds in the vicinity of the A-site. SUBUNIT: Monomer. SUBCELLULAR LOCATION: Cytoplasm."
my_prompt_1,"AAAA","PROTEIN NAME: SH3 domain protein. FUNCTION: Small adaptor module that binds proline-rich motifs and mediates protein-protein interactions in signal transduction. SUBCELLULAR LOCATION: Cytoplasm."
EOF
```

We will now apply the embedding half of BioM3 (Stages 1 and 2) to transform the input text and captions into vector representations in a joint embedding space. We specify the use of the "Run 1" weights that we fetched above with the `--weight_set` argument, which points to a configuration file specifying the particular set of model weights to use for each stage. Both stages also require a configuration file specifying the particular hyperparameters of each module. Prebuilt configuration files are included in the checkout, and referred to in the command below.

```bash
biom3_embedding_pipeline \
    -i data/prompts.csv \
    -o outputs/demo --prefix example \
    --weight_set configs/weights/run1_base.json \
    --pencl_config configs/inference/stage1_PenCL.json \
    --facilitator_config configs/inference/stage2_Facilitator.json
```

The embedding pipeline first transforms the input text captions into a vector $z_t$ and the protein sequence into a vector $z_p$ (Stage 1). Then, the Facilitator module further refines the text representation $z_$ into a "refined" text embedding $z_c$ (Stage 2). The result of the command above is a directory `outputs/demo` containing the following:

```text
outputs/demo/
├── example.PenCL_emb.pt          # Stage 1 output: z_t and z_p
├── example.Facilitator_emb.pt    # Stage 2 output: z_t, z_p, and z_c
├── example.compiled_emb.hdf5     # the same embeddings packaged into hdf5 format
├── example.build_manifest.json   # the exact arguments, weights, and configs used
└── example.run.log               # the run's console output
```

The filenames come from `--prefix`, and everything lands directly in the `--output_dir`
with no subdirectories. Note that the Stage 2 file is a superset of the Stage 1 file: the
Facilitator adds `z_c` to the dictionary it was given rather than writing a new one, so
`example.Facilitator_emb.pt` alone is enough for the generation step that follows.

After running the embedding half of BioM3, one can then use the refined text embeddings, $z_c$, to condition the Stage 3 module ProteoScribe and generate novel sequences. The command below takes as input the refined embeddings, and generates a specified number of sequences (default 5) for each individual embedding (i.e. text caption). Here again, we specify the weight file to use as well as a configuration file.

```bash
biom3_ProteoScribe_sample \
    -i outputs/demo/example.Facilitator_emb.pt \
    -c configs/inference/stage3_ProteoScribe_sample.json \
    -m weights/ProteoScribe/run1_base_proteoscribe.bin \
    -o outputs/demo/generation/generated.pt --fasta
```

Results populate under `outputs/demo/generation`. The results file `generated.pt` stores the generated sequences by prompt. Results can be loaded and viewed in an interactive python session as follows:

```python
import torch
results = torch.load("outputs/demo/generation/generated.pt")
prompt0_sequences = results["prompt_0"]
for i, s in enumerate(prompt0_sequences):
    print(f"Replicate {i}:", s)
```

By including the `--fasta` argument, per-prompt fasta files are produced under the `fasta` subdirectory.

### Case 2: Using BioM3 via Docker

Docker images can be thought of as self-contained software packages. One simply needs to "pull" the image from a public registry and use Docker to run it, as what is called a "container." The advantage of this approach is that one doesn't need to create a brand new conda or virtual environment. The full set of dependencies is specified in the image and are downloaded the first time the container is started.

To use Docker, one must first install it on their machine. Instructions can be found and followed online.

<!-- TODO: Eventually need to change this to ranganathanlaba -->
BioM3 Docker images are published at `ghcr.io/natural-machine/biom3`. There are four variants:

| Variant | Use case | Architectures |
| ------- | --- | ------------- |
| `cuda` | Workstations with NVIDIA GPUs | amd64, arm64 |
| `cpu` | Inference without a GPU.  | amd64, arm64 |
| `xpu` | ALCF Aurora, single node | amd64 |
| `xpu-oneapi` | ALCF Aurora, multi-node | amd64 |

Tags are patterned `<variant>-<commit>`, where the commit specifies the particular state of the BioM3 code used to create the image. As new versions of BioM3 are released, these images will be updated. To pull an image onto your workstation, specify a particular tag and run:

```bash
docker pull ghcr.io/natural-machine/biom3:cuda-779859b
```

The image carries the code, standard configuration files under `configs/`, and test
fixtures under `tests/`. It does not carry weights or data. Start by creating these directories.

```bash
cd my-project
mkdir -p weights data outputs
```

Now, fetch the current set of model weights with the following. We first point to the image that we previously pulled. Docker runs the specified biom3 command inside of this container and stops the container when the command finished (`--rm`). Docker typically runs as root, so we specify our user profile explicitly (`-u "$(id -u):$(id -g)"`). Finally, we mount the local weights directory that we created above, which Docker will have access to with the path to the right of the colon (`"$PWD/weights:/app/weights"`).

```bash
export BIOM3_IMAGE=ghcr.io/natural-machine/biom3:cuda-779859b
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/weights:/app/weights" \
    $BIOM3_IMAGE biom3_fetch_weights run1_base -o /app/weights

```

This command may take some time, as it downloads multiple gigabytes of model weights.
This will populate the `weights` directory with the necessary weights for each module.
With these weights, one can use BioM3 to embed protein sequences and text prompts, and then generate new sequences from those embeddings.

#### Embedding and generation workflow

The general workflow to embed protein sequences and captions, and to then generate novel proteins from those embeddings, is detailed above under [Case 1: Embedding and generation workflow](#embedding-and-generation-workflow). Every `biom3` command in that section runs unchanged inside the container — only the `docker run` prefix and the mounted paths differ.

Refer to the previous section for conceptual details. The equivalent commands, run using Docker, are provided below.

The mount points matter here. The container's working directory is `/app`, and both the weight-set bundle and the `biom3` arguments use paths relative to it, so `weights/`, `data/` and `outputs/` have to land at `/app/weights`, `/app/data` and `/app/outputs` for those relative paths to resolve. Results appear under `outputs/demo/`, owned by the user rather than by root because of `-u`. Drop `--gpus all` when using the `cpu` image.

```bash
# Name the image once, so the commands below can refer to it.
export BIOM3_IMAGE=ghcr.io/natural-machine/biom3:cuda-779859b

# Make sure directories exist before mounting them, otherwise Docker creates missing ones as root.
mkdir -p data outputs

# Write data/prompts.csv on the host, exactly as in Case 1 above.
cat > data/prompts.csv <<'EOF'
primary_Accession,protein_sequence,[final]text_caption
P69222,MAKEDNIEMQGTVLETLPNTMFRVELENGHVVTAHISGKMRKNYIRILTGDKVTVELTPYDLSKGRIVFRSR,"PROTEIN NAME: Translation initiation factor IF-1. FUNCTION: One of the essential components for the initiation of protein synthesis. Binds in the vicinity of the A-site. SUBUNIT: Monomer. SUBCELLULAR LOCATION: Cytoplasm."
my_prompt_1,"AAAA","PROTEIN NAME: SH3 domain protein. FUNCTION: Small adaptor module that binds proline-rich motifs and mediates protein-protein interactions in signal transduction. SUBCELLULAR LOCATION: Cytoplasm."
EOF

# Stages 1 and 2: embed the captions and sequences into z_t, z_p, and z_c.
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD/weights:/app/weights:ro" \
    -v "$PWD/data:/app/data:ro" \
    -v "$PWD/outputs:/app/outputs" \
    $BIOM3_IMAGE \
    biom3_embedding_pipeline \
        -i data/prompts.csv \
        -o outputs/demo --prefix example \
        --weight_set configs/weights/run1_base.json \
        --pencl_config configs/inference/stage1_PenCL.json \
        --facilitator_config configs/inference/stage2_Facilitator.json

# Stage 3: generate novel sequences conditioned on the refined embeddings z_c.
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD/weights:/app/weights:ro" \
    -v "$PWD/outputs:/app/outputs" \
    $BIOM3_IMAGE \
    biom3_ProteoScribe_sample \
        -i outputs/demo/example.Facilitator_emb.pt \
        -c configs/inference/stage3_ProteoScribe_sample.json \
        -m weights/ProteoScribe/run1_base_proteoscribe.bin \
        -o outputs/demo/generation/generated.pt --fasta
```

### Case 3: Using BioM3 on HPC environments through Apptainer

Many HPC environments do not permit `docker run` because it runs as root. They instead use Apptainer (formerly Singularity). The idea is largely the same as using Docker images, and we start by converting a Docker image into an apptainer compatible `.sif` file.

In this guide, we provide instructions specific to running BioM3 on ALCF's supercomputer Aurora, through Apptainer. 
Start by creating a .sif file from a specific BioM3 docker image. For working on Aurora, we use the `xpu-oneapi` image version. 

Full instructions and troubleshooting hints for building on both Aurora and Polaris are included in the files

- [setup/setup_polaris_container.md](./setup/setup_polaris_container.md)
- [setup/setup_aurora_container.md](./setup/setup_aurora_container.md)

"Bare-metal" installs (i.e. running from code installed into a local environment) are covered in [setup/setup_polaris.md](./setup/setup_polaris.md) and [setup/setup_aurora.md](./setup/setup_aurora.md).

For the following workflow, a clone of the BioM3 repo is required. Clone the repo.

```bash
git clone https://github.com/ranganathanlab/BioM3-dev.git && cd BioM3-dev
```

Next, we create a .sif image from a Docker image. 

```bash
# On a login node of Aurora
module load apptainer
export APPTAINER_CACHEDIR=/flare/NLDesignProtein/$USER/.apptainer/cache
export APPTAINER_TMPDIR=/tmp/$USER/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
apptainer build biom3_xpu-oneapi-779859b.sif docker://ghcr.io/natural-machine/biom3:xpu-oneapi-779859b
```

This command should produce a file `biom3_xpu-oneapi-779859b.sif` under the current directory. Whereas the Docker commands above point to a specified Docker image, the following commands instead point to the .sif image.
Importantly, running `apptainer exec` on Aurora requires that one be on a compute node. As such, request an interactive job or submit a pbs job to run the following. Also note that one must load apptainer.

```bash
# On a compute node of Aurora
cd /path/to/BioM3-dev
module load apptainer

export BIOM3_IMAGE="biom3_xpu-oneapi-779859b.sif"
mkdir -p weights
apptainer exec --bind "$PWD/weights:/app/weights" \
    "$BIOM3_IMAGE" biom3_fetch_weights run1_base -o /app/weights
```

Three differences from the Docker form: Apptainer already runs as you rather than as root,
so there is no `-u` and no ownership problem to correct. `--bind host:container` replaces
`-v`, with the same `/app/weights` target, since the bundle's paths still resolve against
the image's `/app` working directory. And `exec` runs the command directly instead of
through the image entrypoint. Create `weights/` yourself beforehand: Apptainer will not
create a missing bind source for you.

This command may take some time, as it downloads multiple gigabytes of model weights.
This will populate the `weights` directory with the necessary weights for each module.
With these weights, one can use BioM3 to embed protein sequences and text prompts, and then generate new sequences from those embeddings.

#### Embedding and generation workflow

The general workflow to embed protein sequences and captions, and to then generate novel proteins from those embeddings, is detailed above under [Case 1: Embedding and generation workflow](#embedding-and-generation-workflow). Every `biom3` command in that section runs unchanged inside the container — only the `apptainer exec` prefix and the bound paths differ.

Refer to that section for conceptual details. The equivalent commands, run using apptainer, are provided below. As with the weights fetch, these commands must be run from a compute node.

```bash
# On a compute node of Aurora, from the root of the checkout.

cd /path/to/BioM3-dev

module load apptainer
export BIOM3_IMAGE="$PWD/biom3_xpu-oneapi-779859b.sif"

# The wrapper mounts data/ and outputs/ by name, so they have to exist.
mkdir -p data outputs

# Write data/prompts.csv on the host, exactly as in Case 1 above.
cat > data/prompts.csv <<'EOF'
primary_Accession,protein_sequence,[final]text_caption
P69222,MAKEDNIEMQGTVLETLPNTMFRVELENGHVVTAHISGKMRKNYIRILTGDKVTVELTPYDLSKGRIVFRSR,"PROTEIN NAME: Translation initiation factor IF-1. FUNCTION: One of the essential components for the initiation of protein synthesis. Binds in the vicinity of the A-site. SUBUNIT: Monomer. SUBCELLULAR LOCATION: Cytoplasm."
my_prompt_1,"AAAA","PROTEIN NAME: SH3 domain protein. FUNCTION: Small adaptor module that binds proline-rich motifs and mediates protein-protein interactions in signal transduction. SUBCELLULAR LOCATION: Cytoplasm."
EOF

# Stages 1 and 2: embed the captions and sequences into z_t, z_p, and z_c.
scripts/aurora/apptainer_run.sh biom3_embedding_pipeline \
    -i data/prompts.csv \
    -o outputs/demo --prefix example \
    --weight_set configs/weights/run1_base.json \
    --pencl_config configs/inference/stage1_PenCL.json \
    --facilitator_config configs/inference/stage2_Facilitator.json

# Stage 3: generate novel sequences conditioned on the refined embeddings z_c.
scripts/aurora/apptainer_run.sh biom3_ProteoScribe_sample \
    -i outputs/demo/example.Facilitator_emb.pt \
    -c configs/inference/stage3_ProteoScribe_sample.json \
    -m weights/ProteoScribe/run1_base_proteoscribe.bin \
    -o outputs/demo/generation/generated.pt --fasta
```





### Fetching weights

BioM3 needs pretrained weights: ESM-2 and BioBERT for Stage 1, plus a checkpoint for each
of the three stages. The standard published set is `run1_base`, 6.4 GB. Fetch it:

```bash
biom3_fetch_weights run1_base -o weights
```

The command is in the image as well as the package, so you can run it before you have a
checkout. It compares each file against the registry's content digest and skips what is
already correct, so re-running resumes rather than restarts. Pass `--dry_run` to see what
it would download, `--force` to replace a file whose bytes differ.

The result is the layout every config expects:

```
weights/
├── Facilitator/run1_base_facilitator.bin
├── LLMs/
│   ├── BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/
│   └── esm2_t33_650M_UR50D.pt
├── PenCL/run1_base_pencl.bin
└── ProteoScribe/run1_base_proteoscribe.bin
```

Named weight sets live in [`configs/weights/`](../configs/weights/); `--weight_set
configs/weights/run1_base.json` points the entry points at all three stage checkpoints at
once, instead of naming each with `--pencl_weights`, `--facilitator_weights` and
`--proteoscribe_weights`.

On a shared cluster the weights are usually already on disk —
[setup/setup_shared_weights.md](./setup/setup_shared_weights.md) has the per-machine paths,
and [`scripts/link_weights.sh`](../scripts/link_weights.sh) symlinks them into a checkout.

## Basic Usage

Everything in this section is a plain `biom3_*` command. It is the same command whether you
run it in a conda environment, under `docker run`, or under `apptainer exec` — only the
prefix changes. The two sections after this one cover those prefixes.

The commands assume the conventional directory layout, which every runtime uses:

| Directory | Holds |
| --------- | ----- |
| `weights/` | model weights (read-only) |
| `data/` | your inputs (read-only) |
| `outputs/` | everything produced (writable) |
| `configs/` | JSON configs — shipped with the code, rarely edited directly |

### Embedding sequences and captions

Stage 1 and Stage 2 in one command. Input is a CSV with one row per prompt, carrying these
three columns; any others are ignored:

| Column | What it is |
| ------ | ---------- |
| `primary_Accession` | An identifier for the row |
| `protein_sequence` | A protein sequence, single-letter amino acid codes |
| `[final]text_caption` | The natural-language description |

The square brackets are part of the column name. Captions are truncated at 512 word-pieces,
and the models were trained on text written as `KEY: value` sections (`PROTEIN NAME:`,
`FUNCTION:`, `SUBUNIT:`, `SUBCELLULAR LOCATION:`), so prompts in that form sit closest to
the training distribution.

`protein_sequence` is required even when you only care about generating from text: Stage 1
encodes it to `z_p` and reports how far the caption landed from it, which is your main
signal that the prompt worked. It does not constrain what gets generated.

```bash
biom3_embedding_pipeline \
    -i data/prompts.csv \
    -o outputs/embeds --prefix run1 \
    --weight_set configs/weights/run1_base.json \
    --pencl_config configs/inference/stage1_PenCL.json \
    --facilitator_config configs/inference/stage2_Facilitator.json
```

This writes, under `outputs/embeds/`:

| File | Contents |
| ---- | -------- |
| `run1.PenCL_emb.pt` | Stage 1 output: `z_t` and `z_p` |
| `run1.Facilitator_emb.pt` | Stage 2 output: `z_c`, the input to generation |
| `run1.compiled_emb.hdf5` | the same embeddings packaged for Stage 3 training |
| `run1.run.log`, `run1.build_manifest.json` | the run's log and its exact settings |

`run1.run.log` reports the mean squared error between `z_c` and `z_p`. A small value means the
caption placed the model near the protein family you described.

Useful options: `--device {auto,cpu,cuda,xpu}` (default `auto`), `--batch_size` (Stage 1,
default 256), `--text_padding max_padding|dynamic` — keep `max_padding`, since `dynamic`
makes `z_t` depend on how rows happened to be batched.

To run all three stages at once, add `--generate` and its config; see the next section.

### Generating sequences

Stage 3, from the `z_c` that embedding produced:

```bash
biom3_ProteoScribe_sample \
    -i outputs/embeds/run1.Facilitator_emb.pt \
    -c configs/inference/stage3_ProteoScribe_sample.json \
    -m weights/ProteoScribe/run1_base_proteoscribe.bin \
    -o outputs/gen_seed42/generated.pt --fasta --seed 42
```

Results land in `outputs/gen_seed42/`: `generated.pt` with the tensors, and a `fasta/`
directory holding `prompt_0.fasta`, `prompt_1.fasta` and so on — one file per input row,
numbered by row order rather than by your accessions. Each holds five replicas by default:

```
>prompt_0_replica_0 seed=42
MSKSEVIEFPGTIKEAMPNAMFIVSLENEHKVIAKASGKIRMVPIRILVKDEVTVGLSPYDLKRRLIRRRASI
```

Two things to know:

- **Seeds.** `--seed` defaults to 0, and a seed of 0 or less means "pick one at random", so
  runs are *not* reproducible unless you pass a positive number. The seed actually used is
  recorded in every FASTA header and in `run.log`, so a run you liked can be reproduced
  after the fact.
- **The `fasta/` directory is created beside the `-o` file.** Give each generation run its
  own output directory, or the second run overwrites the first one's FASTA files.

Sampling behaviour is controlled by `--unmasking_order {random,confidence,confidence_no_pad}`,
`--token_strategy {sample,argmax}`, and `--num_replicas N` (sequences per prompt). Each
overrides the same-named key in the config; `num_replicas` defaults to 5 when neither sets
it.

To do everything in one command instead, add `--generate` to `biom3_embedding_pipeline`:

```bash
biom3_embedding_pipeline \
    -i data/prompts.csv \
    -o outputs/demo --prefix demo \
    --weight_set configs/weights/run1_base.json \
    --pencl_config configs/inference/stage1_PenCL.json \
    --facilitator_config configs/inference/stage2_Facilitator.json \
    --generate \
    --proteoscribe_config configs/inference/stage3_ProteoScribe_sample.json
```

`--proteoscribe_config` is required with `--generate` — the weights come from
`--weight_set`, but the config does not. `--unmasking_order`, `--token_strategy`, and
`--num_replicas` are forwarded to the sampler.

Separate steps are worth the extra command whenever you want several generation runs from
one corpus: embedding is the expensive half, and this way you pay it once.

### Finetuning

Adapting ProteoScribe to your own protein family. There are two entry points, and which you
want depends on the form your data is in:

| Entry point | Input | `z_c` is |
| ----------- | ----- | -------- |
| `biom3_train_stage3 --finetune True` | precomputed HDF5 | fixed, computed ahead of time |
| `biom3_finetune_stage3` | JSONL records | computed on-device each epoch from a composed caption |

The second is the generalized path and usually the one you want. It holds the caption
*fields* rather than a finished caption, and composes a fresh caption every epoch with
per-key dropout and shuffling, so the model sees the same protein described many ways. One
JSONL record per line:

```json
{
  "accession": "...",
  "sequence": "RALYDYQADDPSYLPFRQGDIIEVLTRLETGWWDGLLNDQRGWFPS",
  "sequence_length": 46,
  "source": "pfam",
  "fields": {
    "protein_name": "Ras GEF",
    "family_name": "SH3 domain",
    "family_description": "SH3 (Src homology 3) domains are often indicative of ...",
    "gene_ontology": "['plasma membrane', 'guanyl-nucleotide exchange factor activity']",
    "lineage": "['Eukaryota', 'Fungi', 'Dikarya', 'Basidiomycota']"
  }
}
```

Which fields exist is up to you; the config's `record_schema` decides how they become a
caption — which keys to keep, how often to drop each one, whether to shuffle. The shipped
[`configs/stage3_training/finetune_generalized_v1.json`](../configs/stage3_training/finetune_generalized_v1.json)
is a worked example, and there is a small fixture at
`tests/_data/stage3_inputs/sample_finetune_records_24.jsonl`.

```bash
biom3_finetune_stage3 \
    --config_path configs/stage3_training/finetune_generalized_v1.json \
    --finetune_data_path data/my_records.jsonl \
    --device auto --num_nodes 1 --devices_per_node 1 \
    --run_id ft001 --output_root outputs/ft001 \
    --epochs 10 --wandb False
```

Arguments worth knowing:

- `--devices_per_node` is always explicit. How many ranks to run per node is a layout
  choice, and a request for more devices than are visible stops the run rather than
  silently using fewer.
- `--pretrained_weights` names the checkpoint to start from; the config already points at
  `run1_base`.
- `--finetune_last_n_blocks` / `--finetune_last_n_layers` / `--finetune_output_layers`
  control how much of the model unfreezes.
- `--wandb False` matters: if `WANDB_API_KEY` is set in your environment it is forwarded
  into the container and every smoke run is uploaded.
- `--distributed_strategy {deepspeed_zero2,ddp}`. Use `ddp` wherever nodes do not share a
  filesystem, so one rank writes one checkpoint.

Outputs are organised under `--output_root`:

```
{output_root}/
├── checkpoints/{run_id}/     ← .ckpt files and derived weights
└── runs/{run_id}/
    ├── logs/                 ← lightning_logs/, wandb/
    └── artifacts/            ← state_dict.best.pth, args.json, build_manifest.json, run.log
```

The HDF5 path is the same trainer with `--finetune True` and a `--primary_data_path`
pointing at a compiled HDF5, which you produce with `biom3_embedding_pipeline` (without
`--generate`) or `biom3_compile_hdf5`.

Pretraining from scratch uses the same wrappers with a `pretrain_scratch_*` config; see
[docker/README.md](../docker/README.md#training-stages-1-2-3) and the per-machine job
templates under [`jobs/`](../jobs/).

## Using Docker

Every command in [Basic Usage](#basic-usage) runs unchanged inside the container. You
prefix it with a `docker run` that attaches your directories and, on a GPU host, the GPU.

With a source checkout, [`docker/run.sh`](../docker/run.sh) assembles that for you:

```bash
docker/run.sh biom3_embedding_pipeline -i data/prompts.csv -o outputs/embeds --prefix run1 ...
```

It mounts the conventional layout, runs the container as you so output files are yours, and
forwards `WANDB_API_KEY`, `NGPU` and the weights-bundle variables.

| Host (default) | Container | Mode |
| --- | --- | --- |
| `./weights` | `/app/weights` | ro |
| `./data` | `/app/data` | ro |
| `./outputs` | `/app/outputs` | rw |
| `./outputs/tests_tmp` | `/app/tests/_tmp` | rw |
| `./configs` | `/app/configs` | ro (optional; overrides the baked-in configs) |

Override the host side with `BIOM3_WEIGHTS_DIR`, `BIOM3_DATA_DIR`, `BIOM3_OUTPUTS_DIR`,
`BIOM3_CONFIGS_DIR`. Set `BIOM3_IMAGE` to choose the image.

Without a checkout, write the same thing out:

```bash
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD/weights:/app/weights:ro" \
    -v "$PWD/data:/app/data:ro" \
    -v "$PWD/outputs:/app/outputs" \
    ghcr.io/natural-machine/biom3:cuda-779859b \
    biom3_embedding_pipeline -i data/prompts.csv -o outputs/embeds --prefix run1 ...
```

Notes:

- `--gpus all` needs the NVIDIA Container Toolkit on the host. Omit it with the `cpu` image.
- `-u "$(id -u):$(id -g)"` is a Linux measure so output files are owned by you rather than
  root. On macOS and Windows, Docker Desktop handles ownership and you should omit it.
- **Symlinked weights or data.** A symlink inside a mounted directory resolves *inside* the
  container, so if `weights/` holds absolute links elsewhere on the host, those locations
  must be mounted too, at the same path. List them in `BIOM3_BIND_EXTRA` (comma-separated).
  Check with `find weights data -type l -exec readlink {} + | cut -d/ -f1-4 | sort | uniq -c`.
  Without this the links dangle and the failure surfaces far from its cause — a local
  Hugging Face model directory is reported as a malformed Hub repo id rather than a missing
  file.
- **Multi-GPU** on one host uses `torchrun`, not MPI. The training wrappers see
  `BIOM3_MACHINE=container` and dispatch to
  [`scripts/launchers/container_singlenode.sh`](../scripts/launchers/container_singlenode.sh),
  which spawns one rank per device. Set `NGPU` to match the wrapper's device argument.
- **Weights without a mount.** `BIOM3_WEIGHTS_BUNDLE=run1_base` makes the entrypoint pull
  the bundle into the image's own `/app/weights` at start-up, which requires running as root
  (`BIOM3_AS_ROOT=1`). On a host you reuse, fetch once to a host directory and mount it
  instead.

The full reference — build options, publishing, the streamlit app, multi-node on SkyPilot —
is [docker/README.md](../docker/README.md).

## Using Apptainer

Same idea, different prefix, plus cluster-specific environment the wrappers set for you.
Run from a compute node; a login node cannot create the user namespace.

**Polaris**, single node, 4 A100s:

```bash
BIOM3_IMAGE=/path/to/biom3_cuda.sif \
scripts/polaris/apptainer_run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 4 auto run001 --epochs 1
```

The wrapper passes `--nv` for the GPUs, binds `/grand`, and unsets `BIOM3_MACHINE` before
sourcing `environment.sh` so the image's baked `container` value does not mask the
`polaris` profile.

**Aurora**, single node, 12 tiles, one container with `torchrun` inside it:

```bash
export BIOM3_IMAGE=/flare/NLDesignProtein/$USER/biom3_xpu.sif
scripts/aurora/apptainer_run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 12 auto run001 --epochs 1
```

The wrapper binds `/flare` and `/lus`, sets `ZE_FLAT_DEVICE_HIERARCHY=FLAT` so each tile is
its own device, and applies the oneCCL settings the container needs but bare metal gets for
free. Bind `/lus` as well as `/flare`: the repo's `weights/` and `data/` entries are
absolute symlinks into `/lus/flare/projects/...`.

**Aurora**, multi-node, one container per rank under the host `mpiexec`:

```bash
NGPU_PER_NODE=12 NGPU_TOTAL=24 BIOM3_RANK_SOURCE=mpi \
BIOM3_FABRIC_DIR=/opt/cray/libfabric/1.22.0/lib64 BIOM3_FI_PROVIDER=cxi \
BIOM3_IMAGE=/flare/NLDesignProtein/$USER/biom3_xpu-oneapi.sif \
scripts/aurora/apptainer_mpi_run.sh \
    biom3_train_stage3 --config_path configs/stage3_training/pretrain_scratch_v1.json \
    --device auto --devices_per_node 12 --num_nodes 2 --run_id mn001 --epochs 2
```

Three things this path requires:

- **`BIOM3_RANK_SOURCE=mpi`.** Without it the wrapper falls back to translating the PALS
  rank variables, and oneCCL setup dies with a SIGSEGV.
- **`BIOM3_FABRIC_DIR` and `BIOM3_FI_PROVIDER=cxi`.** Aurora's CXI provider lives in HPE's
  Cray libfabric, not in the image. Without them the run falls back to `tcp` and ends up
  slower than a single node.
- **Entry points are called directly**, not through the `scripts/stage*_{single,multi}node.sh`
  wrappers. `mpiexec` has already spawned one process per rank; a wrapper would spawn them
  a second time.

Run this from the shell `qsub -I` gives you — the wrapper reads `$PBS_NODEFILE`, which PBS
sets only there — and do not `module load frameworks`, which exports host values the
wrapper has to override.

There is no progress bar under `mpiexec`: each rank's stdout is a pipe, and the bar only
appears on a terminal. Follow the run in TensorBoard, W&B, or the per-epoch validation
lines.

Because the `.sif` is read-only and some code writes into the image tree, both wrappers pass
`--writable-tmpfs` and mount `<outputs>/tests_tmp` at `/app/tests/_tmp`. Invoking
`apptainer exec` by hand means adding both yourself.

Environment variables, failure modes and measured throughput are in
[setup/setup_aurora_container.md](./setup/setup_aurora_container.md) and
[setup/setup_polaris_container.md](./setup/setup_polaris_container.md).
