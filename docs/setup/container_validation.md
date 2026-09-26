# Container validation matrix

Tracks whether each BioM3 container image runs each main entry point on each machine,
launched through the current wrappers (`docker/run.sh`, `scripts/{aurora,polaris}/apptainer_*.sh`,
`cloud/run.mithril.yaml`). A cell is one machine and node count crossed with one entry point.
It is checked off only after that cell's command below has run and met the pass criteria.

Cell IDs are `<row>-<column>`, e.g. `au2-pt`. Use the cell ID as the `run_id` and as the
output directory name, so every result is traceable.

## Grid

| Row | Machine | Nodes × devices | Image | emb | s1 | pt | ft | gft | gen |
| --- | ------- | --------------- | ----- | --- | -- | -- | -- | --- | --- |
| `sp1` | DGX Spark | 1 × GB10 | cuda (arm64) | pass | pass | pass | pass | fail | pass |
| `sp2` | DGX Spark | 2 × GB10 | cuda (arm64) | blocked | blocked | blocked | blocked | blocked | blocked |
| `mac` | Local Mac | 1 × CPU | cpu (arm64) | todo | n/a | n/a | n/a | n/a | todo |
| `au1` | Aurora | 1 × 12 tiles | xpu | todo | todo | todo | todo | blocked | todo |
| `au2` | Aurora | 2 × 12 tiles | xpu-oneapi | todo | todo | todo | todo | blocked | todo |
| `po1` | Polaris | 1 × 4 A100 | cuda (amd64) | todo | todo | todo | todo | blocked | todo |
| `po2` | Polaris | 2 × 4 A100 | cuda (amd64) | blocked | blocked | blocked | blocked | blocked | blocked |
| `mi1` | Mithril | 1 × N GPU | cuda (amd64) | todo | todo | todo | todo | blocked | todo |
| `mi2` | Mithril | 2 × N GPU | cuda (amd64) | blocked | todo | todo | todo | blocked | todo |

Status key:

- `pass`: the cell's command ran and met the pass criteria. Add a line to the [results log](#results-log).
- `fail`: it ran and did not. Add a line to the results log with the error.
- `todo`: not run yet.
- `n/a`: not applicable by design (see [row notes](#row-notes)).
- `blocked`: a known gap, found by reading the code, stops this cell before it can run (see [open items](#open-items)).

Columns:

| Column | Entry point | Covers |
| ------ | ----------- | ------ |
| `emb` | `biom3_embedding_pipeline` | CSV → Stage 1 (PenCL) → Stage 2 (Facilitator) → compiled HDF5 |
| `s1` | `biom3_train_stage1`, via `scripts/stage1_train_*node.sh` | PenCL training from scratch |
| `pt` | `biom3_train_stage3`, via `scripts/stage3_train_*node.sh` | ProteoScribe pretraining from scratch on precomputed z_c (HDF5) |
| `ft` | `biom3_train_stage3 --finetune True` | ProteoScribe finetuning from `run1_base` on precomputed z_c (HDF5) |
| `gft` | `biom3_finetune_stage3` | Generalized finetuning on JSONL records, z_c computed on the device |
| `gen` | `biom3_ProteoScribe_sample` | Sequence generation from z_c |

## Images under test

| Variant | Tag | Rows |
| ------- | --- | ---- |
| cuda | `ghcr.io/natural-machine/biom3:cuda-2066a75` (amd64 + arm64; also `cuda-dev`) | `sp*`, `po*`, `mi*` |
| cpu | `ghcr.io/natural-machine/biom3:cpu-3e6d7ab` (amd64 + arm64; also `cpu-dev`) | `mac` |
| xpu | `ghcr.io/natural-machine/biom3:xpu-2066a75` (amd64) | `au1` |
| xpu-oneapi | `ghcr.io/natural-machine/biom3:xpu-oneapi-2066a75` (amd64) | `au2` |

The image contents have not changed since 2066a75: later commits touch only the host-side
Apptainer wrappers and `cloud/`. The cpu image is older (2026-09-02) and predates
`--device auto`; see [open items](#open-items). If an image is rebuilt, update this table
and re-run the affected cells.

## Standard inputs

Every cell uses the `run1_base` weight set (ESM-2, BioBERT, and the PenCL, Facilitator
and ProteoScribe `run1_base` weights), either fetched and mounted as `weights/` or pulled
by the entrypoint with `BIOM3_WEIGHTS_BUNDLE=run1_base`
([docs/setup/weights_bundle.md](weights_bundle.md)). The data inputs are baked into the
image under `/app/tests/_data`, so they need no mount, except where noted.

| Input | Path inside the container | Used by |
| ----- | ------------------------- | ------- |
| Prompts CSV, 5 rows | `tests/_data/stage1_inputs/sample_text_seqs1.csv` | `emb`, `s1` (1 rank) |
| Larger CSV for more than 1 rank | not chosen yet; see open item 4 | `emb`, `s1` (more than 1 rank) |
| z_c HDF5, 1,000 rows | `tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5` | `pt`, `ft` |
| JSONL records | not in the image; see open item 3 | `gft` |
| z_c for 5 prompts | `tests/_data/embeddings/test_Facilitator_embeddings.pt` | `gen` |

Getting `run1_base` into the checkout's `weights/` before a row that mounts it:
`scripts/link_weights.sh` symlinks them from a shared canonical directory (on Spark,
`/data/data-share/BioM3-data-share/data/weights`), or fetch the published bundle once with
`scripts/weights_bundle/fetch_bundle.sh`. Rows that set `BIOM3_WEIGHTS_BUNDLE` instead
(`mac`, `mi1`, `mi2`) need no local `weights/` at all.

## Running a row

Every command below is literal. Run one **row preamble** first: it defines `$R` (the
container runner), `$ROW`, `$NGPU` (devices per node), `$LAUNCH` and `$JSONL`, and exports
the image and mount settings. After that the six **cell commands** are identical on every
machine — copy them as-is. Run everything from the BioM3-dev checkout, since the wrappers
live in the repo. The two Aurora multi-node and the two Mithril rows do not fit that
shape and carry their own complete commands below.

`$LAUNCH` is empty when `NGPU=1` and is `scripts/launchers/container_singlenode.sh` when
`NGPU` is greater than 1, which is what spawns one rank per device for the entry points
that are not called through a training wrapper (`emb`, `gft`, `gen`). `NGPU` reaches the
container either way: `docker/run.sh` forwards it explicitly, and no Apptainer wrapper
uses `--cleanenv`, so the host environment passes through.

Each command tees to `outputs/validation/logs/$ROW-<column>.log`, which is what the
[results log](#results-log) refers to.

### Row preambles (`sp1`, `mac`, `au1`, `po1`)

```bash
# --- sp1: DGX Spark, 1 GB10 ---------------------------------------------
export ROW=sp1 NGPU=1
export BIOM3_IMAGE=ghcr.io/natural-machine/biom3:cuda-2066a75
export BIOM3_BIND_EXTRA=/data/data-share,/data/biom3_data
R="docker/run.sh"; LAUNCH=""
JSONL=data/SH3_caption_fields_all.jsonl
mkdir -p outputs/validation/logs
```

`BIOM3_BIND_EXTRA` is needed because this checkout's `weights/` holds absolute symlinks
into `/data/data-share` and `/data/biom3_data`; a link inside a mount resolves inside the
container, so both have to be mounted at the same path. To see where yours point:
`find weights -type l -exec readlink {} + | cut -d/ -f1-4 | sort | uniq -c`.

```bash
# --- mac: local Mac, CPU ------------------------------------------------
export ROW=mac NGPU=1
export BIOM3_IMAGE=ghcr.io/natural-machine/biom3:cpu-3e6d7ab
export BIOM3_WEIGHTS_BUNDLE=run1_base BIOM3_AS_ROOT=1
R="docker/run.sh"; LAUNCH=""
mkdir -p outputs/validation/logs
```

`run.sh` passes no GPU flags for a `:cpu-*` tag. The bundle is pulled into the image's own
`/app/weights` at start, which needs root, hence `BIOM3_AS_ROOT=1`; outputs are then owned
by root, so the ownership part of pass criterion 3 does not apply to this row. Only `emb`
and `gen` apply here, both with `--device cpu` (see [row notes](#row-notes)).

```bash
# --- au1: Aurora, 1 node, 12 tiles --------------------------------------
# from a `qsub -I` shell on a compute node
export ROW=au1 NGPU=12
export BIOM3_IMAGE=/flare/NLDesignProtein/$USER/biom3_xpu.sif
export BIOM3_BIND_EXTRA=/lus
R="scripts/aurora/apptainer_run.sh"
LAUNCH="scripts/launchers/container_singlenode.sh"
JSONL=<path to the JSONL on /flare>       # see open item 3
mkdir -p outputs/validation/logs
```

```bash
# --- po1: Polaris, 1 node, 4 A100 ---------------------------------------
# from a `qsub -I` shell on a compute node
export ROW=po1 NGPU=4
export BIOM3_IMAGE=<path>/biom3_cuda.sif
R="scripts/polaris/apptainer_run.sh"
LAUNCH="scripts/launchers/container_singlenode.sh"
JSONL=<path to the JSONL on /grand>       # see open item 3
mkdir -p outputs/validation/logs
```

See open item 2 before the `s1`, `pt` and `ft` cells on this row: they are expected to
fail, and `BIOM3_LAUNCHER=container` is the likely workaround.

### Cell commands (single node: `sp1`, `mac`, `au1`, `po1`)

Identical on every single-node row once its preamble has run. The training commands pass
`--wandb False`: without it a `WANDB_API_KEY` in your environment is forwarded into the
container and every smoke run is uploaded to the team W&B project.

```bash
# emb
$R $LAUNCH biom3_embedding_pipeline \
    -i tests/_data/stage1_inputs/sample_text_seqs1.csv \
    -o outputs/validation/$ROW-emb --prefix emb \
    --weight_set configs/weights/run1_base.json \
    --pencl_config configs/inference/stage1_PenCL.json \
    --facilitator_config configs/inference/stage2_Facilitator.json \
    2>&1 | tee outputs/validation/logs/$ROW-emb.log

# s1
$R scripts/stage1_train_singlenode.sh \
    configs/stage1_training/pretrain_scratch_v1.json $NGPU auto $ROW-s1 \
    --data_path tests/_data/stage1_inputs/sample_text_seqs1.csv \
    --epochs 1 --batch_size 2 --valid_size 0.5 --wandb False \
    --output_root outputs/validation/$ROW-s1 \
    2>&1 | tee outputs/validation/logs/$ROW-s1.log

# pt
$R scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json $NGPU auto $ROW-pt \
    --primary_data_path tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5 \
    --epochs 1 --limit_val_batches 1.0 --wandb False \
    --output_root outputs/validation/$ROW-pt \
    2>&1 | tee outputs/validation/logs/$ROW-pt.log

# ft
$R scripts/stage3_train_singlenode.sh \
    configs/stage3_training/finetune_v1.json $NGPU auto $ROW-ft \
    --finetune True \
    --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.bin \
    --primary_data_path tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5 \
    --epochs 1 --limit_val_batches 1.0 --wandb False \
    --output_root outputs/validation/$ROW-ft \
    2>&1 | tee outputs/validation/logs/$ROW-ft.log

# gft  — --device auto is broken here; see open item 8
$R $LAUNCH biom3_finetune_stage3 \
    --config_path configs/stage3_training/finetune_generalized_v1.json \
    --finetune_data_path $JSONL \
    --device auto --num_nodes 1 --devices_per_node $NGPU --run_id $ROW-gft \
    --epochs 1 --limit_train_batches 4 --limit_val_batches 2 --wandb False \
    --output_root outputs/validation/$ROW-gft \
    2>&1 | tee outputs/validation/logs/$ROW-gft.log

# gen
$R $LAUNCH biom3_ProteoScribe_sample \
    -i tests/_data/embeddings/test_Facilitator_embeddings.pt \
    -c configs/inference/stage3_ProteoScribe_sample.json \
    -m weights/ProteoScribe/run1_base_proteoscribe.bin \
    -o outputs/validation/$ROW-gen/generated.pt --fasta --seed 42 \
    2>&1 | tee outputs/validation/logs/$ROW-gen.log
```

On the `mac` row, add `--device cpu` to the `emb` and `gen` commands and skip the rest.

### Aurora, 2 nodes (`au2`)

`apptainer_mpi_run.sh` starts one rank per process, so the entry points are called
directly, never through the `scripts/stage*_{single,multi}node.sh` wrappers, which would
spawn ranks a second time.

```bash
# from a 2-node `qsub -I` shell
export ROW=au2 NGPU_PER_NODE=12 NGPU_TOTAL=24
export BIOM3_IMAGE=/flare/NLDesignProtein/$USER/biom3_xpu-oneapi.sif
export BIOM3_FABRIC_DIR=/opt/cray/libfabric/1.22.0/lib64 BIOM3_FI_PROVIDER=cxi
export BIOM3_BIND_EXTRA=/lus
R="scripts/aurora/apptainer_mpi_run.sh"
MN="--device auto --num_nodes 2 --devices_per_node 12"
CSV=<path to the larger CSV>              # see open item 4
JSONL=<path to the JSONL on /flare>       # see open item 3
mkdir -p outputs/validation/logs
```

```bash
# emb
$R biom3_embedding_pipeline \
    -i $CSV -o outputs/validation/$ROW-emb --prefix emb \
    --weight_set configs/weights/run1_base.json \
    --pencl_config configs/inference/stage1_PenCL.json \
    --facilitator_config configs/inference/stage2_Facilitator.json \
    2>&1 | tee outputs/validation/logs/$ROW-emb.log

# s1
$R biom3_train_stage1 \
    --config_path configs/stage1_training/pretrain_scratch_v1.json $MN \
    --run_id $ROW-s1 --data_path $CSV \
    --epochs 1 --batch_size 2 --valid_size 0.5 --wandb False \
    --output_root outputs/validation/$ROW-s1 \
    2>&1 | tee outputs/validation/logs/$ROW-s1.log

# pt
$R biom3_train_stage3 \
    --config_path configs/stage3_training/pretrain_scratch_v1.json $MN \
    --run_id $ROW-pt \
    --primary_data_path tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5 \
    --epochs 1 --limit_val_batches 1.0 --wandb False \
    --output_root outputs/validation/$ROW-pt \
    2>&1 | tee outputs/validation/logs/$ROW-pt.log

# ft
$R biom3_train_stage3 \
    --config_path configs/stage3_training/finetune_v1.json $MN \
    --run_id $ROW-ft --finetune True \
    --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.bin \
    --primary_data_path tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5 \
    --epochs 1 --limit_val_batches 1.0 --wandb False \
    --output_root outputs/validation/$ROW-ft \
    2>&1 | tee outputs/validation/logs/$ROW-ft.log

# gft  — --device auto is broken here; see open item 8
$R biom3_finetune_stage3 \
    --config_path configs/stage3_training/finetune_generalized_v1.json $MN \
    --run_id $ROW-gft --finetune_data_path $JSONL \
    --epochs 1 --limit_train_batches 4 --limit_val_batches 2 --wandb False \
    --output_root outputs/validation/$ROW-gft \
    2>&1 | tee outputs/validation/logs/$ROW-gft.log

# gen
$R biom3_ProteoScribe_sample \
    -i tests/_data/embeddings/test_Facilitator_embeddings.pt \
    -c configs/inference/stage3_ProteoScribe_sample.json \
    -m weights/ProteoScribe/run1_base_proteoscribe.bin \
    -o outputs/validation/$ROW-gen/generated.pt --fasta --seed 42 \
    2>&1 | tee outputs/validation/logs/$ROW-gen.log
```

For `au1` run with the `xpu-oneapi` image (see [row notes](#row-notes)), use this same
block with `NGPU_PER_NODE=12 NGPU_TOTAL=12`, `MN="--device auto --num_nodes 1
--devices_per_node 12"` and `ROW=au1-oneapi`.

### Mithril, 1 node (`mi1`)

The command goes inside `--env CMD=`, and the cluster is torn down when `CMD` ends, so the
listing at the end of `CMD` is what records the outputs in the log. `--gpus` is the
per-node count; substitute a real accelerator (for example `A100:1`).

```bash
export ROW=mi1 NGPU=1
mkdir -p outputs/validation/logs

# emb — substitute each cell's command from the single-node block above,
#       with $LAUNCH empty (NGPU=1) and paths under /app.
scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml val-$ROW-emb \
  --gpus A100:$NGPU \
  --config mithril.limit_price=6.00 \
  --env BIOM3_WEIGHTS_BUNDLE=run1_base \
  --env CMD="biom3_embedding_pipeline \
      -i tests/_data/stage1_inputs/sample_text_seqs1.csv \
      -o /app/outputs/validation/$ROW-emb --prefix emb \
      --weight_set configs/weights/run1_base.json \
      --pencl_config configs/inference/stage1_PenCL.json \
      --facilitator_config configs/inference/stage2_Facilitator.json \
      && ls -lR /app/outputs/validation/$ROW-emb" \
  2>&1 | tee outputs/validation/logs/$ROW-emb.log
```

### Mithril, 2 nodes (`mi2`)

Same launcher plus `--num-nodes 2`; `--gpus` stays the per-node count. The nodes share no
filesystem, so `--distributed_strategy ddp` is required (DeepSpeed's per-rank optimizer
shards could not be consolidated), only the head node's `/app/outputs` holds the
checkpoint, and anything not baked into the image has to be fetched onto every node at the
start of `CMD`.

```bash
export ROW=mi2 NGPU=1
mkdir -p outputs/validation/logs

# pt — swap the config, run id and overrides for the other cells
scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml val-$ROW-pt \
  --num-nodes 2 --gpus A100:$NGPU \
  --config mithril.limit_price=6.00 \
  --env BIOM3_WEIGHTS_BUNDLE=run1_base \
  --env CMD="bash scripts/stage3_train_multinode.sh \
      configs/stage3_training/pretrain_scratch_v1.json 2 $NGPU auto $ROW-pt \
      --distributed_strategy ddp \
      --primary_data_path tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5 \
      --epochs 1 --limit_val_batches 1.0 --wandb False \
      --output_root /app/outputs/validation/$ROW-pt \
      && ls -lR /app/outputs/validation/$ROW-pt" \
  2>&1 | tee outputs/validation/logs/$ROW-pt.log
```

The wrapper for each cell, with the same positional signature
`CONFIG_PATH NUM_NODES NGPU_PER_NODE DEVICE RUN_ID` and the overrides from the single-node
block:

| Cell | Wrapper and config |
| ---- | ------------------ |
| `s1` | `scripts/stage1_train_multinode.sh configs/stage1_training/pretrain_scratch_v1.json` |
| `pt` | `scripts/stage3_train_multinode.sh configs/stage3_training/pretrain_scratch_v1.json` |
| `ft` | `scripts/stage3_train_multinode.sh configs/stage3_training/finetune_v1.json` |
| `gft` | `scripts/stage3_finetune_multinode.sh configs/stage3_training/finetune_generalized_v1.json` |

`gen` has no wrapper, so it runs under the multi-node launcher directly:

```bash
  --env CMD="NGPU_PER_NODE=$NGPU bash scripts/launchers/container_multinode.sh \
      biom3_ProteoScribe_sample \
      -i tests/_data/embeddings/test_Facilitator_embeddings.pt \
      -c configs/inference/stage3_ProteoScribe_sample.json \
      -m weights/ProteoScribe/run1_base_proteoscribe.bin \
      -o /app/outputs/validation/$ROW-gen/generated.pt --fasta --seed 42 \
      && ls -lR /app/outputs/validation/$ROW-gen"
```

## Pass criteria

Every cell:

1. Exit status 0, with no traceback in the log.
2. The log shows the expected backend (`cuda`, `xpu` or `cpu`) and world size (nodes × devices per node).
3. Outputs land under `outputs/validation/<row>-<column>/` on the host (on Mithril, in the logged listing). For `docker/run.sh` rows, the caller owns them, not root.

Per column:

| Column | Criteria |
| ------ | -------- |
| `emb` | `emb.PenCL_emb.pt`, `emb.Facilitator_emb.pt` and `emb.compiled_emb.hdf5` exist. The HDF5 has one row per input row. With more than 1 rank: same rows, in input order, with no duplicates. |
| `s1`, `pt`, `ft`, `gft` | At least one optimizer step and one validation pass, with finite training and validation loss. A checkpoint is written under `checkpoints/<row>-<column>/` and run artifacts (`args.json`, `run.log`) under `runs/<row>-<column>/`. With more than 1 node, rank 0 writes the checkpoint once. |
| `ft` | The log shows the `run1_base` weights loaded with no missing or unexpected keys, and the trainable parameter count matches the finetune flags. |
| `gft` | The log shows captions composed from the records and z_c computed on the device, not read from a file. |
| `gen` | 25 sequences (5 prompts × 5 replicas) in `generated.pt` and the FASTA, containing only amino-acid letters. With more than 1 rank: the same sequences as a 1-rank run on the same machine with `--seed 42`. |

## Row notes

- `mac`: the cpu image is for inference only. It has no wandb, tensorboard or mpi4py, and
  training refuses the CPU under `--device auto`, so the training columns are `n/a`.
  There is no multi-node row for the Mac.
- `au1` uses the `xpu` image, and `au2` uses `xpu-oneapi`. The plan is to converge on one
  image, most likely `xpu-oneapi`. To inform that, also run `au1` with the `xpu-oneapi`
  image under `apptainer_mpi_run.sh` (`NGPU_PER_NODE=12 NGPU_TOTAL=12`).
- `sp2`, `po2`: bare-metal multi-node works on both machines. `blocked` means there is no
  container path yet (open item 5).

## Open items

1. The cpu image is stale. `cpu-3e6d7ab` is 44 commits behind HEAD and predates the
   `--device auto` default. Either rebuild and publish cpu from HEAD before the `mac` row,
   or run the `mac` row with `--device cpu` and record the deviation.
2. The Polaris training wrappers are expected to fail (not yet verified). Inside the
   Polaris container, `environment.sh` sets `BIOM3_MACHINE=polaris`, so
   `stage*_train_singlenode.sh` dispatches to `scripts/launchers/polaris_singlenode.sh`.
   That script calls `mpiexec --envall --ppn`, which are flags for the host's Cray MPI
   launcher. The image only has OpenMPI's `mpiexec`, which does not accept them.
   `scripts/aurora/apptainer_run.sh` avoids this by setting `BIOM3_LAUNCHER=container`,
   but `scripts/polaris/apptainer_run.sh` does not. Record the plain command's result
   first. If it fails, `BIOM3_LAUNCHER=container` on the host is the likely workaround,
   since Apptainer passes the host environment through.
3. There is no JSONL input for `gft` in the image. `data/SH3_caption_fields_all.jsonl`
   (56 MB) exists only in the BioM3-dev checkout on Spark, not in BioM3-data-share. Either
   put a copy on each machine, or add a small JSONL fixture under `tests/_data` so the
   image carries it.
4. The 5-row prompts CSV is too small for `emb` and `s1` with more than a few ranks. Pick
   a larger CSV (at least 2,000 rows, with the columns `stage1_PenCL.json` expects) and
   put it on each machine.
5. There is no container multi-node path yet for:
   - `sp2`: `docker/run.sh` passes neither `--net=host` nor `NUM_NODES`, `NODE_RANK` or
     `MASTER_ADDR`, and `container_multinode.sh` detects the NCCL interface by looking for
     a `10.x` address. The Spark cable is `192.168.100.x`.
   - `po2`: `scripts/polaris/apptainer_run.sh` supports a single node only. Multi-node
     needs the host MPI bound in, as `au2` does.
6. `mi2-emb` is blocked because Stage 1 inference merges per-rank shard files on disk
   (`_merge_rank_shards` in `src/biom3/Stage1/run_PenCL_inference.py`). Mithril nodes
   share no filesystem, so rank 0 cannot see the other nodes' shards.
7. `gft` has no single-node wrapper (only `scripts/stage3_finetune_multinode.sh`), so
   single-node cells call the entry point directly or through `container_singlenode.sh`.
8. **Fixed in source, pending an image rebuild.** `gft` failed with `--device auto` on
   every backend, because `src/biom3/Stage3/run_ProteoScribe_finetuning.py` never called
   `resolve_device` (unlike its two siblings), so the raw string `auto` reached
   `torch.load(..., map_location="auto")`. Fixed on 2026-09-26 by adding the same
   resolve/`check_devices_per_node` block the Stage 3 trainer uses. Verified with the
   published image plus the edited `src/` bind-mounted over `/app/src`: `--device auto`
   resolves to cuda and the run completes in 40 s. `pytest tests/ --quick` is clean
   (1382 passed, 162 skipped), as is `tests/stage3_tests/test_proteoscribe_finetuning.py`
   (21 passed) and the `--dry_run` path. **The published images still carry the bug**, so
   the `gft` column stays `blocked` until all four variants are rebuilt from a commit that
   includes this fix. The fix also gives `gft` the `check_devices_per_node` guard it never
   had, so an over-large `--devices_per_node` now fails early rather than late.
9. The baked z_c inputs predate `run1_base`. `Stage2_MMD_swissprot_embedding_subset_1000.hdf5`
   dates from March 2026 and `test_Facilitator_embeddings.pt` from February 2026, so their
   z_c come from older PenCL/Facilitator weights. As a result, `ft` starts above the
   from-scratch validation loss (5.51 vs 3.45 on `sp1`), and `gen` produces full-length,
   low-complexity sequences. Both still show that the machinery runs, but not that the
   outputs make sense. Regenerating both files with `run1_base` would make these checks
   meaningful.

10. `--max_steps` is silently ignored by the `gft` and `pt`/`ft` smoke commands.
   `train_model` only sets `trainer_params['max_steps']` on the `combine` branch
   ([`src/biom3/Stage3/run_PL_training.py:1744`](../../src/biom3/Stage3/run_PL_training.py#L1744));
   the `primary_only` branch sets `max_epochs` instead and drops `max_steps` on the
   floor. `training_strategy` defaults to `auto`, which resolves to `primary_only`
   whenever there are no `--secondary_data_paths`
   ([`src/biom3/Stage3/run_PL_training.py:1140`](../../src/biom3/Stage3/run_PL_training.py#L1140)), so that is the usual
   case. The flag is still accepted and still advertised in `--help`, so it looks like it
   works. `--limit_train_batches` is applied unconditionally
   ([`src/biom3/Stage3/run_PL_training.py:1748`](../../src/biom3/Stage3/run_PL_training.py#L1748)) and is what the commands
   above now use. Found when the `sp1-gft` `--device cuda` diagnostic ran 870 batches (37
   minutes) instead of the 20 steps requested. Worth deciding whether `max_steps` should
   warn or apply in epoch mode. Not fixed here — reported only.

## Prior evidence

These runs happened before this matrix existed. They are context only: re-run each one
with the standard command before checking it off.

- `sp1-pt`: `docker/run.sh scripts/stage3_train_singlenode.sh ... 1 auto` ran in
  `cuda-2066a75` on the GB10 as uid 1000, and the caller owned the outputs (2026-09-25).
- `sp1-gen`: `run1_base` generation passed in a local cuda build on the GB10 as uid
  1000 (2026-09-25).
- `au1-pt`: 12-tile training worked in `xpu-5c3a0d2` (2026-09-25, run by the user).
- `au2-pt`: 2-node MPI training worked in `xpu-oneapi-5c3a0d2` (2026-09-25, run by the
  user). The user also reported the `2066a75` Aurora images working.

## Results log

Add one line per run, newest last. List any deviation from the standard command.

| Cell | Date | Image tag | Result | Wall time | Deviation / notes |
| ---- | ---- | --------- | ------ | --------- | ----------------- |
| `sp1-gen` | 2026-09-26 | cuda-2066a75 | pass | 7 min | 25 sequences, amino-acid letters only, loaded on cuda. Sequences are ~1,000-residue low-complexity (open item 9). |
| `sp1-emb` | 2026-09-26 | cuda-2066a75 | pass | 17 s | 3 outputs, 5 rows in the HDF5. |
| `sp1-pt` | 2026-09-26 | cuda-2066a75 | pass | 94 s | Ran **without** `--wandb False`, so it synced W&B run `7xcr9yq4` to thenaturalmachine/BioM3-dev. val_loss 3.45 → 1.97 over 25 steps. |
| `sp1-ft` | 2026-09-26 | cuda-2066a75 | pass | 86 s | Strict load of `run1_base`; 50.4M trainable / 35.8M frozen. val_loss 5.51 → 2.56 (open item 9). |
| `sp1-s1` | 2026-09-26 | cuda-2066a75 | pass | 28 s | 1 step; train_loss 3.30, valid_loss 8.83. |
| `sp1-gft` | 2026-09-26 | cuda-2066a75 | fail | 6 s | `RuntimeError: don't know how to restore data location ... (tagged with auto)` from `torch.load` (open item 8). Dataset loaded first: 27,833 train / 6,959 val. |
| `sp1-gft` (fix verified) | 2026-09-26 | cuda-2066a75 + bind-mounted `src/` | pass | 40 s | Not a grid cell: the published image does not yet contain the fix. `--device auto` → cuda, 4/4 train batches, val_loss 0.150 → 0.147, checkpoint at step 4. Re-run as a real cell once the images are rebuilt. |
| `sp1-gft` (`--device cuda`) | 2026-09-26 | cuda-2066a75 | — | 37 min | Diagnostic for open item 8, not a grid cell. Trains normally: 50.4M trainable, val_loss 0.159 → 0.098, checkpoint at step 870. It ran a **full 870-batch epoch, not the 20 steps asked for** — which is how open item 10 was found. |

Reproduction notes for the `sp1` runs above:

- All used the `sp1` preamble exactly as written, so `BIOM3_BIND_EXTRA` covered the
  `weights/` symlinks into `/data/data-share` and `/data/biom3_data`.
- They redirected with `> outputs/validation/logs/<cell>.log 2>&1` rather than `tee`;
  the commands were otherwise character-for-character the blocks above, except `sp1-pt`
  as noted.
- `$JSONL` was `data/SH3_caption_fields_all.jsonl` from the host checkout, mounted at
  `/app/data` (open item 3 — it is not in the image).
- Logs are kept in `outputs/validation/logs/` on spark-nm, which is gitignored.
