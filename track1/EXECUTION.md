# Track 1 execution and recovery

Only Track 1 jobs are authorized in the current execution stage. The prepared
queue never uploads scoring files and never starts Track 2/3. It uses at most
two GPU notebook sessions; the user requested Kaggle Tesla T4.

## Current producing implementation

The native dependency smoke test passed in
`priyamsaha17/v2d-reconstruction-checkpoint-access` version 6. The first actual
reconstruction is `priyamsaha17/v2d-track1-reconstruction`. Its first step builds
and saves a runtime archive, including original third-party source/license files.
Its explicit archive directories exclude credentials and model caches. The run
failed after saving that cache because Kaggle expanded the uploaded checkpoint
tar into `track1_models/`. The loader now handles that actual layout. The cache
was moved to a ready private dataset with a `.bin` suffix to retain its archive
format. Failed notebook outputs cannot be attached as a new notebook input.
`v2d-track1-model-preparation` version 2 completed on CPU and caches public
weights separately; subsequent jobs restore both caches with hash verification.

Approved weights live in the private dataset
`priyamsaha17/v2d-track1-private-pretrained-models`. Each file is verified against
`model_manifest.json` before use. The original SAM3D config requests BF16, which
T4 does not support. The inference notebook records its change to FP32.

The first segmentation notebook has produced every frame for episode 0. Batch A
handles episodes 1–15; batch B handles episodes 16–29. FP16 stored memory matches
autocast precision; a failed run retries FP32 in a new child process. Frame
coverage and finite logits are mandatory. Inspect saved overlays and area trends
before treating all masks as reliable. Propagation can drift despite finite output.

## Pipeline and geometric checks

`run_episode.py` runs native human/depth bootstrap in a separate directory, then
uses one automatically selected observed frame for physical mesh-scale search.
The final export contains the calibrated mesh. Only unchanged human/depth stages
are transferred, with file identities recorded. The native tracker and CoCoNet
then estimate the full trajectory, followed by contact/temporal refinement.

The current scale-search cost is trimmed observed point-to-triangle-surface distance
plus an occlusion-aware silhouette penalty. This is an unbenchmarked heuristic;
metric depth near thin objects and occlusion can bias it. The hoop's alternative
circular-tube prior fits its automatic mask, without reference geometry. Generic
objects use an image-conditioned Hunyuan3D hypothesis; source code and weights
retain their external licenses. Batch generation selects the highest automatic
seed confidence among available released clips of each object, shares normalized
geometry, and estimates metric scale independently in each episode.

`prepare_mesh.py` removes degenerate generation debris and simplifies real
geometry to 4096 faces/vertices. It records closed-surface and Euler-characteristic changes; `--strict-topology`
can reject them. The official fixed-budget packer also uses quadric simplification.
Generated geometry is retained with explicit quality warnings rather than silently
substituting a generic shape. Closed-surface preservation is not a format requirement.
`export_cari4d.py` requires a complete original timeline and verifies native MHR
parameter export by decoding the actual reference MHR model within 0.5 mm.

## Recovery

Run commands from the repository root. Load the already configured Kaggle
credential privately; never print or commit it.

```bash
source .secrets/kaggle.env
.venv/bin/python scripts/track1_queue.py --state outputs/track1_production_queue.json --watch --hours 6
```

Current production queue state is `outputs/track1_production_queue.json`; restarting reads it and does not
resubmit completed/failed notebooks. A failed execution is not retried automatically.
The bounded log utility writes raw logs locally and prints only a tail:

```bash
.venv/bin/python scripts/kaggle_live_snapshot.py priyamsaha17/v2d-track1-reconstruction \
  --seconds 15 --output outputs/track1_inference/live.json
```

After the first reconstruction's cache is available, the cached-inference builder
prepares subsequent episode notebooks. Do not launch those before input notebooks
have completed and their output reports have been reviewed.

## Final submission remains pending

Collect actual episode NPZ/GLB pairs for all 30 episodes. Push the exact producing
code, then invoke `track1/submit.py` with its GitHub commit URL. The official kit
packs the exact scored rows, checks entry/quota in dry run, and sends one file to
all five metrics only with `--upload`. Retain the packed file, kit, receipt and
producing artifacts for container reproduction. Poll the same receipt until all
scores are available. An environment preflight alone does not validate a reconstruction.

The dedicated completion runner handles the frozen production notebook versions,
paginated final-output downloads, collection, official validation and a single
five-metric upload. It reserves the exact official receipt before uploading and
resumes polling its IDs without re-packing a counted file. An ambiguous upload
stops for diagnosis instead of automatically consuming more quota.

```bash
source .secrets/kaggle.env
.venv/bin/python scripts/finish_track1.py --output outputs/track1_completion \
  --commit https://github.com/priyam-saha-17/nvidia-v2d-challenge/commit/f4891dbcd4161c82ab522ab806157133f2c217ab \
  --upload
```

Inspect `outputs/track1_completion/completion_state.json` before any separate
upload. Once its receipt is reserved, use that receipt for recovery. The original
unfiltered submission and the optional translation candidate have separate files
and receipts; a candidate requires its actual reference decoder validation first.

The cached interpreter needs uv's unversioned alias restored to the archived
Python 3.11.18 base. The native MHR numerical diagnostic now passes independently
of image inference. MoGe's automatic multi-device threading triggered PyTorch's
CUDA linear-algebra lazy-init race; use `moge_devices="cuda:0"` per episode and
parallelize independent episodes across devices instead.

New batches retain native prediction bundles, initialization outputs, reports,
meshes and preview videos. Regenerable RGB/depth caches are removed between
episodes unless `--retain-intermediates` is requested, bounding notebook output
storage. The raw network output is preserved when shared per-clip identity and
zero expression are projected into a separate bundle before contact refinement.


## Collect completed production jobs

Download only final episode NPZ/GLB/export reports and the batch report first,
using `kaggle kernels output --file-pattern` to avoid unnecessary cache downloads.
Pass each exact notebook/version and local download directory to
`track1/collect_outputs.py --source NOTEBOOK/VERSION=DIRECTORY --output OUTPUT`.
It records artifact hashes and requires 30 unique successful episodes before
returning success. Retain the complete native bundles and previews separately
for the final GitHub artifacts and required reproduction container.


## Retain native artifacts for GitHub

After both production versions finish with 30 successful episode records, launch
`notebooks/track1_native_archive/` on CPU. It attaches only the actual production
outputs, verifies the allowed prediction PTH filenames and native schemas, and
compresses each episode's predictions, final NPZ/GLB, reports and available
previews. It excludes pretrained checkpoints, runtime caches and credentials.
Download every output page and verify `native_artifact_manifest.json` hashes before
publishing these archives. The job is prepared but has not yet executed.
