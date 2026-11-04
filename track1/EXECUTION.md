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
is being moved to a private dataset with a `.bin` suffix to retain its archive
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

The current scale-search cost is trimmed observed point-to-mesh-vertex distance
plus an occlusion-aware silhouette penalty. This is an unbenchmarked heuristic;
metric depth near thin objects and occlusion can bias it. The hoop's alternative
circular-tube prior fits its automatic mask, without reference geometry. Generic
objects use an image-conditioned Hunyuan3D hypothesis; source code and weights
retain their external licenses. Batch generation selects the highest automatic
seed confidence among available released clips of each object, shares normalized
geometry, and estimates metric scale independently in each episode.

`prepare_mesh.py` removes degenerate generation debris and simplifies real
geometry to 4096 faces/vertices. It checks closed-surface preservation and Euler
characteristic; failures require inspection rather than silently changing topology.
`export_cari4d.py` requires a complete original timeline and verifies native MHR
parameter export by decoding the actual reference MHR model within 0.5 mm.

## Recovery

Run commands from the repository root. Load the already configured Kaggle
credential privately; never print or commit it.

```bash
source .secrets/kaggle.env
.venv/bin/python scripts/track1_queue.py --watch --hours 8
```

Queue state is `outputs/track1_queue.json`; restarting reads it and does not
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
