# Track 1 execution status

Active scope: Track 1 only. Track 2/3 execution remains on hold until Track 1
has actual reconstructions, submissions to all five metric competitions and
published artifacts. No Track 1 scored submission exists yet.

## Verified

- All 30 Track 1 videos and their released metadata are downloaded.
- The official native-MHR runtime passes on Tesla T4: PyTorch 2.5.1/CUDA 12.4,
  PyTorch3D GPU nearest-neighbours, nvdiffrast CUDA rendering context and CARI4D import.
- Episode 0 automatic segmentation completed all 790 original frames. Preview
  overlays and alpha topology were inspected; the hoop mask has an open centre.
- The approved private Kaggle model dataset is ready. It contains eight explicit
  CARI4D/SAM3D model/config/license files, no credentials. Public Git stores hashes.
- A circular-tube prior was fitted to the observed hoop mask (2048 vertices,
  4096 faces). Its scale and 3D pose still require observed-depth calibration.

## Running / pending

- `v2d-track1-object-shape` completed. Its real hoop geometry was simplified to
  4096 faces/2048 vertices, preserving a watertight open-centre surface.
- First generated artifacts are published at
  https://github.com/priyam-saha-17/nvidia-v2d-challenge/releases/tag/track1-development-20261104
  and explicitly marked unscored development outputs.
- `v2d-track1-reconstruction` passed runtime validation and saved its 4.81 GB
  cache, then failed because Kaggle had expanded the checkpoint tar automatically.
  The loader is fixed; the verified runtime is being recovered into a private
  dataset because failed notebook outputs cannot be attached to another notebook.
- `v2d-track1-model-preparation` version 2 completed on CPU: approved checkpoint
  integrity checks passed and public models were saved in a 1.91 GB reusable cache.
- `v2d-track1-episode-000000` is prepared for the cached reconstruction retry.
  Human/depth bootstrap, metric object calibration, CARI4D inference and native
  MHR export remain to run. A CPU exporter diagnostic is also prepared.
- Both remaining-video segmentation batches are running. Episodes 1–6 and
  16–19 have passed in FP16 (plus the previously verified episode 0).
  Remaining-video segmentation batches use FP16 with matching SAM2 memory dtype,
  finite output checks and FP32 retry on failure. Throughput is about 2.2 frames/s;
  additional per-video visual quality review remains pending.
- No actual episode reconstruction has finished. Exporter numerical validation,
  full-roster packing, official preflight/dry run, five uploads, score receipts
  and publication of produced artifacts are still pending.

The first reconstruction separates bootstrap and final exports. Its stage reuse
records actual source/output identities; it never substitutes reference poses.
T4 SAM3D uses FP32 because the original checkpoint config requests unsupported BF16.

Generated reconstructions, meshes, previews, reports, final Parquet and receipts
will be published in GitHub Releases after validation. Third-party checkpoint
files remain in private Kaggle storage under their original licenses.
