# Track 1 execution status

Active scope: complete Track 1's five Kaggle submissions and publish the actual
producing artifacts. Track 2/3 remain on hold. No scored Track 1 submission exists yet.

Verified:
- All 30 original videos and metadata are available. Automatic segmentation
  completed all 30 timelines. Visual and automatic semantic audits identified
  wrong-object candidates; eight affected videos are being corrected automatically.
- The T4 native-MHR runtime passes CUDA nearest-neighbour, rendering and CARI4D
  import checks. Its verified private runtime dataset is ready. Restoring uv's
  unversioned Python alias fixes the cached interpreter link.
- The approved private CARI4D/SAM3D checkpoint dataset is ready and file hashes pass.
  CPU model preparation completed and its 1.91 GB public-model cache is reusable.
- The independent native MHR export diagnostic passed: maximum translation error
  1.43e-6 m and decoder round-trip error 1.57e-6 m. Synthetic fixture arrays were
  removed and never used as challenge reconstructions.
- Eight object classes have actual image-conditioned meshes, packaged for 24
  episodes in a private compute dataset. The erroneous tripod/curtain shape
  candidates are excluded by automatic semantic margins. The iron is pending.
  Budget simplification warnings are retained; the official format does not
  require exact preservation of generated topology.
- Initial hoop masks/geometry and runtime reports are published in the explicitly
  unscored development release on GitHub.

Running:
- `v2d-track1-episode-000000` version 3: actual pilot reconstruction. Depth
  estimation passed; native SAM3D human fitting is progressing across the full clip.
  MoGe uses one device per episode to avoid a confirmed threaded CUDA lazy-init race.
- `v2d-track1-segments-semantic`: Grounding DINO proposals reranked by SigLIP2
  against released object classes/distractors; bidirectional SAM2 masks follow.
- `v2d-track1-crop-ranking` completed: automatic semantic consistency audit for
  all 30 original source crops. Recovered geometry selection uses these scores.

Prepared, not yet validated end-to-end:
- Iron reconstruction from the best positive-semantic source view.
- Two-device reconstruction batches with full original-frame coverage, observed
  triangle-surface scale calibration, clip-shared identity before contact refinement,
  native MHR export and retention of actual model outputs/reports/previews.

Still required: complete real reconstructions for all 30 episodes; official packing,
preflight and dry run; producing GitHub commit; uploads and score receipts for all
five metrics; publish actual final artifacts and reproduction instructions. No
placeholder trajectories or synthetic diagnostic fixtures enter the scoring roster.
