# Track 1 reconstruction

This is the active track. Track 2/3 execution stays on hold until Track 1 is
implemented, scored on all five leaderboards, and its produced artifacts are
published. See [STATUS.md](STATUS.md) for current verification limits.

## Pipeline under implementation

1. `segment_video.py`: automatic Grounding DINO seed selection and bidirectional
   SAM2 human/object masks, using only Track 1 video and released text prompts.
2. `reconstruct_object.py`: Hunyuan3D geometry from the observed object crop.
   The output is normalized, explicitly requiring metric calibration.
3. Native MHR initialization and human-aligned depth from the pinned official
   CARI4D pipeline; `calibrate_object.py` selects scale/registration hypotheses
   using observed depth and silhouette agreement.
4. Native CARI4D object tracking, CoCoNet and contact refinement.
5. `export_cari4d.py`: shared per-clip identity, native MHR parameter decoding
   checks, complete original-frame coverage, actual object mesh/poses.
6. `submit.py`: official full-roster packing, preflight, dry run and optional
   upload to all five metric competitions, with a real producing commit URL.

The local scripts have syntax checks. Actual reconstruction and model adapter
validation are pending. No Track 1 scoring artifact has been produced yet.

## Artifacts

Our reconstructed episode arrays, object meshes, previews, diagnostic reports,
submission Parquet, score receipts and reproduction instructions belong in
GitHub Releases after verification. Source code and manifests belong in Git.
Third-party pretrained checkpoint files retain their original licenses and are
referenced by source revision/hash. The user approved an explicitly private
Kaggle compute copy; no token or login/cache files are included in it.

## Runtime

`notebooks/track1_runtime/` builds the official PyTorch 2.5.1/CUDA 12.4 stack
for T4's SM 7.5. NVIDIA redistributable compiler archives are hash-verified.
`notebooks/track1_segmentation/` runs the first automatic segmentation experiment.
These are development runs, not completed competition submissions.
