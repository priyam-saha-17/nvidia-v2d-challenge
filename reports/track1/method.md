# Track 1 implemented method

Status: production execution in progress; no held-out scores or ranking claims.
This report describes the implemented reconstruction, rather than the earlier
unimplemented all-track research proposal.

## Problem and permitted inputs

Estimate a human body and a moving object from each released third-person RGB
video. Each output contains native MHR pose parameters on the original timeline,
one body identity per clip, a reconstructed object mesh, metric scale, and object
rotation/translation for every frame. The official kit selects the exact scored
rows and packs one Parquet for CD-H, CD-O, ACC-H, ACC-O and PEN. All are lower better.

The method uses Track 1 videos, released text prompts/object labels, and pretrained
models. It does not consume Track 2 meshes, reference camera parameters, held-out
poses or reference human/object geometry.

## Reconstruction

Grounding DINO proposes human and object boxes at automatically sampled frames.
For semantically ambiguous clips, SigLIP2 ranks object crops against the target,
other released classes and distractors. A positive class margin is required.
Bidirectional SAM2 propagation produces human/object masks over every original
frame. An automatic class audit and visual review identified eight clips requiring
semantic repair; corrected masks replace their original candidates.

Hunyuan3D-2 reconstructs normalized non-hoop geometry from automatically selected
object observations. Generated candidates are selected by positive semantic class
margin. The best accepted geometry is shared among the three videos of the same
released object; metric calibration is estimated independently per episode.
Degenerate debris is removed and actual generated surfaces are simplified to the
official 4096-face/4096-vertex limits. Topology changes are recorded as warnings;
no bounding box, convex hull or unrelated mesh replaces a generated object.
For hoops, an ellipse fit and observed mask thickness estimate a circular-tube
shape prior. Its metric scale and 3D trajectory are subsequently estimated.

The pinned official native-MHR CARI4D route initializes SAM 3D Body human parameters
and MoGe2 depth/intrinsics. Human-aligned metric depth supplies the reconstruction
gauge. FoundationPose registers generated mesh hypotheses at an automatically
selected observation. A coarse/fine scale search scores trimmed observed
point-to-triangle-surface distances and occlusion-aware silhouette agreement.
The calibrated mesh is then used for register-first-then-track initialization
across the complete video, without a reference first-frame rotation.

CoCoNet estimates native MHR and object residuals and contact predictions. Before
contact refinement, per-frame identity predictions are projected to their per-clip
median and facial expression is set to zero, matching the challenge schema.
The raw network output is retained separately. Native contact refinement uses
300 optimization steps, contiguous 32-frame batches, human-pose prior weight 200,
temporal weight 100, and a 5 cm initial contact activation threshold. These are
implementation settings, not claimed optimal values.

The exporter decodes the actual MHR reference model to verify the final native
parameters within 0.5 mm. It rejects incomplete timelines, nonfinite values,
invalid object rotations and unexpected mesh coordinate transforms. Actual NPZ,
GLB, native prediction bundles and source identities are retained.

## Compute and recovery

Each episode uses an isolated process on one Tesla T4. Two independent episodes
run concurrently per notebook; this avoids a threaded multi-device CUDA lazy-init
failure observed in MoGe. SAM 3D Body uses FP32 because T4 lacks BF16 support.
Runtime and model caches are revision/hash verified before use.

Depth-cache compression changes only the 16-bit PNG encoding options from maximum
compression with optimization to level 1 without optimization. A numeric codec
check confirms identical decoded pixels. A synthetic 1152×1536 codec microbenchmark
measured approximately 8.4× faster encoding; this is not an end-to-end accuracy
or throughput claim. Actual production depth alignment also completes substantially
faster than the original pilot.

The cached source checkout initially contained Git LFS pointers for the public MHR
collision proxy and hand-surface specification. The corrected bootstrap retrieves
both pinned assets and checks their exact LFS byte counts and SHA-256 hashes.
Cached genuine CoCoNet outputs can then be refined/exported without recomputing
body fitting or substituting trajectories. Recovery failures remain explicit.

## Validation and limitations

The independent native exporter diagnostic and all 27 non-hoop source mesh budget,
hash and finite-geometry checks pass. The collision proxy is closed and both public
topology assets identify the same MHR reference model. Official packing, full-roster
validation, Kaggle scores and the required final reproduction container remain
separate completion gates. No ablation or leaderboard improvement is established.

Image-conditioned geometry can be wrong in occluded regions, depth near thin
objects can bias metric calibration, and generated desk/cart meshes retain recorded
topology warnings. Per-clip identity projection and the scale objective are
unbenchmarked implementation choices. Held-out scoring is required to assess them.

## Primary implementation sources

- [Official runtime, pinned revision](https://github.com/nvidia-isaac/video_to_data/tree/8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef/reconstruction)
- [CARI4D](https://arxiv.org/html/2512.11988v4)
- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)
- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- [MoGe](https://github.com/microsoft/MoGe)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [SigLIP2 implementation documentation](https://huggingface.co/docs/transformers/model_doc/siglip2)

Third-party sources and pretrained weights retain their original licenses. Gated
checkpoint files are kept in the user-approved private Kaggle compute dataset.


## Additional candidate: shared translation refinement

`filter_common_translation.py` prepares a separate candidate from the complete
actual exports. It smooths the mean human-root/object translation with a
second-difference penalty and a fixed first frame. The resulting correction is
applied to the human and object together. A 5 cm maximum correction bound reduces
its gain on clips with large proposed changes. Body pose controls, rotations,
identities, mesh and object scale stay exactly unchanged.

A shared translation preserves hand/object geometry, and fixing the first frame
preserves the scorer's alignment. Native MHR root translation units and coordinate
axes must still be checked numerically against the actual reference model on real
frames. `check_common_translation.py` performs that check. The submission wrapper
requires its successful report and matching artifact hashes for this candidate.

The solver preserves constant velocity and reduces injected observation jitter in
an in-memory numerical check; no fixture scoring files are generated. This
candidate has not yet been applied to the full actual roster or scored. Lower
trajectory acceleration energy alone does not establish lower held-out error.
