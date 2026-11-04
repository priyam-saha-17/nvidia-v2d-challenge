# Track 1 execution status

As of 2026-11-04 15:32 UTC: Track 1 only. No scored submission exists yet.
Track 2/3 remain on hold.

Completed:
- Original-frame segmentation for all 30 released videos; semantic repair for
  episodes 6, 8, 9, 16, 17, 19, 27 and 28. Corrected crops have positive automatic
  class margins and were visually checked.
- Nine actual image-conditioned non-hoop object reconstructions, including the
  corrected iron from episode 6, packaged for all 27 non-hoop episodes in private
  Kaggle dataset `priyamsaha17/v2d-track1-reconstructed-shapes`, version 2 ready.
  Hoops use their own automatically observed mask-fitted circular tube geometry.
- Hash-verified native T4 runtime, approved private checkpoints, and public model
  cache. Native MHR export numerical diagnostic passes independently of inference.
- Lossless 16-bit depth PNG cache encoding patch: decoded pixels unchanged; local
  codec diagnostic approximately 8.4 times faster. No timeline/resolution changes.
- Public code and unscored development artifacts on GitHub. Pretrained gated
  checkpoint files remain private.

Active Kaggle GPU jobs (account limit: two concurrent sessions):
- `v2d-track1-episode-000000`, version 3: development pilot has completed all
  790 frames through object tracking and CoCoNet; refinement exposed two unresolved
  public Git LFS topology assets. Both are now restored and hash-verified; the
  pilot prediction bundle is being recovered.
- `v2d-track1-inference-hoops`, version 1: production episodes 1 and 2 on separate
  T4 devices, using the lossless cache patch and shared identity before refinement.

Queued production batches:
- `v2d-track1-inference-main-a`: episodes 0 and 3–14.
- `v2d-track1-inference-main-b`: episodes 15–29.
- Dedicated state `outputs/track1_production_queue.json`; the queue launches when
  existing GPU sessions finish. Do not restart the old preprocessing state.

Production main batches include hash-verified hydration of both public runtime
topology assets before inference. The initial hoop batch predates this fix and
will require recovery or rerun before it can produce a valid final export.
`collect_outputs.py` preserves notebook/version and file hashes and refuses
conflicting artifacts. Collection does not replace official packing validation.

Still required: complete real reconstructions for all 30 episodes; official
packing/preflight/dry run; upload the identical Parquet to all five Track 1 metrics;
wait for scores; publish actual final artifacts and receipts. Retain the exact kit,
models and outputs for the subsequent required container reproduction process.
