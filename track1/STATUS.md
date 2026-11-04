# Track 1 execution status

As of 2026-11-04 16:28 UTC: Track 1 only. Four of 30 actual episode exports
have completed remotely and passed native MHR decoder checks. No scored submission
exists. Track 2/3 remain on hold. The official deadline is 22:00 UTC today.

Completed preprocessing:
- Original-frame segmentation for all 30 videos; semantic repairs for episodes
  6, 8, 9, 16, 17, 19, 27 and 28. Corrected crops pass automatic class checks.
- Nine actual image-conditioned non-hoop reconstructions, including corrected
  iron from episode 6. All 27 non-hoop mesh hashes, finite geometry and budgets pass.
- Ready private Kaggle runtime, approved pretrained checkpoints, public model
  cache, and reconstructed geometry dataset version 2.
- Lossless depth PNG encoding patch and exact-hash hydration of both public MHR
  topology assets. The original LFS-pointer failure is resolved; production
  refinement and export have now completed successfully.
- Authoritative masks/crops and prepared geometry published in the explicitly
  unscored GitHub preprocessing release. See reports/track1/preprocessing_archive.json.

Actual completed native exports:
- Episodes 1 and 2: recovered real CoCoNet predictions, completed 300-step native
  contact refinement and validated native export.
- Episodes 6 and 11: full corrected production pipeline including previews/export.
- These outputs are retained in the active notebooks and will be downloaded after
  the complete jobs finish. Full-roster official packing remains mandatory.

Active Kaggle jobs (two concurrent GPU sessions, each with two isolated T4 workers):
- `priyamsaha17/v2d-track1-inference-main-a`, version 1: episodes 0 and 3–14.
  Producing notebook source commit aadf1fb5b9e64635c958484f7fb6ae51b0c48e16.
- `priyamsaha17/v2d-track1-inference-main-b`, version 1: recovered episodes 1/2,
  then episodes 15–29. Source commit f4891dbcd4161c82ab522ab806157133f2c217ab.
- Current queue state: outputs/track1_production_queue.json. Both jobs are started;
  do not reset the queue or relaunch the old preprocessing jobs.

The final roster is main-a (13) + main-b (15) + main-b/recovered (2). The original
pilot version 3 and original hoop job stopped at missing topology assets; their
successful earlier stages were retained. Their COMPLETE/ERROR notebook status alone
is not a complete trajectory submission.

Optional candidate prepared, not applied or scored:
- Shared human/object translation smoothing, fixed first frame, bounded correction.
- Its actual reference MHR decoder check is mandatory before submission; the upload
  wrapper checks the report and artifact hashes. Numerical solver checks alone do
  not establish accuracy. It will not replace the original outputs.

Remaining: finish all 30 actual exports; download/collect with notebook-version
provenance; official packing, preflight and dry run; upload the identical Parquet to
all five metrics; poll exact receipt IDs; publish final NPZ/GLB, native artifacts,
Parquet and receipts on GitHub. Retain exact kit/models/output hashes for the
subsequent required offline container reproduction process.
