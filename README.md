# V2D Challenge experiments

Work in progress for NVIDIA's [Video to Data Challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/).
Track 1 reconstruction is the active task. Its 30-video pipeline uses automatic
segmentation, observed-image object reconstruction, native MHR human fitting,
depth-based metric calibration, object tracking and contact refinement on Kaggle
T4 GPUs. Production reconstruction is running; no scored submission exists yet.
Track 2/3 are on hold until Track 1 is submitted and its artifacts are published.

See [current Track 1 status](track1/STATUS.md), [execution and recovery](track1/EXECUTION.md),
and [method](reports/track1/method.md). Earlier all-track infrastructure experiments
are documented in [the original audit](reports/approach.md).

## Verified

- All 23 Kaggle metric competitions entered under `priyamsaha17`.
- Two-device T4 native runtime, approved private checkpoint access and hash checks.
- Full original-frame segmentation for all 30 Track 1 videos, including eight
  automatic semantic repairs.
- Actual image-conditioned geometry for nine non-hoop classes and mask-fitted
  hoop geometry. All 27 non-hoop mesh hashes and face/vertex budgets pass checks.
- Independent native MHR export diagnostic, with decoder round-trip error below
  0.5 mm; no diagnostic fixture is used for scoring.
- Lossless depth-cache PNG encoding speedup, preserving decoded depth values.

Actual 30-episode reconstruction coverage, official packing and scoring remain
pending. Runtime smoke tests and input checks do not establish leaderboard quality.

## Local setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy scipy pandas pyarrow nbformat kaggle huggingface_hub
.venv/bin/python -m unittest discover -s tests -v
```

Download the [official submission kit](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip)
into `references/submission_kit/v2d_submission_kit/`. Evaluation uses the official
runtime revision `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`.

Kaggle notebooks and their generators live under `notebooks/` and `scripts/`.
The training integration notebook downloads the public runtime and example assets
from NVIDIA and verifies Git LFS file hashes before use. Each notebook is private
on Kaggle during development.

## Submission requirements

Generate reconstructions and actual policy rollouts from each track/tier's permitted
inputs. Package them with the official kit, record the producing commit URL, retain
checkpoints/evaluator hashes, run preflight, then upload and retain scoring receipts.
The current notebooks do not generate a complete challenge submission.

Store credentials in ignored local storage or Kaggle Secrets. Do not add tokens,
gated model weights or challenge data to Git. Third-party models and dependencies
retain their own licenses; this repository's license covers its original code.

## License

Apache-2.0; see [LICENSE](LICENSE).
