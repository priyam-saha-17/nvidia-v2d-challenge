# V2D Challenge experiments

Work in progress for NVIDIA's [Video to Data Challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/).
This repository currently contains infrastructure checks and an experimental trajectory
refiner. It has no scored submission or demonstrated leaderboard improvement yet.

## Verified

- All 23 Kaggle metric competitions entered under `priyamsaha17`.
- Kaggle 2× T4 GPU computation and dataset access.
- Isolated Python 3.11 installation of the pinned FlashCHORD runtime; JAX GPU
  computation, Warp CUDA availability, Newton and FlashCHORD imports passed.
- Three synthetic trajectory-refinement tests.

Actual tissue-box simulation/training completed 131,072 environment steps and saved
policy checkpoints. A 32-frame G1 retargeting probe passed on actual Tier 1 input;
complete Tier 1 and Tier 2 retargeting batches are running. Model reconstruction,
retargeting of the complete challenge roster, policy training and scored submissions
remain unfinished. See [approach and status](reports/approach.md).

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
