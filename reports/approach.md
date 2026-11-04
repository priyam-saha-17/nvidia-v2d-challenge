# V2D competition audit — 4 November 2026

## Objective and current state

Produce valid, reproducible submissions to all 23 metric competitions. Kaggle authentication now works as `priyamsaha17`. Entry is confirmed for 4 of 23
competitions; 19 require rules acceptance (see `rules-to-accept.md`). No competition
submission or policy training has run. A private T4 diagnostic completed successfully:
2× Tesla T4 (15,360 MiB each), driver 580.178.04, CUDA 13.0, PyTorch 2.11.0.
The notebook default is Python 3.13.16. An isolated Python 3.11 runtime installation
probe has been launched. Credentials are stored only in an ignored mode-600 local
file. The repository currently has no GitHub remote.

The official deadline is 4 November 2026, 17:00 EST, or **5 November, 03:30 IST**.
There are five submission groups: Track 1 (5 metrics), Track 2 tiers 1–3
(4 metrics each), and Track 3 (6 metrics). Separate notebooks for every metric
would waste compute: each group reuses the same evaluated result.

## Required outputs

| Group | Required work | Submission |
|---|---|---|
| Track 1 | Reconstruct 30 videos, native MHR bodies and own object meshes/poses | Parquet, five metrics |
| Track 2 tier 1 | Retarget clean input; train and evaluate 30 G1+Dex3 policies | CSV, four metrics |
| Track 2 tier 2 | Recover noisy input without using clean captions; train 30 policies | CSV, four metrics |
| Track 2 tier 3 | Reconstruct input from videos; retarget and train 30 policies | CSV, four metrics |
| Track 3 | Reconstruct objects/motion from 20 ego videos; train and evaluate policies | Combined mesh/pose Parquet, six metrics |

All scored episodes are required. Tracks 2 and 3 require real learned-policy
rollouts, checkpoint hashes, evaluator hashes and a producing GitHub commit URL.
Winning entries additionally require reproducible containers and a technical report.
Track 1 may not use Track 2 meshes or camera parameters. Tier 2/3 must not use
Tier 1 clean trajectories for recovery; the FAQ allows Tier 3 first-frame alignment.
Track 3 requires reconstructed meshes even when a public mesh of an object exists.

## Research-informed approach

1. **Reproduce before extending.** Use the pinned official evaluation runtime and
   establish one complete reconstruction and one simulated policy rollout first.
2. **Reconstruction:** use CARI4D's foundation-model initialization, pose hypotheses
   and contact refinement. Its research implementation uses SMPL-H, whereas Track 1
   requires native MHR; use the official MHR route or an independently verified fitting
   conversion. Never treat SMPL-H parameters as MHR parameters.
3. **Proposed extension (unvalidated):** confidence-weighted trajectory optimization
   on SE(3), with weaker smoothing across inferred contact transitions, followed by
   a physically feasible contact/force check. Keep multiple plausible hypotheses in
   occluded intervals. Choose hypotheses using visible-image agreement, temporal
   consistency and simulated tracking feasibility, never prohibited clean captions.
4. **Control:** FlashCHORD/FlashSAC with residual actions and contact guidance.
   Preserve the official evaluation semantics. Start with a small world count and
   replay buffer on T4, then increase only after measuring memory and throughput.
5. **Ablation:** baseline; temporal refinement; confidence weighting; contact phase
   adaptation; feasibility feedback. Validate reconstruction on allowed external or
   public training data. Use synthetic corruption on separate allowed data for tuning.
   Smoothing alone cannot remove low-frequency random-walk drift.

The implemented `src/trajectory_refinement.py` is only the temporal-refinement
prototype. It is not a reconstruction system, policy, or demonstrated novel result.
The contact/force feasibility loop and image front end remain to be implemented.

## Runtime constraints and immediate sequence

The pinned FlashCHORD recipe requires Python 3.11/3.12, CUDA 13 JAX and driver
580+, and upstream validated full training on a 48 GiB GPU. The default G1 recipe
uses 2,048 simulation worlds and a 10-million-transition replay buffer. The local
launcher reduces these for a **proposed** T4 experiment; it is not GPU validated.
Two T4 GPUs do not automatically pool VRAM. Existing model assets, retargeting,
body-model access and robot-controller assets must all be checked.

Next: restore authorized Kaggle access → verify all 23 entry states and rules →
run the diagnostic notebook → install and test the pinned simulator → obtain
required model assets → run one episode end to end → measure throughput and
budget all sequences → produce, validate, upload and poll all five groups.

## Sources

- [Challenge and deadline](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/)
- [Official FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf)
- [Submission kit](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip)
- [Dataset](https://huggingface.co/datasets/nvidia/video_to_data_challenge/tree/main)
- [CARI4D paper](https://arxiv.org/html/2512.11988v4)
- [CHORD paper](https://arxiv.org/html/2607.00033v1)
- [FlashSAC paper](https://arxiv.org/html/2604.04539v1)
- [Pinned FlashCHORD runtime](https://github.com/nvidia-isaac/video_to_data/tree/8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef/robotic_grounding/flash_chord)
- [Kaggle notebook CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)

## Executed Kaggle notebooks

- [Completed T4 diagnostic](https://www.kaggle.com/code/priyamsaha17/v2d-t4-runtime-diagnostic)
- [Isolated FlashCHORD runtime probe](https://www.kaggle.com/code/priyamsaha17/v2d-flashchord-runtime-probe)

Native MHR CARI4D inference requires approved Hugging Face access to both
`nvidia/cari4d_commercial` and `facebook/sam-3d-body-dinov3`.
