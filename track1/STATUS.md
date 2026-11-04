# Track 1 execution status

Active scope: Track 1 only. Do not start further Track 2/3 work until Track 1 has
been implemented, submitted to all five metric competitions, and its produced
artifacts and reproducibility instructions have been published to GitHub.

Track 2 has no active GPU job; the latest G1 probe ended with an error. Previous
retargeting outputs remain preserved for later use.

Track 1 requirements: reconstruct all 30 videos into native MHR parameters,
object geometry and continuous poses. Pack with the official kit and a real
producing commit; upload the same validated Parquet to all five leaderboards,
then poll and retain score receipts. Publish our generated artifacts in GitHub
Releases and the source/report in Git. Reference third-party checkpoint sources
and hashes without adding tokens or gated model files to Git.

Current work: a T4-specific build of the official native-MHR CARI4D dependencies
is running under the private reconstruction notebook. Video preprocessing,
object mesh reconstruction, full inference and scored outputs remain unfinished.
No Track 1 submission exists yet.

Update: all 30 Track 1 videos are local. The first automatic segmentation run is
propagating valid finite masks; FP32 throughput is about 1.4 seconds/frame. The
native runtime initially lacked CUDA sparse/BLAS headers; the matching Torch
CUDA headers were added and a rebuild is running. The checkpoint archive (8
explicit files, 4,889,774,080 bytes) was approved for private Kaggle upload by the
user and is uploading. No credentials are in the archive. Native MHR export and
full-roster official submission orchestration code have been written but remain
to be validated with actual reconstructed episodes.
