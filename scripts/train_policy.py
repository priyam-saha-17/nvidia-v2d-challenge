"""Launch the official trainer with an experimental memory-reduced configuration.

Requires a correctly retargeted reference and the pinned runtime's dependencies
and assets. This does not convert raw challenge parquets into robot references.
"""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--embodiment', choices=['g1', 'sharpa'], required=True)
p.add_argument('--steps', type=int, default=1_000_000)
p.add_argument('--worlds', type=int, default=128)
p.add_argument('--dry-run', action='store_true')
a = p.parse_args()
runtime = a.runtime.resolve()
script = runtime / 'scripts/train_flash_sac.py'
if not script.is_file() or not a.reference.is_file():
    p.error('Runtime trainer and retargeted reference parquet must exist')
if a.worlds < 1 or a.steps < 1:
    p.error('Worlds and steps must be positive')
experiment = 'g1_recon_body_flash_sac' if a.embodiment == 'g1' else 'sharpa_flash_sac'
command = [sys.executable, str(script), f'experiment={experiment}',
    f'task.parquet="{a.reference.resolve()}"', f'output_dir="{a.output.resolve()}"',
    f'scene.world_count={a.worlds}', f'training.total_environment_steps={a.steps}',
    'training.replay.capacity=250000', 'training.replay.minimum_size=8192',
    'training.replay.batch_size=512', 'training.replay.observation_storage_dtype=float16',
    'logging.mode=disabled', 'reset.seed=42']
if a.dry_run:
    print('\n'.join(command))
else:
    subprocess.run(command, cwd=runtime, check=True)
