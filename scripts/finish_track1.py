"""Finish the two frozen production jobs, validate all 30 episodes, submit once.

Read-only retries are allowed. Ambiguous uploads never retry automatically; exact
official receipt IDs are polled. No placeholder or failed episode is packaged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi


ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'references/submission_kit/v2d_submission_kit'
SOURCES = {
    'a': 'priyamsaha17/v2d-track1-inference-main-a/1',
    'b': 'priyamsaha17/v2d-track1-inference-main-b/1',
}
METRICS = {'cd_h_cm', 'cd_o_cm', 'acc_h_cm', 'acc_o_cm', 'interpenetration_cm'}
PATTERN = (r'^(?:(?:recovered/)?(?:episodes/episode_\d{6}(?:\.npz|_object\.glb|\.report\.json)'
           r'|batch_reconstruction_report\.json)|runtime_patch_manifest\.json'
           r'|track1_runtime_report\.json|t4_precision_manifest\.json)$')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def command(arguments, *, check=True, cwd=ROOT):
    print('EXECUTING', ' '.join(map(str, arguments)), flush=True)
    return subprocess.run(list(map(str, arguments)), cwd=cwd, check=check)


def scored(receipt):
    entries = receipt.get('submissions', {})
    return (set(entries) == METRICS and all(
        row.get('state') == 'COMPLETE' and row.get('ref') and row.get('public_score') not in ('', None)
        for row in entries.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--upload', action='store_true')
    p.add_argument('--deadline-utc', default='2026-11-04T22:00:00+00:00')
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    state_path = a.output / 'completion_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'sources': SOURCES, 'code_commit_url': a.commit, 'stage': 'waiting_for_actual_outputs'}
    if state['sources'] != SOURCES or state['code_commit_url'] != a.commit:
        raise ValueError('Existing completion state identifies different producing artifacts')
    deadline = datetime.fromisoformat(a.deadline_utc)
    api = KaggleApi()
    api.authenticate()

    def persist(stage):
        state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)

    if not state.get('receipt'):
        previous = None
        while True:
            statuses = {}
            for label, notebook in SOURCES.items():
                try:
                    response = api.kernels_status(notebook)
                    statuses[label] = getattr(response.status, 'name', str(response.status).split('.')[-1])
                except Exception as error:
                    statuses[label] = 'READ_ERROR_' + type(error).__name__
            if statuses != previous:
                print('PRODUCTION_STATUS', json.dumps(statuses), flush=True)
                previous = statuses
            state['statuses'] = statuses
            persist('waiting_for_actual_outputs')
            if any(value in {'ERROR', 'CANCELLED'} for value in statuses.values()):
                raise RuntimeError('A production job failed; diagnose its real outputs before proceeding')
            if all(value == 'COMPLETE' for value in statuses.values()):
                break
            if datetime.now(timezone.utc) >= deadline:
                raise RuntimeError('Competition deadline reached before complete actual outputs were ready')
            time.sleep(45)

        persist('downloading_final_outputs')
        for label, notebook in SOURCES.items():
            destination = a.output / 'downloads' / label
            token = None
            seen_tokens = set()
            while True:
                # Pagination is required: native bundles can push recovered outputs
                # beyond the first page even though only final NPZ/GLB are fetched.
                _, token = api.kernels_output(notebook, str(destination), file_pattern=PATTERN,
                                              quiet=True, page_size=200, page_token=token)
                if not token:
                    break
                if token in seen_tokens:
                    raise RuntimeError('Output pagination repeated a token')
                seen_tokens.add(token)
        episodes = a.output / 'episodes'
        command([sys.executable, ROOT / 'track1/collect_outputs.py',
                 '--source', SOURCES['a'] + '=' + str(a.output / 'downloads/a'),
                 '--source', SOURCES['b'] + '=' + str(a.output / 'downloads/b'),
                 '--source', SOURCES['b'] + '=' + str(a.output / 'downloads/b/recovered'),
                 '--output', episodes])
        persist('official_validation')
        command([sys.executable, ROOT / 'track1/submit.py', '--episodes', episodes,
                 '--commit', a.commit, '--output', a.output / 'submission'])
        packed = a.output / 'submission/track1.parquet'
        state['packed_sha256'] = hashlib.sha256(packed.read_bytes()).hexdigest()
        persist('validated_not_uploaded')
        if not a.upload:
            print('Complete genuine roster packed and validated; upload not requested.', flush=True)
            return
        if datetime.now(timezone.utc) >= deadline:
            raise RuntimeError('Competition deadline reached before upload')
        receipt = a.output / 'submission' / ('receipt-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
        # Persist the intended receipt before the only uploading command. If the
        # process is interrupted before it exists, reconcile Kaggle manually.
        state['receipt'] = str(receipt)
        persist('upload_attempt_reserved')
        command([sys.executable, KIT / 'submit/v2d_submit.py', packed, '--track', 'track_1',
                 '-m', 'Track 1 native MHR, observed-depth geometry and contact refinement',
                 '--wait', '--receipt', receipt], check=False, cwd=KIT)

    receipt_path = Path(state['receipt'])
    if not receipt_path.is_file():
        raise RuntimeError('An upload attempt was reserved without a receipt; inspect Kaggle before any retry')
    packed = a.output / 'submission/track1.parquet'
    if hashlib.sha256(packed.read_bytes()).hexdigest() != state['packed_sha256']:
        raise ValueError('Counted packed artifact changed; never repack or replace it')
    while True:
        receipt = json.loads(receipt_path.read_text())
        if scored(receipt):
            persist('scored_all_five')
            print('TRACK1_SCORED_ALL_FIVE', json.dumps(receipt), flush=True)
            return
        entries = receipt.get('submissions', {})
        if set(entries) != METRICS or any(not row.get('ref') or row.get('state') in {'ERROR', 'UNREADABLE'} for row in entries.values()):
            persist('requires_upload_or_scoring_diagnosis')
            raise RuntimeError('Receipt has an incomplete/failed/ambiguous upload or scoring error; inspect exact IDs before retrying')
        persist('polling_saved_submission_ids')
        command([sys.executable, KIT / 'submit/v2d_submit.py', '--track', 'track_1',
                 '--poll-receipt', receipt_path], check=False, cwd=KIT)
        time.sleep(15)


if __name__ == '__main__':
    main()
