"""Summarize retained production logs without API calls or credential access."""
import json
from pathlib import Path
import re
import time

root = Path(__file__).resolve().parents[1]
complete, failed, progress = set(), {}, {}
ages = {}
for job in ('main-a', 'main-b'):
    path = root / 'outputs' / ('track1_inference_' + job) / 'tail-v1.jsonl'
    if not path.exists():
        continue
    ages[job] = round(time.time() - path.stat().st_mtime)
    with path.open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = event.get('data', '')
            timestamp = float(event.get('time') or 0)
            for marker in ('EPISODE_RESULT ', 'RECOVERED_EPISODE '):
                for match in re.finditer(re.escape(marker) + r'(\{[^\n]+\})', data):
                    try:
                        row = json.loads(match.group(1))
                    except json.JSONDecodeError:
                        continue
                    if row.get('success') is True:
                        complete.add(row['episode'])
                        failed.pop(row['episode'], None)
                    else:
                        failed[row['episode']] = row.get('error', 'unspecified failure')[:140]
            for match in re.finditer(r'EPISODE_PROGRESS\s+episode_(\d+)\s+([^\n]*)', data):
                episode = int(match.group(1))
                key = (job, episode)
                if timestamp >= progress.get(key, (-1, ''))[0]:
                    text = re.sub(r'[^\x20-\x7e]', '', match.group(2).split('EPISODE_PROGRESS')[0])
                    progress[key] = (timestamp, text[:160])
print('Actual successful exports:', len(complete), '/ 30;', sorted(complete))
if failed:
    print('FAILED:', json.dumps(failed))
for (job, episode), (_, text) in sorted(progress.items()):
    if episode not in complete and episode not in failed:
        print(job, f'episode_{episode:06d}:', text)
print('Log ages (seconds):', ages)
state = root / 'outputs/track1_completion/completion_state.json'
if state.is_file():
    value = json.loads(state.read_text())
    print('Submission runner:', value['stage'])
    if value.get('receipt'):
        print('Existing receipt:', value['receipt'])
