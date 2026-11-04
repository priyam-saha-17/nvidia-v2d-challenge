"""Export the official competition mapping without guessing Kaggle slugs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
kit = ROOT / 'references/submission_kit/v2d_submission_kit'
config = json.loads((kit / 'config/tracks.json').read_text())
rows = []
for track, spec in config['tracks'].items():
    for metric, slug in spec['competitions'].items():
        rows.append({'track': track, 'metric': metric, 'slug': slug,
                     'higher_is_better': config['metric_display'][metric]['higher_is_better'],
                     'rules_url': f'https://www.kaggle.com/competitions/{slug}/rules',
                     'entry_status': 'not_checked', 'submission_status': 'not_submitted'})
assert len(rows) == 23 and len({r['slug'] for r in rows}) == 23
(ROOT / 'configs/competitions.json').write_text(json.dumps(rows, indent=2) + '\n')
print('Exported 23 official competition mappings')
