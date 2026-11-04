from pathlib import Path
import argparse,ast,json
import nbformat as nb
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--tier',choices=['tier1','tier2'],required=True);a=p.parse_args()
b=nb.read(root/'notebooks/tier1_retarget_probe/retarget.ipynb',as_version=4)
b.cells[0]=nb.v4.new_markdown_cell(f'''# Track 2 {a.tier} complete retargeting
Retarget all frames of all 30 sequences using only this tier's released input.
Preserve fitted ground planes and support surfaces. These outputs are robot
references, not achieved policy trajectories or a leaderboard submission.''')
tier_folder='tier_1_multiview_caption' if a.tier=='tier1' else 'tier_2_synthetic_noise'
b.cells[3]=nb.v4.new_code_cell(f'''source=Path('/tmp/{a.tier}');(source/'meta').mkdir(parents=True)
base='https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_2/{tier_folder}/'
def get(relative):
 dest=source/relative;dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():urllib.request.urlretrieve(base+relative,dest)
 return dest
get('meta/info.json');metadata=get('meta/episodes_metadata.jsonl')
items=[json.loads(line) for line in metadata.read_text().splitlines()]
assert len(items)==30
paths=set()
for m in items:
 paths.update([f"data/chunk-000/episode_{{m['episode_index']:06d}}.parquet",m['mesh'],m['ground_plane'],m['support_surface']])
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(get,sorted(paths)))
''')
b.cells[-1]=nb.v4.new_code_cell('''motion_root=Path('/kaggle/working/motion_assets/human_motion_data')
failures=[]
for m in items:
 command=[py,'scripts/retarget/soma_to_g1.py','/tmp/retarget-inputs/'+m['sequence_id'],
  '--save','--motion-root',str(motion_root),'--soma-subdir','soma']
 result=subprocess.run(command,cwd=rg,env=env)
 if result.returncode:
  failures.append(m['episode_index']);continue
 stage=motion_root/'whole_body/reconstructed_stage';stage.mkdir(parents=True,exist_ok=True)
 shutil.copy2(source/m['support_surface'],stage/Path(m['support_surface']).name)
report={'source_tier':str(source),'episodes_requested':30,'failures':failures,
 'parquets':[str(p) for p in motion_root.rglob('*.parquet')]}
Path('/kaggle/working/retarget_report.json').write_text(json.dumps(report,indent=2))
assert not failures,failures
assert len(report['parquets'])==30,report
print('Complete retargeting:',report)
''')
for c in b.cells:
 if c.cell_type=='code':ast.parse(c.source)
nb.validate(b);dest=root/f'notebooks/{a.tier}_retarget_all';dest.mkdir(exist_ok=True)
nb.write(b,dest/'retarget.ipynb')
meta=json.loads((root/'notebooks/tier1_retarget_probe/kernel-metadata.json').read_text())
meta.update(id=f'priyamsaha17/v2d-{a.tier}-retarget-all',title=f'V2D {a.tier} Retarget All')
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
