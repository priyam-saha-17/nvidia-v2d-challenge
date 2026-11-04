"""Build a bounded Track 1 segmentation batch with FP16/FP32 recovery on T4."""
import argparse,ast,json
from pathlib import Path
import nbformat as nb
p=argparse.ArgumentParser()
p.add_argument('--episodes',type=int,nargs='+',required=True)
p.add_argument('--name',required=True)
a=p.parse_args()
if any(x not in range(1,30) for x in a.episodes):p.error('Use remaining Track 1 episodes 1..29')
root=Path(__file__).resolve().parents[1]
b=nb.read(root/'notebooks/track1_segmentation/segment.ipynb',as_version=4)
b.cells=b.cells[:2]
b.cells[0]=nb.v4.new_markdown_cell('''# Track 1 automatic segmentation batch
Automatic Grounding DINO + SAM2 prompts from released Track 1 videos. T4 FP16
uses consistent stored-memory precision; retry FP32 on numerical failure.
Retain per-frame finite checks and exact original-frame coverage. No manual labels.
''')
# Store an unmodified predictor copy to allow a clean per-process precision switch.
s=b.cells[1].source
start=s.index("original=predictor_source.read_text()")
end=s.index("subprocess.run([sys.executable,'-m','pip'",start)
s=s[:start]+"original=predictor_source.read_text()\npredictor_backup=original\n"+s[end:]
b.cells[1].source=s
source=(root/'track1/segment_video.py').read_text()
code="Path('/tmp/segment_video.py').write_text("+repr(source)+")\n"
code+=f"episode_ids={a.episodes!r}\n"
code+='''base='https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_1/'
with urllib.request.urlopen(base+'meta/episodes_metadata.jsonl') as response: metadata={x['episode_index']:x for x in map(json.loads,response.read().decode().splitlines())}
with urllib.request.urlopen(base+'meta/episodes.jsonl') as response: episodes={x['episode_index']:x for x in map(json.loads,response.read().decode().splitlines())}
import shutil,time
results=[]
for episode_id in episode_ids:
 m=metadata[episode_id];episode=episodes[episode_id]
 video=Path('/tmp')/(m['sequence_id']+'.0.color.mp4')
 urllib.request.urlretrieve(base+f'videos/chunk-000/observation.images.exo_camera/episode_{episode_id:06d}.mp4',video)
 output=Path('/kaggle/working')/f'episode_{episode_id:06d}'
 started=time.monotonic();success=False
 for precision in ('float16','float32'):
  predictor_source.write_text(predictor_backup.replace('torch.bfloat16','torch.'+precision))
  # Avoid stale bytecode when switching dtypes in a fresh child interpreter.
  for cache in predictor_source.parent.glob('__pycache__/sam2_video_predictor.*.pyc'):cache.unlink()
  command=[sys.executable,'/tmp/segment_video.py','--video',str(video),'--object-prompt',m['object_prompt'],
   '--sam2-checkpoint',str(checkpoint),'--output',str(output),'--expected-frames',str(episode['length']),'--precision',precision]
  process=subprocess.run(command,env=env)
  if process.returncode==0:success=True;break
  print('Retrying failed segmentation in full precision',episode_id,flush=True)
 result={'episode':episode_id,'success':success,'precision':precision,'elapsed_seconds':time.monotonic()-started}
 results.append(result)
 Path('/kaggle/working/batch_report.json').write_text(json.dumps(results,indent=2))
 if success:
  shutil.rmtree(output/'frames');video.unlink()
 print('EPISODE_RESULT',result,flush=True)
Path('/kaggle/working/model_manifest.json').write_text(json.dumps({'sam2_revision':sam_revision,
 'sam2_weight_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'track':'track_1',
 'precision_policy':'FP16 with matching stored memory; retry FP32 on failure'},indent=2))
if any(not x['success'] for x in results):raise RuntimeError('Some episodes failed; inspect batch_report.json')
'''
b.cells.append(nb.v4.new_code_cell(code))
for c in b.cells:
 if c.cell_type=='code':ast.parse(c.source)
nb.validate(b)
dest=root/'notebooks'/('track1_segments_'+a.name);dest.mkdir(exist_ok=True)
nb.write(b,dest/'segment.ipynb')
meta=json.loads((root/'notebooks/track1_segmentation/kernel-metadata.json').read_text())
meta.update(id='priyamsaha17/v2d-track1-segments-'+a.name,title='V2D Track1 Segments '+a.name,code_file='segment.ipynb')
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
