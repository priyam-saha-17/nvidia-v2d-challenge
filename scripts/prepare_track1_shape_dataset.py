"""Package only recovered, reviewed Track 1 generated meshes for private inference."""
from pathlib import Path
import hashlib,json,shutil
root=Path(__file__).resolve().parents[1]
metadata=list(map(json.loads,(root/'data/challenge/track_1/meta/episodes_metadata.jsonl').read_text().splitlines()))
ranking=json.loads((root/'outputs/track1_crop_ranking/semantic_crop_ranking.json').read_text())
by_hash={entry['input_sha256']:entry for entry in ranking}
selected={}
for group in ('a','b'):
 for report_path in (root/f'outputs/track1_shapes_{group}/raw_shapes').glob('*/shape_report.json'):
  report=json.loads(report_path.read_text());name=report_path.parent.name
  score=by_hash.get(report['input_sha256'])
  if score is None or score['predicted_class']!=name or score['semantic_margin']<=0:continue
  if name not in selected or score['semantic_margin']>selected[name][2]['semantic_margin']:
   selected[name]=(group,score['episode'],score)
dest=root/'outputs/track1_shape_dataset';dest.mkdir(exist_ok=True)
manifest=[]
for entry in metadata:
 name=entry['object']
 if name not in selected:continue
 group,source_episode,semantic=selected[name]
 prepared=root/'outputs/track1_accepted_shapes'/name
 mesh=prepared/'normalized_shape.glb'
 raw_report=root/f'outputs/track1_shapes_{group}/raw_shapes'/name/'shape_report.json'
 folder=dest/f"episode_{entry['episode_index']:06d}";folder.mkdir(exist_ok=True)
 shutil.copy2(mesh,folder/'normalized_shape.glb')
 report={'episode':entry['episode_index'],'object':name,'shape_source_episode':source_episode,
   'mesh_sha256':hashlib.sha256(mesh.read_bytes()).hexdigest(),
   'generation':json.loads(raw_report.read_text()),
   'source_selection':'maximum positive automatic semantic margin among generated shape sources',
   'semantic_source_check':semantic,
   'mesh_preparation':json.loads((prepared/'mesh_preparation_report.json').read_text()),
   'ground_truth_used':False,'metric_scale_estimated':False}
 (folder/'shape_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
 manifest.append(report)
(dest/'shape_dataset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(dest/'dataset-metadata.json').write_text(json.dumps({'title':'V2D Track1 Reconstructed Shapes',
 'id':'priyamsaha17/v2d-track1-reconstructed-shapes','licenses':[{'name':'other'}],
 'description':'Private compute artifacts: actual image-conditioned Track 1 object reconstruction, fixed-budget meshes and provenance. No reference geometry, credentials or pretrained checkpoint files. Eight reviewed object classes; corrected iron reconstruction is pending. Original model sources retain their own licenses.'},indent=2)+'\n')
print('Prepared actual image-conditioned meshes for',len(manifest),'episodes; iron and hoop not included')
