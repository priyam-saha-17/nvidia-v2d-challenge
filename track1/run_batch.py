"""Run independent Track 1 episodes on both T4 devices, preserving real artifacts.

Intermediate caches remain in compressed private notebook artifacts. The public
submission directory contains only actual completed NPZ/mesh pairs, never fixtures.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib,json,os,shutil,subprocess,sys,tarfile,tempfile,threading,time,urllib.request
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()


@contextmanager
def retained_work(prefix):
    # Cleanup is done after success/failure artifacts have been retained.
    yield Path(tempfile.mkdtemp(prefix=prefix))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,nargs='+',required=True)
    p.add_argument('--devices',default='0,1')
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--input-root',type=Path,default=Path('/kaggle/input'))
    p.add_argument('--dataset-root',type=Path,help='Read released videos/meta offline from a mounted challenge dataset')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--retain-intermediates',action='store_true')
    a=p.parse_args()
    if len(a.episodes)!=len(set(a.episodes)) or any(x not in range(30) for x in a.episodes):p.error('Expected distinct Track 1 episodes 0..29')
    if len(a.episodes)>30:p.error('Track 1 has only 30 episodes')
    devices=a.devices.split(',')
    if len(set(devices))!=len(devices) or not all(x.isdigit() for x in devices):p.error('Expected distinct CUDA device numbers')
    base='https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_1/'
    if a.dataset_root:
        dataset=a.dataset_root/'track_1' if (a.dataset_root/'track_1/meta').is_dir() else a.dataset_root
        metadata={x['episode_index']:x for x in map(json.loads,(dataset/'meta/episodes_metadata.jsonl').read_text().splitlines())}
        episodes={x['episode_index']:x for x in map(json.loads,(dataset/'meta/episodes.jsonl').read_text().splitlines())}
    else:
        with urllib.request.urlopen(base+'meta/episodes_metadata.jsonl') as f:metadata={x['episode_index']:x for x in map(json.loads,f.read().decode().splitlines())}
        with urllib.request.urlopen(base+'meta/episodes.jsonl') as f:episodes={x['episode_index']:x for x in map(json.loads,f.read().decode().splitlines())}
    a.output.mkdir(parents=True,exist_ok=True)
    records=[];lock=threading.Lock();code=Path(__file__).resolve().parent

    def worker(device,ids):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device;env['PYTHONUNBUFFERED']='1'
        env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
        env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        for episode_id in ids:
            name=f'episode_{episode_id:06d}';started=time.monotonic()
            log=a.output/(name+'.log')
            result={'episode':episode_id,'device':device,'success':False}
            work=None
            try:
                m=metadata[episode_id]
                reports=[x for x in a.input_root.rglob('segmentation_report.json') if x.parent.name==name]
                semantic=[x for x in reports if json.loads(x.read_text()).get('semantic_selection')]
                if semantic:reports=semantic
                if len(reports)!=1:raise ValueError('Expected one authoritative actual segmentation report for '+name)
                report_path=reports[0];report=json.loads(report_path.read_text())
                if report['frames']!=episodes[episode_id]['length']:raise ValueError('Incorrect original frame coverage')
                with retained_work(name+'-') as work:
                    work=Path(work);video=work/(m['sequence_id']+'.0.color.mp4')
                    if a.dataset_root:shutil.copy2(dataset/f'videos/chunk-000/observation.images.exo_camera/{name}.mp4',video)
                    else:urllib.request.urlretrieve(base+f'videos/chunk-000/observation.images.exo_camera/{name}.mp4',video)
                    if sha256(video)!=report['video_sha256']:raise ValueError('Video differs from segmented source')
                    if m['object']=='hula_hoop':
                        subprocess.run([sys.executable,str(code/'fit_hoop.py'),'--image',str(report_path.with_name('object_observation.png')),'--output',str(work/'shape')],env=env,check=True)
                        mesh=work/'shape/normalized_shape.glb'
                    else:
                        candidates=[x for x in a.input_root.rglob('normalized_shape.glb') if x.parent.name==name]
                        if len(candidates)!=1:raise ValueError('Missing or ambiguous actual reconstructed shape for '+name)
                        mesh=candidates[0]
                    reconstruction=work/'reconstruction'
                    command=[sys.executable,str(code/'run_episode.py'),'--video',str(video),
                        '--masks',str(report_path.with_name('masks.h5')),'--mesh',str(mesh),
                        '--weights',str(a.weights),'--output',str(reconstruction),'--episode',str(episode_id),
                        '--frames',str(report['frames']),'--seed-frame',str(report['seed']['frame'])]
                    print('EPISODE_STARTED',name,'CUDA device',device,flush=True)
                    with log.open('w') as f:
                        process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT)
                        while process.poll() is None:
                            time.sleep(15)
                            with log.open('rb') as stream:
                                stream.seek(max(0,log.stat().st_size-2048));tail=stream.read().decode(errors='replace').replace('\r','\n').strip().splitlines()
                            if tail:print('EPISODE_PROGRESS',name,tail[-1][:220],flush=True)
                    if process.returncode:raise subprocess.CalledProcessError(process.returncode,command)
                    target=a.output/'episodes';target.mkdir(exist_ok=True)
                    for path in (reconstruction/'episodes').iterdir():shutil.copy2(path,target/path.name)
                    previews=a.output/'previews'/name;previews.mkdir(parents=True,exist_ok=True)
                    for path in (reconstruction/'final').rglob('*.mp4'):shutil.copy2(path,previews/path.name)
                    retained=a.output/'native_outputs'/name
                    for path in reconstruction.rglob('*'):
                        if not path.is_file():continue
                        relative=path.relative_to(reconstruction)
                        if relative.parts[0]=='bootstrap':continue
                        if path.suffix in {'.pth','.pkl','.json','.glb'} or path.name.endswith('.sam3d.h5'):
                            destination=retained/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
                    result.update(success=True,source_video_sha256=report['video_sha256'],frames=report['frames'],
                        retained_native_outputs=True,regenerable_rgb_depth_caches_retained=a.retain_intermediates)
                    if a.retain_intermediates:
                        archive=a.output/(name+'_intermediates.tar.gz')
                        with tarfile.open(archive,'w:gz',compresslevel=1) as handle:handle.add(reconstruction,arcname=name+'/reconstruction')
                        result.update(intermediate_archive_sha256=sha256(archive),intermediate_archive_bytes=archive.stat().st_size)
            except Exception as error:
                result['error']=type(error).__name__+': '+str(error)
                if log.exists():print(log.read_text(errors='replace')[-2400:],flush=True)
            finally:
                if work is not None and work.exists():
                    if not result['success'] and (work/'reconstruction').exists():
                        try:
                            archive=a.output/(name+'_failed_intermediates.tar.gz')
                            with tarfile.open(archive,'w:gz',compresslevel=1) as handle:
                                handle.add(work/'reconstruction',arcname=name+'/reconstruction')
                            result['failed_intermediate_archive_sha256']=sha256(archive)
                        except Exception as archive_error:result['archive_error']=str(archive_error)
                    shutil.rmtree(work)
            result['elapsed_seconds']=time.monotonic()-started
            with lock:
                records.append(result)
                (a.output/'batch_reconstruction_report.json').write_text(json.dumps(records,indent=2)+'\n')
            print('EPISODE_RESULT',json.dumps(result),flush=True)
    assigned=[[] for _ in devices];loads=[0 for _ in devices]
    for episode in sorted(a.episodes,key=lambda value:episodes[value]['length'],reverse=True):
        index=min(range(len(devices)),key=lambda value:loads[value])
        assigned[index].append(episode);loads[index]+=episodes[episode]['length']+128
    print('DEVICE_ASSIGNMENTS',json.dumps(dict(zip(devices,assigned))),flush=True)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures=[pool.submit(worker,device,assigned[index]) for index,device in enumerate(devices)]
        for future in futures:future.result()
    # Keep completed episodes attachable even if an independent episode failed.
    # Coverage and per-episode success are mandatory before submission.
    print('Batch finished; successful episodes',sum(x['success'] for x in records),'of',len(records),flush=True)


if __name__=='__main__':main()
