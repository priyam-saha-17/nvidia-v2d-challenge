"""Apply a lossless depth-cache encoding speedup to the pinned native runtime."""
import argparse,hashlib,json,urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/tmp/video_to_data'))
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    # The source archive deliberately skipped LFS smudging. These small public
    # topology assets are required by contact refinement, not pretrained weights.
    revision='8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef'
    assets={
        'mhr_collision_proxy_4000v.npz':(126142,'a026fe82599609ee82814b4c18eb0335fa70925740dea5e1c1e1c0930d5edcf7'),
        'mhr_hand_surface_spec.npz':(47140,'65e467ae534281c8c5370b76d672c80cf99b95bc73af9c3ac64d5bea6c7f60c8'),
    }
    hydrated=[]
    for name,(size,digest) in assets.items():
        relative='reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/'+name
        target=a.root/relative
        data=target.read_bytes()
        if data.startswith(b'version https://git-lfs.github.com/spec/v1'):
            url='https://media.githubusercontent.com/media/nvidia-isaac/video_to_data/'+revision+'/'+relative
            with urllib.request.urlopen(url,timeout=90) as response:data=response.read()
            if len(data)!=size or hashlib.sha256(data).hexdigest()!=digest:
                raise ValueError('LFS asset identity mismatch: '+name)
            target.write_bytes(data)
        if len(data)!=size or hashlib.sha256(data).hexdigest()!=digest:
            raise ValueError('Invalid runtime topology asset: '+name)
        hydrated.append({'name':name,'bytes':size,'sha256':digest,'source_revision':revision})
    path=a.root/'reconstruction/modules/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py'
    before=path.read_bytes();source=before.decode()
    old='compress_level=9, optimize=True';new='compress_level=1, optimize=False'
    if source.count(old)!=1:raise ValueError('Pinned depth PNG encoder differs; inspect before patching')
    path.write_text(source.replace(old,new))
    report={'patch':'lossless 16-bit PNG cache encoding','path':str(path),
        'before_sha256':hashlib.sha256(before).hexdigest(),
        'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'before_options':old,'after_options':new,
        'depth_values_changed':False,'spatial_resolution_changed':False,
        'frame_timeline_changed':False,'canonical_reencode_uses_same_patched_encoder':True,
        'verified_public_topology_assets':hydrated}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
