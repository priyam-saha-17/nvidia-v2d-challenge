"""Apply a lossless depth-cache encoding speedup to the pinned native runtime."""
import argparse,hashlib,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/tmp/video_to_data'))
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
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
        'frame_timeline_changed':False,'canonical_reencode_uses_same_patched_encoder':True}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
