"""Run actual Track 1 reconstruction with observed-depth object calibration.

A bootstrap pass estimates human geometry and metric depth. A separate final
export uses the calibrated object; unchanged camera/human/depth artifacts are
transferred explicitly, with their source identities retained in the run report.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from v2d.cari4d.lib import run_inference as native


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--masks',type=Path,required=True)
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episode',type=int,required=True)
    p.add_argument('--frames',type=int,required=True)
    p.add_argument('--seed-frame',type=int,default=15)
    a=p.parse_args()
    original=native._stage_run
    bootstrap_outputs={}

    class BootstrapComplete(Exception):pass

    def bootstrap(name,command,inputs,outputs,marker_root,env,overwrite):
        if name=='05_object_pose':raise BootstrapComplete()
        result=original(name,command,inputs,outputs,marker_root,env,overwrite)
        bootstrap_outputs[name]=list(outputs)
        return result

    options=dict(download_models=False,expected_frames=a.frames,moge_batch_size=1,moge_devices="cuda:0",
        sam3d_chunk_size=1,sam3d_refit_batch_size=32,
        depth_alignment_render_batch_size=1,depth_alignment_encoding_workers=2,
        depth_alignment_write_batch_size=16,coconet_render_batch_size=1,
        coconet_crop_workers=2,coconet_crop_buffer_count=1,
        postopt_batch_size=32,render_batch_size=1)
    native._stage_run=bootstrap
    try:
        native.run_inference(str(a.video),str(a.masks),str(a.mesh),str(a.weights),
                             str(a.output/'bootstrap'),**options)
    except BootstrapComplete:
        pass
    finally:
        native._stage_run=original
    sequence=a.video.name[:-len('.0.color.mp4')]
    source_root=a.output/'bootstrap'/sequence
    export=source_root/'export'/sequence
    from prep.mhr_export_utils import (camera_calibration,frame_names,load_edex,
                                      read_depth_m,read_rgb,read_mask)
    K,_=camera_calibration(load_edex(export),0)
    names=frame_names(export,0)
    if len(names)!=a.frames:raise ValueError('Bootstrap changed the video timeline')
    frame=names[min(a.seed_frame,a.frames-1)]
    calibration=a.output/'calibration';calibration.mkdir(parents=True,exist_ok=True)
    frame_data=calibration/'observed_frame.npz'
    np.savez_compressed(frame_data,rgb=read_rgb(export,0,frame),
        depth=read_depth_m(export,0,frame,depth_root=source_root/'depth/moge2/aligned.h5'),
        human_mask=read_mask(export,'human',0,frame),
        object_mask=read_mask(export,'object',0,frame),K=K)
    subprocess.run([sys.executable,str(Path(__file__).with_name('calibrate_object.py')),
        '--mesh',str(a.mesh),'--frame-data',str(frame_data),'--output',str(calibration)],check=True)
    transferred=[]

    def final_stage(name,command,inputs,outputs,marker_root,env,overwrite):
        if name=='07_refinement':
            # The scoring schema has one identity and zero expression per clip.
            # Refine contacts against that same geometry, preserving the raw prediction.
            import torch
            original_bundle=Path(command[command.index('--bundle')+1])
            projected=original_bundle.with_name('shared_identity.pth')
            bundle=torch.load(original_bundle,map_location='cpu',weights_only=False)
            shared={key:np.median(np.asarray(bundle['pr'][key]),axis=0).astype(np.float32)
                    for key in ('mhr_shape','mhr_scale')}
            for section in ('pr','pr_initial'):
                if section not in bundle:continue
                for key,value in shared.items():bundle[section][key]=np.repeat(value[None],a.frames,axis=0)
                bundle[section]['mhr_face']=np.zeros((a.frames,72),np.float32)
            projection={'method':'per-clip median identity; zero expression matching scorer',
                'source_bundle':str(original_bundle),
                'source_sha256':hashlib.sha256(original_bundle.read_bytes()).hexdigest(),
                'shape':shared['mhr_shape'].tolist(),'scale_latents':shared['mhr_scale'].tolist(),
                'ground_truth_used':False}
            bundle['metadata']['shared_identity_projection']=projection
            torch.save(bundle,projected)
            (projected.parent/'identity_projection_report.json').write_text(json.dumps(projection,indent=2)+'\n')
            command=list(command);command[command.index('--bundle')+1]=str(projected)
            inputs=[projected,*list(inputs)[1:]]
            return original(name,command,inputs,outputs,marker_root,env,overwrite)
        if name not in {'01_depth','03_human_initialization','04_depth_alignment'}:
            return original(name,command,inputs,outputs,marker_root,env,overwrite)
        source=bootstrap_outputs[name]
        if len(source)!=len(outputs):raise ValueError('Native stage output contract changed')
        for src,dst in zip(source,outputs):
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():dst.unlink()
            try:os.link(src,dst)
            except OSError:shutil.copy2(src,dst)
        entry={'name':name,'reused':True,'elapsed_seconds':0.,
            'reuse_reason':'Identical video, masks and camera; object-independent stage',
            'source_outputs':native._output_identities(source),
            'outputs':native._output_identities(outputs)}
        transferred.append(entry)
        return entry

    native._stage_run=final_stage
    try:
        native.run_inference(str(a.video),str(a.masks),str(calibration/'metric_mesh.glb'),
            str(a.weights),str(a.output/'final'),**options)
    finally:
        native._stage_run=original
    (a.output/'stage_transfer_report.json').write_text(json.dumps(transferred,indent=2)+'\n')
    final=a.output/'final'/sequence
    subprocess.run([sys.executable,str(Path(__file__).with_name('export_cari4d.py')),
        '--bundle',str(final/'inference/refined.pth'),
        '--mesh',str(final/'export'/sequence/'object_mesh/output_aligned.glb'),
        '--mhr-assets-root',str(a.weights/'sam3d_body'),
        '--episode',str(a.episode),'--frames',str(a.frames),
        '--output',str(a.output/'episodes')],check=True)


if __name__=='__main__':main()
