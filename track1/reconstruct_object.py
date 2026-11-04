"""Reconstruct an object from its automatic RGBA video crop using Hunyuan3D.

This produces a normalized shape hypothesis. Metric scale and camera registration
must be estimated from the Track 1 observations before using it in a submission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from huggingface_hub import HfApi, snapshot_download
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--revision',help='Pin a previously recorded HF model revision')
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--resolution',type=int,default=384)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.steps<1 or a.resolution<64:p.error('Invalid sampling settings')
    image=Image.open(a.image).convert('RGBA')
    alpha=np.asarray(image)[:,:,3]
    if not (alpha>0).any() or (alpha>0).all():
        raise ValueError('Expected an actual masked RGBA observation')
    repo='tencent/Hunyuan3D-2'
    revision=a.revision or HfApi().model_info(repo).sha
    snapshot_download(repo,revision=revision,local_dir=a.weights,
        allow_patterns=['hunyuan3d-dit-v2-0/**','hunyuan3d-vae-v2-0/**','*.json'])
    pipeline=Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(str(a.weights),
        subfolder='hunyuan3d-dit-v2-0',device='cuda',dtype=torch.float16)
    with torch.inference_mode():
        mesh=pipeline(image=image,num_inference_steps=a.steps,
            octree_resolution=a.resolution,num_chunks=4000,
            generator=torch.Generator(device='cuda').manual_seed(a.seed))[0]
    if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('Image-conditioned model produced invalid geometry')
    a.output.mkdir(parents=True,exist_ok=True)
    mesh.export(a.output/'normalized_shape.glb')
    report={'model':repo,'revision':revision,'seed':a.seed,'steps':a.steps,
        'resolution':a.resolution,'input_sha256':hashlib.sha256(a.image.read_bytes()).hexdigest(),
        'vertices':len(mesh.vertices),'faces':len(mesh.faces),
        'metric_scale_estimated':False,'camera_registration_estimated':False,
        'ready_for_submission':False,'manual_geometry_used':False}
    (a.output/'shape_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
