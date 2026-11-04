"""Estimate metric mesh scale/pose from Track 1 RGB and human-aligned depth.

Inputs are an actual image-conditioned mesh and an observed-frame NPZ containing
rgb, depth, human_mask, object_mask and K. FoundationPose registration proposes
pose at each scale; observed surface agreement and occlusion-aware silhouette
select the hypothesis. No clean trajectory or external object template is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from prep.run_foundationpose_mhr_export import (
    _pose_estimator, _prepare_foundationpose_inputs, _render_candidate_masks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--frame-data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with np.load(a.frame_data,allow_pickle=False) as f:
        rgb,depth,human,mask,K=_prepare_foundationpose_inputs(
            f['rgb'],f['depth'],f['human_mask'],f['object_mask'],f['K'])
    visible=mask.astype(bool)&~human.astype(bool)&np.isfinite(depth)&(depth>0)
    ys,xs=np.where(visible)
    if len(xs)<50:raise ValueError('Insufficient observed, unoccluded object depth')
    depth_values=depth[ys,xs]
    cloud=np.column_stack([(xs-K[0,2])*depth_values/K[0,0],
                           (ys-K[1,2])*depth_values/K[1,1],depth_values])
    extent=np.percentile(cloud,95,axis=0)-np.percentile(cloud,5,axis=0)
    mesh=trimesh.load(a.mesh,force='mesh')
    if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('Expected a reconstructed triangle mesh')
    initial=float(np.linalg.norm(extent)/max(mesh.extents))
    if not np.isfinite(initial) or initial<=0:raise ValueError('Invalid scale initialization')
    rng=np.random.default_rng(42)
    cloud=cloud[rng.choice(len(cloud),min(2048,len(cloud)),replace=False)]
    candidates=[];estimator=None

    def evaluate(scale):
        nonlocal estimator
        candidate=mesh.copy();candidate.vertices*=scale
        if estimator is None:
            estimator=_pose_estimator(candidate,a.output/'registration_debug')
        else:
            estimator.reset_object(model_pts=np.asarray(candidate.vertices),
                model_normals=np.asarray(candidate.vertex_normals),mesh=candidate)
        pose=np.asarray(estimator.register(K=K,rgb=rgb,depth=depth,
            ob_mask=visible,iteration=5),dtype=np.float32)
        if pose.shape!=(4,4) or not np.isfinite(pose).all():
            raise ValueError('Invalid registered pose')
        obj_points=(cloud-pose[:3,3])@pose[:3,:3]
        _,distances,_=trimesh.proximity.closest_point(candidate,obj_points)
        cutoff=np.quantile(distances,.8)
        surface_error=float(distances[distances<=cutoff].mean())
        centered=pose@np.linalg.inv(estimator.get_tf_to_centered_mesh().detach().cpu().numpy())
        rendered=_render_candidate_masks(estimator,centered[None],K,depth.shape,chunk_size=1)[0]
        rendered&=~human.astype(bool)
        union=np.logical_or(rendered,visible).sum()
        iou=float(np.logical_and(rendered,visible).sum()/max(1,union))
        score=surface_error+.03*(1-iou)
        result={'scale':float(scale),'pose':pose.tolist(),'surface_error_m':surface_error,
                'visible_mask_iou':iou,'selection_cost':score}
        candidates.append(result)
        print(json.dumps(result),flush=True)
        return result

    for factor in np.geomspace(.6,1.6,7):evaluate(initial*factor)
    coarse=min(candidates,key=lambda x:x['selection_cost'])
    for factor in np.linspace(.85,1.15,5):evaluate(coarse['scale']*factor)
    best=min(candidates,key=lambda x:x['selection_cost'])
    mesh.vertices*=best['scale']
    # Project observed appearance onto visible vertices; unseen vertices use the
    # observed object's median colour rather than background pixels.
    pose=np.asarray(best['pose']);camera=np.asarray(mesh.vertices)@pose[:3,:3].T+pose[:3,3]
    positive=camera[:,2]>1e-6
    u=np.zeros(len(camera),dtype=int);v=u.copy()
    u[positive]=np.rint(K[0,0]*camera[positive,0]/camera[positive,2]+K[0,2]).astype(int)
    v[positive]=np.rint(K[1,1]*camera[positive,1]/camera[positive,2]+K[1,2]).astype(int)
    inside=positive&(u>=0)&(u<rgb.shape[1])&(v>=0)&(v<rgb.shape[0])
    colour=np.median(rgb[visible],axis=0).astype(np.uint8)
    colours=np.tile(np.r_[colour,255],(len(camera),1)).astype(np.uint8)
    ids=np.flatnonzero(inside)
    visible_vertices=ids[visible[v[ids],u[ids]]&
        (np.abs(depth[v[ids],u[ids]]-camera[ids,2])<.03)]
    colours[visible_vertices,:3]=rgb[v[visible_vertices],u[visible_vertices]]
    mesh.visual=trimesh.visual.ColorVisuals(vertex_colors=colours)
    mesh.export(a.output/'metric_mesh.glb')
    np.save(a.output/'initial_camera_pose.npy',pose)
    report={'initial_scale':initial,'selected':best,'candidates':candidates,
        'selection':'observed trimmed surface distance + 0.03m*(1-visible-mask-IoU)',
        'manual_geometry_used':False,'ground_truth_used':False,
        'frame_data_sha256':hashlib.sha256(a.frame_data.read_bytes()).hexdigest()}
    (a.output/'scale_report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
