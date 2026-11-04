"""Fit a circular tube shape prior to an automatically segmented hoop image.

The released object class supplies circular topology; the observed mask estimates
its projected axes and tube-to-ring radius. Scale and 3D pose remain unestimated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import least_squares
import trimesh


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    image=np.asarray(Image.open(a.image).convert('RGBA'))
    mask=image[:,:,3]>127
    y,x=np.where(mask)
    if len(x)<50 or mask.mean()>.6:
        raise ValueError('Expected a segmented thin hoop, not a filled object/background')
    xy=np.column_stack([x,y]).astype(float)
    center=np.median(xy,axis=0)
    covariance=np.cov(xy.T)
    values,vectors=np.linalg.eigh(covariance)
    axes=np.sqrt(2*values[::-1]);direction=vectors[:,::-1]
    angle=np.arctan2(direction[1,0],direction[0,0])

    def residual(parameters):
        cx,cy,log_a,log_b,theta=parameters
        c,s=np.cos(theta),np.sin(theta)
        q=(xy-[cx,cy])@np.array([[c,-s],[s,c]])
        return np.sqrt((q[:,0]/np.exp(log_a))**2+(q[:,1]/np.exp(log_b))**2)-1

    fit=least_squares(residual,[*center,*np.log(axes),angle],loss='soft_l1',f_scale=.03)
    if not fit.success:raise ValueError('Projected ellipse fit failed')
    major=float(max(np.exp(fit.x[2:4])))
    distance=ndimage.distance_transform_edt(mask)
    ridge=mask&(distance==ndimage.maximum_filter(distance,size=3))
    # Pixel-centre distances overestimate tube radius by half a pixel.
    tube_pixels=max(.5,float(np.median(distance[ridge]))-.5)
    ratio=tube_pixels/major
    if not .002<ratio<.15:raise ValueError(f'Implausible observed hoop tube ratio {ratio}')
    mesh=trimesh.creation.torus(major_radius=1.,minor_radius=ratio,
                               major_sections=128,minor_sections=16)
    colour=np.median(image[:,:,:3][mask],axis=0).astype(np.uint8)
    mesh.visual.vertex_colors=np.tile(np.r_[colour,255],(len(mesh.vertices),1))
    a.output.mkdir(parents=True,exist_ok=True)
    mesh.export(a.output/'normalized_shape.glb')
    report={'method':'mask-fitted circular tube prior','class':'hula_hoop',
        'image_sha256':hashlib.sha256(a.image.read_bytes()).hexdigest(),
        'mask_pixels':int(mask.sum()),'alpha_coverage':float(mask.mean()),
        'ellipse_parameters':fit.x.tolist(),'ellipse_residual_median':float(np.median(np.abs(fit.fun))),
        'major_radius_pixels':major,'tube_radius_pixels':tube_pixels,
        'tube_to_major_radius':ratio,'vertices':len(mesh.vertices),'faces':len(mesh.faces),
        'metric_scale_estimated':False,'camera_registration_estimated':False,
        'manual_geometry_used':False,'ground_truth_used':False,'ready_for_submission':False}
    (a.output/'shape_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
