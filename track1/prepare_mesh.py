"""Remove degenerate generation debris and fit a real mesh to the scoring budget."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import trimesh


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--strict-topology',action='store_true')
    a=p.parse_args()
    mesh=trimesh.load(a.mesh,force='mesh',process=False)
    if not isinstance(mesh,trimesh.Trimesh) or not np.isfinite(mesh.vertices).all():
        raise ValueError('Expected finite triangle geometry')
    original={'vertices':len(mesh.vertices),'faces':len(mesh.faces)}
    mesh.process(validate=True)
    if mesh.area<=0 or not len(mesh.faces):raise ValueError('Degenerate object geometry')
    parts=mesh.split(only_watertight=False)
    threshold=mesh.area*1e-6
    retained=[x for x in parts if x.area>=threshold]
    if not retained:raise ValueError('No nondegenerate object component')
    mesh=trimesh.util.concatenate(retained)
    before={'components':len(retained),'watertight':mesh.is_watertight,'euler_number':int(mesh.euler_number)}
    if len(mesh.faces)>4096 or len(mesh.vertices)>4096:
        mesh=mesh.simplify_quadric_decimation(face_count=4096)
        mesh.remove_unreferenced_vertices()
        mesh.process(validate=True)
        if before['watertight'] and not mesh.is_watertight:mesh.fill_holes()
    if len(mesh.faces)>4096 or len(mesh.vertices)>4096:
        raise ValueError('Simplification did not meet the official mesh budget')
    if not np.isfinite(mesh.vertices).all() or mesh.area<=0:
        raise ValueError('Simplification produced invalid geometry')
    warnings=[]
    if before['watertight'] and not mesh.is_watertight:warnings.append('Budget simplification opened surface boundaries')
    if int(mesh.euler_number)!=before['euler_number']:warnings.append('Budget simplification changed Euler characteristic')
    if a.strict_topology and warnings:raise ValueError('; '.join(warnings))
    a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/'normalized_shape.glb';mesh.export(path)
    report={'source_sha256':hashlib.sha256(a.mesh.read_bytes()).hexdigest(),
        'mesh_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'original':original,'before_simplification':before,
        'vertices':len(mesh.vertices),'faces':len(mesh.faces),
        'watertight':mesh.is_watertight,'euler_number':int(mesh.euler_number),
        'removed_components':len(parts)-len(retained),'metric_scale_estimated':False,
        'quality_warnings':warnings,'strict_topology':a.strict_topology,
        'budget_method':'quadric edge collapse, matching the official fixed face-budget approach'}
    (a.output/'mesh_preparation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
