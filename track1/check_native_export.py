"""Numerical MHR exporter diagnostic using synthetic parameters, never submitted.

Decode the actual licensed MHR model and check known translations against the
competition's independent TorchScript coordinate convention. Temporary fixture
arrays are removed; only an explicitly labelled diagnostic report is retained.
"""
import argparse,json,shutil,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import torch
from lib_mhr.rotations import rotmat_to_6d
from sam_3d_body.models.modules.mhr_utils import compact_model_params_to_cont_body


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--actual-mesh',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='mhr-export-diagnostic-') as directory:
        temporary=Path(directory);frames=3
        translation=np.array([[0.,0.,2.],[.1,-.05,2.2],[-.2,.15,1.8]],dtype=np.float32)
        params={'mhr_global_rot6d':rotmat_to_6d(torch.eye(3).expand(frames,-1,-1)).numpy(),
            'mhr_trans':translation,'mhr_body_pose_cont':compact_model_params_to_cont_body(torch.zeros(frames,133)).numpy(),
            'mhr_hand':np.zeros((frames,108),np.float32),'mhr_shape':np.zeros((frames,45),np.float32),
            'mhr_scale':np.zeros((frames,28),np.float32),'pose_abs':np.tile(np.eye(4,dtype=np.float32),(frames,1,1))}
        bundle={'frames':[str(i) for i in range(frames)],'pr':params,
                'metadata':{'ground_truth_used':False,'synthetic_parameter_diagnostic':True}}
        torch.save(bundle,temporary/'fixture.pth')
        mesh=temporary/'output_aligned.glb';shutil.copy2(a.actual_mesh,mesh)
        subprocess.run([sys.executable,str(Path(__file__).with_name('export_cari4d.py')),
            '--bundle',str(temporary/'fixture.pth'),'--mesh',str(mesh),
            '--mhr-assets-root',str(a.assets),'--episode','0','--frames',str(frames),
            '--output',str(temporary/'diagnostic_only'),'--device','cpu','--batch-size','3'],check=True)
        output=temporary/'diagnostic_only/episode_000000.npz'
        model=torch.jit.load(str(a.assets/'checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt'),map_location='cpu').eval()
        with np.load(output) as values,torch.inference_mode():
            native=torch.from_numpy(np.concatenate([values['pose'],np.tile(values['scales'],(frames,1))],axis=1))
            shape=torch.from_numpy(values['shape']).expand(frames,-1).contiguous()
            vertices,_=model(shape,native,torch.zeros(frames,72),True)
            vertices=vertices.numpy()*np.array([1.,-1.,-1.],np.float32)/100.
        expected=translation[1:]-translation[0]
        measured=vertices[1:]-vertices[0]
        error=float(np.max(np.linalg.norm(measured-expected[:,None,:],axis=-1)))
        if error>5e-5:raise ValueError(f'Independent known-translation test failed: {error} m')
        export_report=json.loads((temporary/'diagnostic_only/episode_000000.report.json').read_text())
        report={'diagnostic_only':True,'synthetic_parameters':True,'challenge_episode_reconstructed':False,
            'frames':frames,'max_known_translation_error_m':error,
            'native_round_trip_error_m':export_report['max_native_round_trip_error_m'],
            'mhr_model_sha256':export_report['mhr_model_sha256'],'fixture_arrays_retained':False}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
