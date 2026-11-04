"""Automatic Track 1 video masks from released text prompts; no manual labels.

Grounding DINO chooses a shared human/object seed frame. SAM2 propagates in both
directions. Output H5 follows the official wild-video mask schema exactly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import h5py
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2_video_predictor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--object-prompt',required=True)
    p.add_argument('--sam2-checkpoint',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-frames',type=int,required=True)
    a = p.parse_args()
    suffix='.0.color.mp4'
    if not a.video.name.endswith(suffix):
        p.error('Name the input <sequence>.0.color.mp4')
    sequence=a.video.name[:-len(suffix)]
    a.output.mkdir(parents=True,exist_ok=True)
    frames=a.output/'frames';frames.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(a.video));fps=float(cap.get(cv2.CAP_PROP_FPS))
    index=0
    while True:
        ok,image=cap.read()
        if not ok:break
        if not cv2.imwrite(str(frames/f'{index:06d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,95]):
            raise OSError('Frame write failed')
        height,width=image.shape[:2];index+=1
    cap.release()
    if index!=a.expected_frames:
        raise ValueError(f'Decoded {index} frames, expected {a.expected_frames}')
    detector_id='IDEA-Research/grounding-dino-tiny'
    processor=AutoProcessor.from_pretrained(detector_id)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to('cuda').eval()
    candidates=[]
    for frame in sorted({min(index-1,t) for t in (0,15,30,60,120)}):
        image=Image.open(frames/f'{frame:06d}.jpg').convert('RGB')
        pair=[]
        for prompt in ('a person.',a.object_prompt):
            inputs=processor(images=image,text=prompt,return_tensors='pt').to('cuda')
            with torch.inference_mode():prediction=detector(**inputs)
            result=processor.post_process_grounded_object_detection(prediction,inputs.input_ids,
                threshold=0.15,text_threshold=0.15,target_sizes=[(height,width)])[0]
            keep=[i for i,b in enumerate(result['boxes']) if (b[2]-b[0])>=8 and (b[3]-b[1])>=8]
            if not keep:break
            best=max(keep,key=lambda i:float(result['scores'][i]))
            pair.append({'box':result['boxes'][best].detach().cpu().tolist(),
                         'confidence':float(result['scores'][best])})
        if len(pair)==2:
            candidates.append({'frame':frame,'human':pair[0],'object':pair[1],
                               'score':pair[0]['confidence']+pair[1]['confidence']})
    del detector,processor
    torch.cuda.empty_cache()
    if not candidates:
        raise ValueError('No automatic human/object seed pair found; do not fabricate masks')
    seed=max(candidates,key=lambda c:c['score'])
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',
        a.sam2_checkpoint,device='cuda',apply_postprocessing=False)
    seen=set();areas={};preview_ids=set(np.linspace(0,index-1,8).astype(int).tolist())
    with torch.inference_mode():
        state=predictor.init_state(str(frames),offload_video_to_cpu=True,offload_state_to_cpu=True)
        for obj_id,role in ((0,'human'),(1,'object')):
            predictor.add_new_points_or_box(state,frame_idx=seed['frame'],obj_id=obj_id,
                box=np.asarray(seed[role]['box'],dtype=np.float32))
        with h5py.File(a.output/'masks.h5','w') as h:
            h.attrs.update(schema='v2d.cari4d.wild_masks.v1',sequence=sequence,
                           frame_count=index,width=width,height=height)
            group=h.create_group(sequence)
            for reverse in (False,True):
                for frame,obj_ids,logits in predictor.propagate_in_video(state,
                        start_frame_idx=seed['frame'],reverse=reverse):
                    if frame in seen:continue
                    if not torch.isfinite(logits).all():
                        raise ValueError(f'Non-finite SAM2 output at frame {frame}')
                    masks={int(obj):logits[i,0].detach().cpu().numpy()>0 for i,obj in enumerate(obj_ids)}
                    if set(masks)!={0,1}:raise ValueError('Incomplete tracked object set')
                    for obj,role in ((0,'person_mask'),(1,'obj_rend_mask')):
                        group.create_dataset(f'{frame:06d}-k0.{role}.png',
                            data=masks[obj].astype(np.uint8)*255,compression='lzf',shuffle=True)
                    areas[frame]=[int(masks[0].sum()),int(masks[1].sum())]
                    if frame in preview_ids:
                        rgb=np.asarray(Image.open(frames/f'{frame:06d}.jpg').convert('RGB')).copy()
                        rgb[masks[0]]=(rgb[masks[0]]*.6+np.array([40,130,255])*.4).astype(np.uint8)
                        rgb[masks[1]]=(rgb[masks[1]]*.5+np.array([255,70,40])*.5).astype(np.uint8)
                        Image.fromarray(rgb).save(a.output/f'preview_{frame:06d}.jpg')
                    if frame==seed['frame']:
                        rgba=np.asarray(Image.open(frames/f'{frame:06d}.jpg').convert('RGBA')).copy()
                        rgba[:,:,3]=masks[1].astype(np.uint8)*255
                        ys,xs=np.where(masks[1])
                        if len(xs)<32:raise ValueError('Object seed mask is empty or degenerate')
                        margin=int(.15*max(xs.max()-xs.min(),ys.max()-ys.min()))+1
                        box=(max(0,xs.min()-margin),max(0,ys.min()-margin),
                             min(width,xs.max()+margin+1),min(height,ys.max()+margin+1))
                        Image.fromarray(rgba).crop(box).save(a.output/'object_observation.png')
                    seen.add(frame)
                    if len(seen)%50==0:print(f'Masks: {len(seen)}/{index}',flush=True)
    if seen!=set(range(index)):raise ValueError('Masks do not cover every original frame')
    report={'sequence':sequence,'frames':index,'fps':fps,'width':width,'height':height,
        'seed':seed,'seed_candidates':candidates,'detector':detector_id,
        'object_prompt':a.object_prompt,'manual_annotations_used':False,
        'frame_mask_areas':{str(k):v for k,v in sorted(areas.items())},
        'video_sha256':hashlib.sha256(a.video.read_bytes()).hexdigest()}
    (a.output/'segmentation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Automatic segmentation complete:',a.output)


if __name__=='__main__':main()
