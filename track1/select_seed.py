"""Select automatic video seeds using grounded proposals and semantic margins."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor,AutoModel,AutoModelForZeroShotObjectDetection

DESCRIPTIONS={
 'hula_hoop':['a thin circular hula hoop','a plastic exercise hoop'],
 'big_red_bowl':['a red plastic bowl','a large red mixing bowl'],
 'iron':['a blue electric clothes iron','a metallic steam iron with blue accents'],
 'white_desk':['a white rectangular office desk with two legs','a white standing desk'],
 'black_pan':['a black frying pan with a handle','a black cooking pan'],
 'foam_grass_block':['a green and brown pixelated Minecraft toy cube','a foam cube with green and brown squares'],
 'paint_roller':['a white paint roller with a blue handle','a paint roller tool with a handle'],
 'pink_foam_roll':['a pink cylindrical foam exercise roller','a pink foam cylinder with a hole'],
 'short_wood_stool':['a short wooden four-legged stool','a wooden square stool'],
 'white_laptop_cart':['a white rolling laptop cart with a central column and wheels','a white mobile laptop stand with casters'],
}
QUERIES={
 'iron':['a clothes iron.','a blue electric iron.'],
 'foam_grass_block':['a green brown cube.','a Minecraft block.','a foam cube.'],
 'white_desk':['a white desk.','an office desk.'],
 'white_laptop_cart':['a rolling laptop stand.','a white cart with wheels.'],
 'paint_roller':['a paint roller.','a paint roller with handle.'],
}
DISTRACTORS=['a camera tripod','an office chair','a curtain','a person','a slipper',
             'a long metal rail','a wall','a wooden floor','a table lamp','an ironing board']


def select_seed(frames:Path,frame_count:int,height:int,width:int,prompt:str,object_class:str):
    if object_class not in DESCRIPTIONS:raise ValueError('Unknown released object class')
    detector_id='IDEA-Research/grounding-dino-tiny'
    processor=AutoProcessor.from_pretrained(detector_id)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to('cuda').eval()
    candidates=[];crops=[]
    queries=list(dict.fromkeys(QUERIES.get(object_class,[DESCRIPTIONS[object_class][0]+'.'])+[prompt]))
    sampled=sorted(set(np.linspace(0,frame_count-1,12).astype(int).tolist()+[min(frame_count-1,x) for x in (0,15,30,60,120)]))
    for frame in sampled:
        image=Image.open(frames/f'{frame:06d}.jpg').convert('RGB')
        def detect(text):
            inputs=processor(images=image,text=text,return_tensors='pt').to('cuda')
            with torch.inference_mode():output=detector(**inputs)
            return processor.post_process_grounded_object_detection(output,inputs.input_ids,
                threshold=.12,text_threshold=.12,target_sizes=[(height,width)])[0]
        people=detect('a person.')
        if not len(people['boxes']):continue
        human_id=int(people['scores'].argmax())
        human={'box':people['boxes'][human_id].cpu().tolist(),'confidence':float(people['scores'][human_id])}
        seen=[]
        for query in queries:
            result=detect(query)
            for box,confidence in zip(result['boxes'].cpu().numpy(),result['scores'].cpu().numpy()):
                x0,y0,x1,y1=box
                if x1-x0<8 or y1-y0<8:continue
                if any(np.max(np.abs(box-prev))<3 for prev in seen):continue
                seen.append(box)
                margin=.06*max(x1-x0,y1-y0)
                crop=image.crop((max(0,int(x0-margin)),max(0,int(y0-margin)),min(width,int(x1+margin+1)),min(height,int(y1+margin+1))))
                candidates.append({'frame':frame,'human':human,'object':{'box':box.astype(float).tolist(),'confidence':float(confidence)},'query':query})
                crops.append(crop)
    del processor,detector;torch.cuda.empty_cache()
    if not candidates:raise ValueError('No automatic grounded candidates')
    model_id='google/siglip2-base-patch16-224'
    processor=AutoProcessor.from_pretrained(model_id)
    model=AutoModel.from_pretrained(model_id).to('cuda').eval()
    target_text=DESCRIPTIONS[object_class]
    other_text=[text for key,values in DESCRIPTIONS.items() if key!=object_class for text in values]+DISTRACTORS
    texts=['This is a photo of '+text+'.' for text in target_text+other_text]
    for start in range(0,len(crops),16):
        inputs=processor(text=texts,images=crops[start:start+16],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):logits=model(**inputs).logits_per_image.float().cpu().numpy()
        if not np.isfinite(logits).all():raise ValueError('Nonfinite semantic ranking logits')
        for offset,row in enumerate(logits):
            item=candidates[start+offset]
            positive=float(row[:len(target_text)].max());negative=float(row[len(target_text):].max())
            item.update(semantic_margin=positive-negative,semantic_target_logit=positive,
                        semantic_best_label=(target_text+other_text)[int(row.argmax())],semantic_model=model_id)
            item['score']=positive-negative+.1*positive+.25*item['object']['confidence']
    eligible=[c for c in candidates if c['semantic_margin']>0]
    if not eligible:raise ValueError('No candidate beats the released alternative classes/distractors')
    seed=max(eligible,key=lambda c:c['score'])
    del model,processor;torch.cuda.empty_cache()
    return seed,candidates,detector_id
