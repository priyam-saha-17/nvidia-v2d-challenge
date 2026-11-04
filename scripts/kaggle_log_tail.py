"""Continuously drain a Kaggle log stream and maintain a compact latest snapshot."""
import argparse,json,time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
p=argparse.ArgumentParser()
p.add_argument('notebook')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
api=KaggleApi();api.authenticate()
last_write=0.;count=0;tail='';latest_time=None;last_event=None
with a.output.with_suffix('.jsonl').open('a') as raw:
 try:
  for event in api.kernels_logs_stream(a.notebook):
   if event==last_event:continue
   last_event=event;count+=1;latest_time=event.get('time')
   raw.write(json.dumps(event)+'\n')
   tail=(tail+event.get('data',''))[-6000:]
   if time.monotonic()-last_write>5:
    raw.flush()
    snapshot={'notebook':a.notebook,'received_events':count,'latest_event_time':latest_time,'tail':tail,'stream_open':True}
    temporary=a.output.with_suffix('.tmp');temporary.write_text(json.dumps(snapshot));temporary.replace(a.output)
    last_write=time.monotonic()
 finally:
  a.output.write_text(json.dumps({'notebook':a.notebook,'received_events':count,'latest_event_time':latest_time,'tail':tail,'stream_open':False}))
print('Log stream ended:',a.notebook,count,flush=True)
