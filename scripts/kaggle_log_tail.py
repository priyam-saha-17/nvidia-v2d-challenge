"""Drain Kaggle logs with reconnection and maintain a compact latest snapshot."""
import argparse,json,time
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
p=argparse.ArgumentParser()
p.add_argument('notebook')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
api=KaggleApi();api.authenticate()
last_write=0.;count=0;tail='';latest_time=None;seen=set();reconnections=0

def snapshot(open_stream):
    value={'notebook':a.notebook,'received_events':count,'latest_event_time':latest_time,
           'tail':tail,'stream_open':open_stream,'reconnections':reconnections}
    temporary=a.output.with_suffix('.tmp');temporary.write_text(json.dumps(value));temporary.replace(a.output)

with a.output.with_suffix('.jsonl').open('a') as raw:
    try:
        while True:
            try:
                for event in api.kernels_logs_stream(a.notebook):
                    identity=(event.get('stream_name'),event.get('time'),event.get('data'))
                    if identity in seen:continue
                    seen.add(identity);count+=1;latest_time=event.get('time')
                    raw.write(json.dumps(event)+'\n')
                    tail=(tail+event.get('data',''))[-6000:]
                    if time.monotonic()-last_write>5:
                        raw.flush();snapshot(True);last_write=time.monotonic()
                break
            except requests.exceptions.RequestException as error:
                reconnections+=1;snapshot(True)
                if reconnections>100:raise
                print('Reconnecting log feed',reconnections,type(error).__name__,flush=True)
                time.sleep(2)
    finally:
        snapshot(False)
print('Log stream ended:',a.notebook,count,flush=True)
