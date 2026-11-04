"""Capture a bounded, read-only notebook log snapshot without downloading files."""
import argparse
import json
from pathlib import Path
import signal
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

p=argparse.ArgumentParser()
p.add_argument('notebook')
p.add_argument('--seconds',type=int,default=15)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if not 1<=a.seconds<=45:p.error('Snapshot duration must be 1–45 seconds')
api=KaggleApi();api.authenticate();events=[]
def stop(*_):raise TimeoutError('Snapshot deadline reached')
signal.signal(signal.SIGALRM,stop);signal.alarm(a.seconds)
try:
    for event in api.kernels_logs_stream(a.notebook):events.append(event)
except (TimeoutError,requests.exceptions.ConnectionError):
    pass
finally:
    signal.alarm(0)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(events))
text=''.join(e.get('data','') for e in events)
print(text[-4500:] if events else 'No log events received during bounded snapshot')
