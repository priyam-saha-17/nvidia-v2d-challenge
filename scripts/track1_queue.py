"""Schedule only prepared Track 1 preprocessing notebooks within two GPU slots.

No scoring submissions, credential changes, Track 2/3 jobs, or automatic retries
of failed notebook executions. State is retained for interrupted monitoring.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from requests.exceptions import HTTPError

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--state',type=Path,default=ROOT/'outputs/track1_queue.json')
p.add_argument('--watch',action='store_true')
p.add_argument('--hours',type=float,default=8)
a=p.parse_args()
if not 0<a.hours<=12:p.error('Queue duration must be within 12 hours')
if a.state.exists():jobs=json.loads(a.state.read_text())
else:
 jobs=[
  {'name':'reconstruction','directory':'track1_inference','started':True,'dependencies':[]},
  {'name':'segments-a','directory':'track1_segments_a','started':True,'dependencies':[]},
  {'name':'segments-b','directory':'track1_segments_b','started':False,'dependencies':[]},
  {'name':'shapes-a','directory':'track1_shapes_a','started':False,'dependencies':['segments-a']},
  {'name':'shapes-b','directory':'track1_shapes_b','started':False,'dependencies':['segments-b']},
 ]
for job in jobs:
 meta=json.loads((ROOT/'notebooks'/job['directory']/'kernel-metadata.json').read_text())
 job['slug']=meta['id']
 if not job['slug'].startswith('priyamsaha17/v2d-track1-'):
  raise ValueError('Refusing a job outside the authorized Track 1 queue')
api=KaggleApi();api.authenticate();deadline=time.monotonic()+a.hours*3600
previous=None
while True:
 try:
  for job in jobs:
   if job['started'] and job.get('status') not in {'COMPLETE','ERROR','CANCELLED'}:
    response=api.kernels_status(job['slug'])
    job['status']=getattr(response.status,'name',str(response.status).split('.')[-1])
    if getattr(response,'failure_message',None):job['failure_message']=response.failure_message
  by_name={j['name']:j for j in jobs}
  active=sum(j.get('gpu',True) and j.get('status') in {'RUNNING','QUEUED','INITIALIZED'} for j in jobs)
  for job in jobs:
   if active>=2:break
   if job['started']:continue
   if not all(by_name[name].get('status')=='COMPLETE' for name in job['dependencies']):continue
   datasets_ready=True
   for dataset in job.get('requires_datasets',[]):
    try:datasets_ready &= api.dataset_status(dataset)=='ready'
    except HTTPError as error:
     if error.response is None or error.response.status_code!=404:raise
     datasets_ready=False
   if not datasets_ready:continue
   result=subprocess.run([sys.executable,'-m','kaggle','kernels','push','-p',str(ROOT/'notebooks'/job['directory']),'--accelerator','NvidiaTeslaT4'],capture_output=True,text=True)
   output=result.stdout+result.stderr
   if 'successfully pushed' in output:
    job.update(started=True,status='QUEUED');active+=1
    print('STARTED',job['slug'],flush=True)
   else:
    job['launch_message']=output[-2000:]
    if 'Maximum batch GPU session count' not in output:
     raise RuntimeError('Queue launch failed: '+output[-2000:])
    break
  a.state.parent.mkdir(exist_ok=True,parents=True)
  a.state.write_text(json.dumps(jobs,indent=2)+'\n')
  state=[(j['name'],j.get('status','PENDING')) for j in jobs]
  if state!=previous:print(json.dumps(state),flush=True);previous=state
 except Exception as error:
  print('QUEUE_CHECK_FAILED',type(error).__name__,str(error),flush=True)
  # Persist the current notebook identities before returning control for diagnosis.
  a.state.write_text(json.dumps(jobs,indent=2)+'\n')
  raise
 if not a.watch or time.monotonic()>=deadline:break
 if all(j.get('status') in {'COMPLETE','ERROR','CANCELLED'} for j in jobs):break
 if any(not j['started'] and any(by_name[d].get('status') in {'ERROR','CANCELLED'} for d in j['dependencies']) for j in jobs):
  print('A preprocessing dependency failed; manual diagnosis required',flush=True);break
 time.sleep(45)
