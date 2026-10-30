"""Contestant-side file-only GPU request bridge. No networking or credentials."""
from pathlib import Path
import argparse,hashlib,json,time,uuid
p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('--poll');p.add_argument('--cases',default='0',help='Comma-separated public case indices; default is a short diagnostic on case 0');a=p.parse_args()
work=Path.cwd().resolve();queue=work/'.gpu';queue.mkdir(exist_ok=True)
if a.poll:
 f=queue/(a.poll+'.result.json')
 print(f.read_text() if f.exists() else json.dumps({'status':'pending','job_id':a.poll}))
else:
 candidate=a.candidate.resolve()
 if not candidate.is_relative_to(work) or not candidate.is_file() or candidate.stat().st_size>2_000_000:raise SystemExit('Candidate must be a Python file inside this workspace, at most 2 MB')
 job=uuid.uuid4().hex;data=candidate.read_bytes();(queue/(job+'.candidate.py')).write_bytes(data)
 (queue/(job+'.request.json')).write_text(json.dumps({'job_id':job,'sha256':hashlib.sha256(data).hexdigest(),'created_at_unix':time.time(),'cases':[int(i) for i in a.cases.split(',')]})+'\n')
 print(json.dumps({'status':'queued','job_id':job,'poll_command':'python gpu_test.py '+str(a.candidate)+' --poll '+job}))
