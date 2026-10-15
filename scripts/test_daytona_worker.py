"""Filesystem and lifecycle tests for the shipped remote bridge, without a model."""
from pathlib import Path
import importlib.util,tempfile,subprocess,json,base64,os,tarfile,hashlib,sys,time
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bridge',root/'Resources/DaytonaWorker.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
with tempfile.TemporaryDirectory(dir=root/'tmp',prefix='daytona-worker-') as name:
 p=Path(name);b.R=p/'private';b.CO=p/'checkout';b.CO.mkdir();b.R.mkdir()
 (b.CO/'RESEARCH_CHECKPOINT.md').write_bytes(b'atomic UTF8 \xe2\x80\x94 checkpoint')
 (b.CO/'RESEARCH_CONTEXT').mkdir();(b.CO/'RESEARCH_CONTEXT/topic.md').write_text('topic')
 snap=b.checkpoint();assert base64.b64decode(snap['checkpoint'])==(b.CO/'RESEARCH_CHECKPOINT.md').read_bytes()
 outside=p/'outside';outside.write_text('private');(b.CO/'RESEARCH_CHECKPOINT.md').unlink();(b.CO/'RESEARCH_CHECKPOINT.md').symlink_to(outside)
 try:b.checkpoint();raise RuntimeError('escaped checkpoint accepted')
 except AssertionError:pass
 (b.CO/'RESEARCH_CHECKPOINT.md').unlink();(b.CO/'RESEARCH_CHECKPOINT.md').write_text('restored')
 b.put_context({'round':'round-1','files':{'context.md':base64.b64encode(b'checkpoint').decode()}})
 try:b.put_context({'round':'round-1','files':{'../../escape':base64.b64encode(b'bad').decode()}});raise RuntimeError('escaped context accepted')
 except AssertionError:pass
 state=b.R/'providers/session/state';state.mkdir(parents=True);(state/'auth.json').write_text('DO_NOT_EXPORT');(state/'session.jsonl').write_text('retained history')
 entries=b.manifest();assert not any(k.endswith('/auth.json') for k in entries);assert any(k.endswith('/session.jsonl') for k in entries)
 # Use relative paths under /, as production manifests do. No credential file is packed.
 receipt=b.pack();archive=b.R/'export/checkout-state.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
 with tarfile.open(archive) as tf:
  names=tf.getnames();assert set(names)==set(entries);assert not any(n.endswith('/auth.json') for n in names)
 if sys.platform=='linux':
  keeper=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
  owned=subprocess.Popen([sys.executable,'-c','import subprocess,time;subprocess.Popen(["sleep","60"]);time.sleep(60)'])
  time.sleep(.1);receipt=b.terminate(owned);owned.wait();assert receipt['remaining_pids']==[];assert keeper.poll() is None;keeper.terminate();keeper.wait()
 print('PASS: verbatim checkpoints, boundary rejection, auth exclusion and complete archive; Linux cancellation tested' if sys.platform=='linux' else 'PASS: verbatim checkpoints, boundary rejection, auth exclusion and complete archive; Linux cancellation requires cloud test')
