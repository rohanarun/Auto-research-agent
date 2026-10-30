"""Trusted host-only provider adapter. API keys never enter a contestant environment."""
import base64,json,os,secrets,urllib.error,urllib.parse,urllib.request
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args):return None

def request(plane,path,method='GET',body=None,timeout=45,raw=False):
 assert path.startswith('/') and '://' not in path
 base=os.environ.get('DAYTONA_API_URL','https://app.daytona.io/api') if plane=='control' else os.environ.get('DAYTONA_TOOLBOX_URL','https://proxy.app.daytona.io/toolbox')
 if not base.startswith('https://'):raise ValueError('Provider control requires HTTPS')
 key=os.environ['DAYTONA_API_KEY'];headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'};payload=json.dumps(body).encode() if body is not None else None
 if isinstance(body,dict) and '__multipart_file_b64' in body:
  boundary='AutoResearch'+secrets.token_hex(16);file_bytes=base64.b64decode(body['__multipart_file_b64'])
  payload=('--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="bundle.tar.gz"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()+file_bytes+('\r\n--'+boundary+'--\r\n').encode();headers['Content-Type']='multipart/form-data; boundary='+boundary
 try:
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
  with opener.open(urllib.request.Request(base+path,data=payload,headers=headers,method=method),timeout=timeout) as r:
   data=r.read();return {'ok':True,'data':{'base64':base64.b64encode(data).decode()} if raw else (json.loads(data) if data else None)}
 except urllib.error.HTTPError as e:return {'ok':False,'http_status':e.code,'error':e.read(1500).decode(errors='replace')}
 except Exception as e:return {'ok':False,'error_type':type(e).__name__,'outcome':'unknown_for_mutations'}
def inventory():
 r=request('control','/sandbox')
 if not r['ok']:raise RuntimeError(str(r))
 d=r['data'];return d.get('items',[]) if isinstance(d,dict) else d
def execute(sid,command,timeout=45):return request('toolbox','/'+sid+'/process/execute','POST',{'command':command,'timeout':timeout},timeout+5)
def upload(sid,path,data):return request('toolbox','/'+sid+'/files/upload-v2?path='+urllib.parse.quote(path,safe=''),'POST',{'__multipart_file_b64':base64.b64encode(data).decode()})
def download(sid,path):
 r=request('toolbox','/'+sid+'/files/download?path='+urllib.parse.quote(path,safe=''),raw=True)
 if not r.get('ok'):raise RuntimeError('Provider file download failed '+str(r))
 return base64.b64decode(r['data']['base64'])
def create(name,image):
 for s in inventory():
  if s.get('name')==name:return s
 defaults=json.loads(os.environ.get('AUTO_RESEARCH_GPU_RESOURCES','{"cpu":16,"memory":100,"disk":300,"gpu":1,"gpuType":["H200"]}'))
 if set(defaults)-{'cpu','memory','disk','gpu','gpuType'}:raise ValueError('Resource configuration contains unsupported fields')
 body={**defaults,'name':name,'buildInfo':{'dockerfileContent':'FROM '+image+'\n'},'public':False,'networkBlockAll':True,'autoStopInterval':5,'autoPauseInterval':0,'autoDeleteInterval':0,'labels':{'owner':'auto-research-benchmark','purpose':'offline-evaluation'}}
 r=request('control','/sandbox','POST',body)
 if r.get('ok'):return r['data']
 for s in inventory():
  if s.get('name')==name:return s
 raise RuntimeError('Creation unconfirmed; do not blindly retry: '+str(r))
def delete(sid):return request('control','/sandbox/'+sid,'DELETE')
