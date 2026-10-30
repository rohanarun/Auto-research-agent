"""Tool-free model transports. All commands execute through the benchmark broker."""
from pathlib import Path
import json,os,time,urllib.error,urllib.request

def validate_action(action):
 if set(action)!={'command','done','note'} or not isinstance(action['command'],str) or not isinstance(action['done'],bool) or not isinstance(action['note'],str):raise ValueError('Invalid structured action')
 return action

def sample(run,prompt,model,effort,index,timeout,schema):
 config=json.loads(Path(os.environ['AUTO_RESEARCH_PLANNER_CONFIG']).read_text());kind=config['kind']
 if kind not in ['responses','chat-completions']:raise ValueError('Unsupported API transport')
 params=config.get('parameters',{})
 if any(k in params for k in ['tools','tool_choice','model','input','messages','text','response_format','functions','function_call','previous_response_id','conversation','store','background','stream']):raise ValueError('Transport parameters cannot override the tool-free action contract')
 key=os.environ[config.get('key_env','MODEL_API_KEY')];endpoint=config['url']
 if not endpoint.startswith('https://') and not config.get('allow_http_for_tests'):raise ValueError('Inference endpoint must use HTTPS')
 if kind=='responses':
  payload={'model':model,'input':prompt,'reasoning':{'effort':effort},'text':{'format':{'type':'json_schema','name':'research_action','strict':True,'schema':schema}},'tools':[],'store':False,**params}
 else:
  payload={'model':model,'messages':[{'role':'system','content':'Return only one JSON object with command (string), done (boolean), and note (string), with no extra keys. Never invoke native tools. Required action schema: '+json.dumps(schema)},{'role':'user','content':prompt}],'response_format':{'type':'json_object'},'tools':[],'store':False,**params}
  if config.get('send_reasoning_effort',True):payload['reasoning_effort']=effort
 (run/f'model-{index:04d}.prompt.txt').write_text(prompt)
 request=urllib.request.Request(endpoint,data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
 class NoRedirect(urllib.request.HTTPRedirectHandler):
  def redirect_request(self,*args):return None
 opener=urllib.request.build_opener(NoRedirect())
 with opener.open(request,timeout=max(1,timeout)) as response:body=json.load(response)
 (run/f'model-{index:04d}.response.json').write_text(json.dumps(body,indent=2)+'\n')
 if kind=='responses':
  text=[]
  for item in body.get('output',[]):
   if item.get('type') not in ['message','reasoning']:raise RuntimeError('INVALID_ISOLATION: provider returned a tool call')
   for part in item.get('content',[]):
    if part.get('type')=='output_text':text.append(part['text'])
  content='\n'.join(text);u=body.get('usage',{});usage={'input_tokens':u.get('input_tokens',0),'output_tokens':u.get('output_tokens',0),'cached_input_tokens':u.get('input_tokens_details',{}).get('cached_tokens',0),'reasoning_output_tokens':u.get('output_tokens_details',{}).get('reasoning_tokens',0)}
 else:
  message=body['choices'][0]['message']
  if message.get('tool_calls') or message.get('function_call'):raise RuntimeError('INVALID_ISOLATION: provider returned a tool call')
  content=message['content'];u=body.get('usage',{});usage={'input_tokens':u.get('prompt_tokens',0),'output_tokens':u.get('completion_tokens',0),'cached_input_tokens':u.get('prompt_tokens_details',{}).get('cached_tokens',0)}
 return validate_action(json.loads(content)),usage
