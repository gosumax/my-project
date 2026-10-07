import hashlib,io,json,sys,subprocess,threading,time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from parser_core.vl_batch_dispatcher import VLBatchDispatcher
from parser_core.vl_chat_client import VLChatClient

def req(i):
 raw=f'crop{i}'.encode();return {'source':raw,'source_sha256':hashlib.sha256(raw).hexdigest(),'right_trim_px':12}
class Client:
 def __init__(self):self.starts=0;self.closes=0;self.batches=[]
 def _start(self):self.starts+=1
 def close(self):self.closes+=1
 def read_batch_bytes(self,rs):self.batches.append(rs);return [{'status':'OK','source_sha256':r['source_sha256'],'raw_text':r['source'].decode()} for r in rs]
def bounded_close(d):
 t=threading.Thread(target=d.close,daemon=True);t.start();t.join(2);assert not t.is_alive(),'dispatcher close hung'

def test_concurrent_callers_share_one_client_with_max_eight_and_partial_batch():
 c=Client();d=VLBatchDispatcher(lambda:c,batch_size=8,batch_wait_ms=40)
 with ThreadPoolExecutor(max_workers=16) as pool:
  futures=[pool.submit(d.read_bytes,**dict(source=req(i)['source'],source_sha256=req(i)['source_sha256'],right_trim_px=12)) for i in range(19)]
  results=[f.result(timeout=3) for f in futures]
 bounded_close(d);bounded_close(d)
 assert [r['raw_text'] for r in results]==[f'crop{i}' for i in range(19)]
 assert c.starts==c.closes==1 and sum(d.batch_sizes)==19 and max(d.batch_sizes)<=8 and any(n<8 for n in d.batch_sizes)
 assert not d.errors
 with pytest.raises(RuntimeError,match='closed'):d.read_bytes(**req(0))

@pytest.mark.parametrize('failure',['startup','count','foreign','exception'])
def test_dispatcher_errors_release_all_callers_and_close(failure):
 class Bad(Client):
  def _start(self):
   if failure=='startup':raise RuntimeError('startup failure')
  def read_batch_bytes(self,rs):
   if failure=='exception':raise ValueError('worker failure')
   result=super().read_batch_bytes(rs)
   if failure=='count':result.pop()
   if failure=='foreign':result[0]['source_sha256']='0'*64
   return result
 c=Bad();d=VLBatchDispatcher(lambda:c,batch_wait_ms=20)
 with ThreadPoolExecutor(max_workers=3) as pool:
  fs=[pool.submit(d.read_bytes,**req(i)) for i in range(3)]
  for f in fs:
   with pytest.raises((RuntimeError,ValueError)):f.result(timeout=3)
 bounded_close(d);assert d.errors and c.closes==1

def fake_client(tmp_path,monkeypatch,mutate):
 c=VLChatClient(__import__('pathlib').Path(sys.executable),tmp_path,'s',tmp_path/'stderr')
 stdin=io.StringIO();c.process=SimpleNamespace(stdin=stdin);monkeypatch.setattr(c,'_start',lambda:None)
 def line(timeout):
  sent=json.loads(stdin.getvalue().splitlines()[-1]);results=[{'request_id':r['request_id'],'source_sha256':r['source_sha256'],'status':'OK','raw_text':r['request_id']} for r in sent['requests']];answer={'type':'batch_result','batch_id':sent['batch_id'],'results':results};mutate(answer);return json.dumps(answer)
 monkeypatch.setattr(c,'_next_line',line);return c

def test_client_reorders_shuffled_ids_and_preserves_partial_error(tmp_path,monkeypatch):
 def mutate(a):a['results'][1]['status']='ERROR';a['results'].reverse()
 c=fake_client(tmp_path,monkeypatch,mutate);r=c.read_batch_bytes([req(0),req(1),req(2)]);assert [x['request_id'] for x in r]==['1','2','3'];assert r[1]['status']=='ERROR'
@pytest.mark.parametrize('kind',['duplicate','foreign_id','foreign_sha','partial','foreign_batch'])
def test_client_rejects_corrupt_batch_identity(tmp_path,monkeypatch,kind):
 def mutate(a):
  if kind=='duplicate':a['results'][1]['request_id']=a['results'][0]['request_id']
  if kind=='foreign_id':a['results'][1]['request_id']='foreign'
  if kind=='foreign_sha':a['results'][1]['source_sha256']='0'*64
  if kind=='partial':a['results'].pop()
  if kind=='foreign_batch':a['batch_id']='foreign'
 c=fake_client(tmp_path,monkeypatch,mutate)
 with pytest.raises(RuntimeError):c.read_batch_bytes([req(0),req(1)])

def test_real_subprocess_client_start_once_close_idempotent(tmp_path,monkeypatch):
 # Real IPC subprocess, protocol-only stub; no torch/model/GPU loading.
 original=subprocess.Popen;launched=[]
 script="import sys,json;print(json.dumps({'type':'ready'}),flush=True);[None for line in sys.stdin]"
 def popen(cmd,**kwargs):launched.append(cmd);return original([sys.executable,'-u','-c',script],**kwargs)
 monkeypatch.setattr('parser_core.vl_chat_client.subprocess.Popen',popen)
 c=VLChatClient(__import__('pathlib').Path(sys.executable),tmp_path,'sha',tmp_path/'stderr');c._start();process=c.process;c._start();assert len(launched)==1;c.close();c.close();assert process.poll()==0 and c.process is None

def test_close_read_race_never_leaves_future_after_sentinel(monkeypatch):
 c=Client();d=VLBatchDispatcher(lambda:c,batch_wait_ms=0);original=d.jobs.put;arrived=threading.Event();release=threading.Event();results=[]
 def put(item,*a,**kw):
  if item is not d._end:arrived.set();assert release.wait(2)
  return original(item,*a,**kw)
 monkeypatch.setattr(d.jobs,'put',put)
 def read():results.append(d.read_bytes(**req(0)))
 reader=threading.Thread(target=read,daemon=True);reader.start();assert arrived.wait(1)
 closer=threading.Thread(target=d.close,daemon=True);closer.start();release.set();reader.join(2);closer.join(2)
 assert not reader.is_alive() and not closer.is_alive(),'accepted read/close hung'
 assert results[0]['raw_text']=='crop0' and c.closes==1
 with pytest.raises(RuntimeError,match='closed'):d.read_bytes(**req(1))
