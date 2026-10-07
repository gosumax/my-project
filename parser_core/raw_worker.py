"""Experimental raw-gray transport; same configured Tiny recognition API."""
import base64, hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import cv2
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'

def decode_raw(item):
    raw=base64.b64decode(item['raw_base64'],validate=True)
    if len(raw)!=item['width']*item['height'] or hashlib.sha256(raw).hexdigest()!=item['exact_hash']:
        raise ValueError('Raw gray size/hash mismatch')
    return np.frombuffer(raw,np.uint8).reshape(item['height'],item['width'])

def forbid_codecs():
    def forbidden(*args,**kwargs):raise RuntimeError('Image codecs forbidden in raw timeline')
    for name in ('imread','imwrite','imencode','imdecode'):setattr(cv2,name,forbidden)

def main():
    forbid_codecs()
    config=json.loads(Path(sys.argv[1]).read_text('utf-8'))
    weights=Path(config['model_dir'])/'inference.pdiparams'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=config['weights_sha256']:
        raise ValueError('Tiny weights mismatch')
    from paddleocr import TextRecognition
    model=TextRecognition(model_name=config['model_name'],device=config['device'])
    sys.stdout.reconfigure(encoding='utf-8');sys.stdin.reconfigure(encoding='utf-8')
    print(json.dumps(dict(type='ready',weights_sha256=config['weights_sha256'],transport='raw-gray-u8')),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        try:
            images=[]
            if not 1<=len(request['items'])<=8:raise ValueError('Invalid batch length')
            for item in request['items']:
                gray=decode_raw(item)
                images.append(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR))
            started=time.perf_counter()
            predictions=list(model.predict(images,batch_size=8))
            if len(predictions)!=len(images):raise ValueError('Missing OCR prediction')
            results=[dict(item_id=item['state_id'],raw_text=(p.get('rec_text') or '').strip(),score=float(p.get('rec_score',0))) for item,p in zip(request['items'],predictions)]
            reply=dict(type='result',request_id=request['request_id'],results=results,inference_seconds=time.perf_counter()-started)
        except Exception:
            reply=dict(type='result',request_id=request['request_id'],results=[],error=traceback.format_exc())
        print(json.dumps(reply,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
