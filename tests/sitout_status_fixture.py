"""Native source fixtures exercise source reproof, rather than mocked authority."""
import base64
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from parser_core.simple_saved_policy import ROOT, SIT_OUT_GATE_AUTHORITY, _sitout_native_source
from parser_core.sitout_text_pattern import SitOutTextPattern, MODEL


def status_source(directory, *, table=1, session='s', seat='seat1', frame=150,
                  hand='123456', window=(100,200), sitout=True):
    box=json.loads((ROOT/'configs/stack_roi_refined.json').read_text('utf8'))['slots'][f'slot_{table:02d}']['seats'][seat]['stack']
    x,y,X,Y=box
    gray=np.zeros((Y-y,X-x),np.uint8)
    if sitout:
        templates=json.loads(MODEL.read_text('utf8'))['templates']
        glyph=next(np.asarray(t['template_gray'],np.uint8) for t in templates
                   if t['template_shape'][0]<=Y-y and t['template_shape'][1]<=X-x)
        gray[:glyph.shape[0],:glyph.shape[1]]=glyph
    pixels=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    ok,png=cv2.imencode('.png',pixels);assert ok
    raw=dict(table_id=table,session_id=session,hand_id=hand,seat=seat,source_frame=frame,
        field='stack',roi=box,pixel_sha256=hashlib.sha256(pixels[:,:,::-1].tobytes()).hexdigest(),
        source_pts=frame*512,time_base=[1,15360],png_base64=base64.b64encode(png).decode('ascii'))
    path=Path(directory)/'start_stack_capture.jsonl'
    path.write_text(json.dumps(raw)+'\n',encoding='utf8')
    route,actual_frame,native,ref=_sitout_native_source('SAVED_STACK_CAPTURE',path,line=1)
    return dict(table_id=table,session_id=session,hand_id=hand,seat=seat,frame=frame,
        hand_window=list(window),status='SIT_OUT',authority=SIT_OUT_GATE_AUTHORITY,
        source=ref,status_classifier=SitOutTextPattern().evaluate(native))
