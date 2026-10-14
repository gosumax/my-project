"""Small retained native source examples and isolated consumer authority guards."""
import base64,copy,json
from pathlib import Path
import cv2,numpy as np,pytest
from parser_core import adjacent_stack_carry as carry

@pytest.fixture
def native():return json.loads((Path(__file__).parent/'fixtures/adjacent_stack_carry_native_headers.json').read_text('utf-8'))

def source_case(native,index):
    c=native['headers'][index];registry=object.__new__(carry.SavedCarrySources)
    registry.physical=c['physical'];registry.inputs=[];registry.admission_clocks=c['clocks']
    pixels=cv2.imdecode(np.frombuffer(base64.b64decode(c['admitted']['png_base64'],validate=True),np.uint8),0)
    return c,registry,pixels

@pytest.mark.parametrize('index',[0,1,2])
def test_native_anchor_and_sidecar_source_contract(native,index):
    before=copy.deepcopy(native);c,r,p=source_case(native,index)
    r.verify_header_source(c['confirmed'],c['admitted'],p,require_physical=c['require_physical'])
    assert native==before

@pytest.mark.parametrize('mutation',['missing_clock','wrong_clock_pts','chat_bounds','physical_pixels','physical_hash','boundary_missing_physical'])
def test_native_source_changes_rejected(native,mutation):
    index=1 if mutation in ['missing_clock','wrong_clock_pts','chat_bounds'] else 0
    c,r,p=source_case(native,index)
    if mutation=='missing_clock':r.admission_clocks=[]
    elif mutation=='wrong_clock_pts':r.admission_clocks[0]['source_pts']+=1
    elif mutation=='chat_bounds':r.admission_clocks[0]['source_chat_box'][2]-=1
    elif mutation=='boundary_missing_physical':r.physical=[]
    else:
        row=next(x for x in r.physical if x['source_span_index']==c['confirmed']['source_span_index'])
        if mutation=='physical_hash':row['row_pixel_sha256']='0'*64
        else:
            raw=bytearray(base64.b64decode(row['row_pixels_base64']));raw[0]^=1;row['row_pixels_base64']=base64.b64encode(raw).decode()
    with pytest.raises(ValueError):r.verify_header_source(c['confirmed'],c['admitted'],p,require_physical=c['require_physical'])

def test_native_receipts_preserve_balances_without_occupancy_epoch(native):
    proof=native['proof_summary'];before=copy.deepcopy(proof);rows=carry.join_receipts(proof)
    assert len(rows)==8 and {r['actor']:r['calculated_start_stack'] for r in rows}==proof['start_stacks']
    assert next(r for r in rows if r['actor']=='iplayupay')['calculated_start_stack']==9234
    assert all(r['rule']==carry.RULE and 'occupancy_epoch' not in r for r in rows)
    assert proof==before

@pytest.mark.parametrize('mutation',['none','current_source','proof_removed','verification_rejected'])
def test_consumer_replays_current_sources_before_authority(native,monkeypatch,mutation):
    proof=native['proof_summary'];h=dict(table_id='T05',session_id=proof['current_hand_key'][1],hand_id=proof['current_hand_key'][2],messages=[])
    provider=object.__new__(carry.AdjacentStackCarryProvider);case=dict(current=dict(observed_hand=copy.deepcopy(h)))
    provider.proofs={carry.key(h):proof};provider.cases={carry.key(h):case};provider.stack_provider=object();provider.registry=object();calls=[]
    def verify(p,c,s,r):calls.append((p,c,s,r));return mutation!='verification_rejected'
    monkeypatch.setattr(carry,'verify_current',verify)
    if mutation=='current_source':h['messages'].append(dict(raw_text='different current native source'))
    elif mutation=='proof_removed':provider.proofs.clear()
    result=provider.for_hand(h)
    if mutation=='none':assert result==[proof] and len(calls)==1
    elif mutation=='verification_rejected':assert result==[] and len(calls)==1
    else:assert result==[] and calls==[]
