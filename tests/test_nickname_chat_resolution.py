"""Isolated candidate safeguards; synthetic cases test rules, not accuracy."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('nickname_resolution_core',ROOT/'experiments/nickname_chat_resolution/core.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)


def read(text, conf=.90, i=0, table=1, seat='seat1'):
    return dict(read_id=f'N{i}',ocr_raw=text,confidence=conf,frame=15*i,time=.5*i,
                crop_hash=f'hash{i}',table=table,seat=seat,**core.classify(text,conf))


@pytest.mark.parametrize('text',['CHECK','CALL','BET','RAISE','FOLD','ANTE','WINNER','OPEN SEAT','PENSEAT','SITOUT','123','1 234','','CALL 120'])
def test_ui_masks(text):
    assert core.classify(text,.40)['category']=='NON_NICKNAME_UI'


@pytest.mark.parametrize('text',['PR','CNL','CAL','SR','CHECV','ANT','RR','ATE','SP','PEN SEAT','sHow','SHOw','sHOw','suow','TIME','WINNGR','MUCK','MuCK'])
def test_exact_observed_ui_variants(text):
    result=core.classify(text,.99)
    assert result['category']=='NON_NICKNAME_UI'
    assert result['ui_mask_reason']
    assert result['normalized_text']==text and result['displayed_nickname']==text


@pytest.mark.parametrize('text',['PRplayer','CNLnick','CALvin','SRname','CHECVfoo','ANTon','RRrr','ATEplayer','SPartan','sHowman','suow_name','TIMEkeeper','WINNGR123','MUCKhero'])
def test_ui_fragments_inside_real_nickname_are_retained(text):
    assert core.classify(text,.20)['category']=='NICKNAME_LOW_CONF'
    assert core.classify(text,.20)['ui_mask_reason'] is None


@pytest.mark.parametrize('conf,band',[(.95,'HIGH'),(.9499,'MEDIUM'),(.85,'MEDIUM'),(.8499,'LOW'),(.01,'LOW'),(None,'UNKNOWN')])
def test_confidence_is_category_not_gate(conf,band):
    r=read('real_nick',conf)
    assert r['confidence_band']==band
    assert r['normalized_text']=='real_nick'


def test_empty_chat_cannot_invent_identity():
    assert core.resolve([],{'player':[]})['resolved_nickname']=='UNKNOWN'


def test_low_exact_retained():
    result=core.resolve([read('player',.2,i) for i in range(3)],{'player':[]})
    assert result['identity_status']=='EXACT'


def test_unique_literal_prefix():
    rs=[read('fantomas...',.8,i) for i in range(3)]
    assert core.resolve(rs,{'fantomas1987':[]})['identity_status']=='PREFIX_MATCH'
    assert core.resolve(rs,{'fantomas1987':[],'fantomas2000':[]})['identity_status']=='AMBIGUOUS'
    assert core.resolve(rs,{'fantomas':[]})['resolved_nickname']=='UNKNOWN'


def test_truncated_fuzzy_prefix_forbidden():
    result=core.resolve([read('fantomasl...',.9,i) for i in range(3)],{'fantomas1987':[]})
    assert result['resolved_nickname']=='UNKNOWN'


def test_truncated_chat_is_not_full_nickname():
    rs=[read('fantomas...',.9,i) for i in range(3)]
    assert core.resolve(rs,{'fantomas...':[]})['full_nickname']=='UNKNOWN'


def test_independent_chat_ocr_conflict_withheld():
    rs=[read('Oxyei',.9,i) for i in range(3)]
    assert core.resolve(rs,{'Oxyei':[dict(raw_ocr_verified=False)]})['identity_status']=='AMBIGUOUS'


def test_ocr_o_zero_collision_not_merged():
    rs=[read('Oxyei',.9,i) for i in range(3)]
    result=core.resolve(rs,{'Oxyei':[],'0xyei':[]})
    assert result['resolved_nickname']=='Oxyei'
    mixed=[read('Oxyei',.9,0),read('0xyei',.9,1),read('Oxyei',.9,2)]
    assert core.resolve(mixed,{'Oxyei':[],'0xyei':[]})['identity_status']=='AMBIGUOUS'


def test_fuzzy_requires_three_and_margin_and_supported():
    rs=[read('yujjanin',.9,i) for i in range(3)]
    result=core.resolve(rs,{'yujanin':[],'greyscorpi':[]})
    assert result['identity_status']=='MULTIREAD_MATCH'
    assert result['identity_confidence']=='SUPPORTED'
    assert core.resolve(rs[:2],{'yujanin':[]})['resolved_nickname']=='UNKNOWN'
    assert core.resolve(rs,{'yujanin':[],'yujjanii':[]})['identity_status']=='AMBIGUOUS'


def test_conflicting_group_abstains():
    rs=[read('hazard',.9,0),read('hazard',.9,1),read('another',.9,2)]
    assert core.resolve(rs,{'hazard':[]})['identity_status']=='AMBIGUOUS'


def test_duplicate_pixel_frame_and_cross_seat():
    rs=[read('player',.9,i) for i in range(3)]
    rs[2]['crop_hash']=rs[0]['crop_hash']
    assert core.resolve(rs,{'player':[]})['resolved_nickname']=='UNKNOWN'
    rs=[read('player',.9,i) for i in range(3)];rs[2]['seat']='seat2'
    assert core.resolve(rs,{'player':[]})['resolved_nickname']=='UNKNOWN'


def test_window_same_table_only_boundary():
    rs=[read('player')]
    actors=[dict(nickname='player',table=1,time=15),dict(nickname='cross_table',table=2,time=0),dict(nickname='outside',table=1,time=15.0001)]
    assert list(core.window_candidates(rs,actors,1))==['player']


def test_spacing_and_support_span():
    rs=[read('player',i=i) for i in range(5)];rs[1]['frame']=1;rs[4]['frame']=400
    selected,_=core.independent_reads(rs)
    assert [r['frame'] for r in selected]==[0,30,45]


def test_off_opens_no_input_creates_no_output(tmp_path):
    out=tmp_path/'new'
    result=subprocess.run([sys.executable,str(ROOT/'experiments/nickname_chat_resolution/run.py'),'--nickname-source','missing','--chat-source','missing','--output',str(out)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '"ocr_calls": 0' in result.stdout and '"outputs_created": 0' in result.stdout
    assert not out.exists()
