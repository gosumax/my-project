import copy
import hashlib
import pytest
from parser_core.canonical_events import join_forced_posts, reconcile_stack_facts
from parser_core.poker_pipeline import analyze_timeline
from parser_core.seat_forced_binding import parse_forced_snapshot


def all_in_fixture(ordinary=True):
    texts=['Hand #2878365444','Dealer: a posts Ante 37','and is All-in',
           'Dealer: b posts Ante 50','Dealer: c posts Ante 50',
           'Dealer: b posts Small Blind 100','Dealer: c posts Big Blind 200','Dealer: Dealing cards']
    source=[]
    for i,text in enumerate(texts):
        digest=hashlib.sha256(text.encode()).hexdigest()
        source.append(dict(table_id=1,session_id='s',hand_id='2878365444',source_frame=10,
            source_pts=10,time_base=[1,30],index=i,source_ref=digest,pixel_sha256=digest,text=text,
            score=.99,ocr_response=dict(status='OK',raw_text=text,score=.99)))
    snapshot,block=parse_forced_snapshot(source,'2878365444',3)
    assert block is None and snapshot['standard_ante'] is None
    posts=[];receipts=[]
    for actor,seat in [('a','seat1'),('b','seat2'),('c','seat3')]:
        refs=[]
        for post in snapshot['posts']:
            post.update(source_frame=10,source_pts=10,time_base=[1,30])
            if post['actor']==actor:
                posts.append(dict(post,table=1,session='s',hand_id='2878365444',seat=seat,epoch='e'))
                refs.append(post['source_refs'])
        receipts.append(dict(table=1,session='s',hand_id='2878365444',actor=actor,seat=seat,epoch='e',
            binding_rule='FIRST_EXACT_LABEL_HISTORICAL_EPOCH',forced_source_frame=10,forced_source_refs=refs))
    summary=dict(bound_posts=posts,historical_binding_receipts=receipts,
                 forced_snapshots=[dict(snapshot,table=1,session='s',hand_id='2878365444')])
    rr=[dict(table_id='T01',session_id='s',frame_id=9,row_index=0,stable_row_id='header',
             raw_ocr_text='Hand #2878365444',video_time=.3,epoch=1,status='RAW_UNREVIEWED')]
    if ordinary:rr.append(dict(rr[0],frame_id=10,row_index=1,stable_row_id='ordinary-ante',
                               raw_ocr_text='Dealer: a posts Ante 37'))
    return rr,summary


@pytest.mark.parametrize('ordinary',[False,True])
def test_source_bound_all_in_survives_insert_or_plain_ordinary_dedup_in_all_consumers(ordinary):
    from parser_core.poker_hand_assembly import make_message
    from parser_core.poker_reconstruction.timeline import parse_rows
    from parser_core.poker_reconstruction.assembly import candidate_assemblies
    rr,ss=all_in_fixture(ordinary);frozen=copy.deepcopy(rr)
    result,report=join_forced_posts(rr,ss)
    row=next(r for r in result if r.get('player')=='a')
    original=row['raw_ocr_text'];message=make_message([row])
    assert message['all_in'] and message['field_sources']['all_in']=='OBSERVED_SOURCE_BOUND'
    assert parse_rows([row])[0].all_in
    choices,_,_=candidate_assemblies([dict(row,source_index=1)])
    assert choices and all(path[0].all_in for path in choices)
    records,_=analyze_timeline(result,ocr_status={r['stable_row_id']:'OK' for r in result})
    assert records[0]['validation']['players']['a']['all_in'] is True
    assert rr==frozen and row['raw_ocr_text']==original
    if ordinary:
        assert original=='Dealer: a posts Ante 37' and report['counts']['DEDUPLICATED']==1


@pytest.mark.parametrize('change',['hand','session','epoch','seat','actor','role','amount','source','receipt','event','conflict','duplicate'])
def test_source_bound_all_in_invalid_proof_is_withheld_in_validator_and_reconstruction(change):
    from parser_core.poker_hand_assembly import make_message
    from parser_core.poker_reconstruction.timeline import parse_rows
    from parser_core.poker_reconstruction.assembly import candidate_assemblies
    rr,ss=all_in_fixture();result,_=join_forced_posts(rr,ss)
    row=copy.deepcopy(next(r for r in result if r.get('player')=='a'))
    p=row['canonical_provenance']['post']
    if change=='hand':row['hand_id']='2878365445'
    if change=='session':row['session_id']='OTHER'
    if change=='epoch':row['canonical_provenance']['binding_receipt']['epoch']='OTHER'
    if change=='seat':row['seat']='seat8'
    if change=='actor':p['actor']='OTHER'
    if change=='role':p['role']='BB'
    if change=='amount':p['amount']=38
    if change=='source':p['source_rows'][1]['text']='unknown text'
    if change=='receipt':row['binding_status']='UNKNOWN'
    if change=='event':row.pop('canonical_event_id')
    if change=='conflict':row['canonical_conflict']='CONFLICT'
    if change=='duplicate':row['canonical_duplicate_of']='OTHER'
    assert make_message([row])['action']=='UNKNOWN_EVENT'
    assert parse_rows([row])[0].kind=='UNKNOWN'
    choices,_,_=candidate_assemblies([dict(row,source_index=1)])
    assert all(path[0].kind=='UNKNOWN' for path in choices)


def fixture(include_start=False):
    texts=['Hand #2878365444','Dealer: a posts Ante 50','Dealer: b posts Ante 50',
           'Dealer: a posts Small','Blind 100','Dealer: b posts Big Blind 200','Dealer: Dealing cards']
    if include_start:texts[1:1]=['Dealer: Starting a new hand','#2878365444']
    rows=[dict(table_id=1,session_id='s',hand_id='2878365444',source_frame=10,source_pts=10,time_base=[1,30],index=i,
        source_ref=str(i)*64,pixel_sha256=str(i)*64,text=t,score=.99,ocr_response={'status':'OK','raw_text':t,'score':.99}) for i,t in enumerate(texts)]
    snapshot,_=parse_forced_snapshot(rows,'2878365444',2)
    posts=[];receipts=[]
    for actor,seat in [('a','seat1'),('b','seat2')]:
        refs=[]
        for p in snapshot['posts']:
            p.update(source_frame=10,source_pts=10,time_base=[1,30])
            if p['actor']==actor:
                posts.append(dict(p,table=1,session='s',hand_id='2878365444',seat=seat,epoch='e'))
                refs.append(p['source_refs'])
        receipts.append(dict(table=1,session='s',hand_id='2878365444',actor=actor,seat=seat,epoch='e',
            binding_rule='FIRST_EXACT_LABEL_HISTORICAL_EPOCH',forced_source_frame=10,forced_source_refs=refs))
    summary=dict(bound_posts=posts,historical_binding_receipts=receipts,
                 forced_snapshots=[dict(snapshot,table=1,session='s',hand_id='2878365444')])
    timeline=[dict(table_id='T01',session_id='s',frame_id=f,row_index=i,stable_row_id=f'r{i}',
                   raw_ocr_text=t,video_time=f/30,epoch=1,status='RAW_UNREVIEWED')
              for i,(f,t) in enumerate([(9,'Hand #2878365444'),(11,'Dealer: a calls 100'),(12,'Dealer: b checks')])]
    return timeline,summary


def test_missing_ordinary_forced_ante_sb_bb_and_provenance_order():
    rr,ss=fixture();frozen=copy.deepcopy(rr)
    result,report=join_forced_posts(rr,ss)
    assert report['counts']=={'JOINED':4}
    assert [r.get('action_type') for r in result[1:5]]==['ANTE','ANTE','SMALL_BLIND','BIG_BLIND']
    assert rr==frozen
    sb=result[3]
    assert sb['amount']==100 and len(sb['canonical_provenance']['post']['source_rows'])==2
    assert sb['frame_id']==10 and sb['action_onset_timestamp'] is None
    assert [r['raw_ocr_text'] for r in result if r['stable_row_id'].startswith('r')]==[r['raw_ocr_text'] for r in rr]
    records,coverage=analyze_timeline(result,ocr_status={r['stable_row_id']:'OK' for r in rr})
    assert coverage['source_rows_covered']==7
    msgs=records[0]['observed_hand']['messages']
    assert [m['action'] for m in msgs[1:5]]==['ANTE','ANTE','SMALL_BLIND','BIG_BLIND']
    assert msgs[3]['canonical_provenance']==sb['canonical_provenance']


def test_ordinary_duplicate_exact_one_money_event():
    rr,ss=fixture()
    r=dict(rr[0],stable_row_id='ante',row_index=1,frame_id=10,raw_ocr_text='Dealer: a posts Ante 50')
    rr.append(r)
    result,report=join_forced_posts(rr,ss)
    assert report['counts']=={'DEDUPLICATED':1,'JOINED':3}
    assert len({r['canonical_event_id'] for r in result if r.get('canonical_event_id')})==4


def test_ordinary_conflict_retained_but_no_money_authority():
    rr,ss=fixture()
    rr.append(dict(rr[0],stable_row_id='ante',row_index=1,frame_id=10,raw_ocr_text='Dealer: a posts Ante 60'))
    result,report=join_forced_posts(rr,ss)
    assert report['counts']['CONFLICT']==1
    row=next(r for r in result if r['stable_row_id']=='ante')
    assert row['raw_ocr_text']=='Dealer: a posts Ante 60' and row['canonical_conflict']
    records,_=analyze_timeline(result,ocr_status={r['stable_row_id']:'OK' for r in rr})
    m=next(m for m in records[0]['observed_hand']['messages'] if 'ante' in m['source_row_ids'])
    assert m['amount'] is None and m['action']=='UNKNOWN_EVENT'
    from parser_core.poker_reconstruction.timeline import parse_rows
    assert parse_rows([row])[0].kind=='UNKNOWN'


def test_missing_actor_or_hand_binding_is_unresolved():
    rr,ss=fixture();ss['historical_binding_receipts']=[]
    result,report=join_forced_posts(rr,ss)
    assert len(result)==len(rr) and report['counts']['UNRESOLVED']==4
    rr,ss=fixture();rr=rr[1:]
    result,report=join_forced_posts(rr,ss)
    assert len(result)==len(rr) and report['counts']['UNRESOLVED']==4


def test_unbound_snapshot_posts_remain_diagnostics():
    rr,ss=fixture();ss['bound_posts']=ss['bound_posts'][:2]
    result,report=join_forced_posts(rr,ss)
    assert report['counts']=={'JOINED':2,'UNRESOLVED':2}


def test_start_stack_same_event_truth_and_conflict_fail_closed():
    rr,ss=fixture();events,_=join_forced_posts(rr,ss)
    events=[r for r in events if r.get('canonical_event_id')]
    facts=dict(table=1,session='s',hand_id='2878365444',seat='seat1',occupancy_epoch='e',
               forced_bets={'ANTE':50,'SB':100},status='PROVEN_FOR_MATH')
    reconciled=reconcile_stack_facts(facts,events)
    assert reconciled['forced_truth']=='CANONICAL_EVENTS' and len(reconciled['canonical_forced_event_ids'])==2
    assert reconciled['vision_fact']==facts
    bad=reconcile_stack_facts(facts,[r for r in events if r['action_type']!='SMALL_BLIND'])
    assert bad['status']=='UNKNOWN'


def test_local_actor_correction_requires_all_physical_posts_and_preserves_vision():
    rr,ss=fixture();events,_=join_forced_posts(rr,ss)
    events=[r for r in events if r.get('canonical_event_id')]
    fact=dict(table=1,session='s',hand_id='2878365444',seat='seat1',occupancy_epoch='e',
              status='UNKNOWN',blocker='SPECIAL_FORCED_SEAT_UNKNOWN',participant_proven=True,
              displayed_stack_status='VALID_FIXED_SEAT_ROI',deal_complete_frame=11)
    frozen=copy.deepcopy(fact)
    fixed=reconcile_stack_facts(fact,events,forced_snapshots=ss['forced_snapshots'])
    assert fixed['status']=='PROVEN_FOR_MATH' and fixed['forced_bets']=={'ANTE':50,'SB':100}
    assert fixed['vision_fact']==frozen and fact==frozen
    for bad in [[r for r in events if r['action_type']!='SMALL_BLIND'],
                [dict(r,seat='seat3') for r in events],
                [dict(r,canonical_provenance=dict(r['canonical_provenance'],post=dict(r['canonical_provenance']['post'],epoch='wrong'))) for r in events]]:
        assert reconcile_stack_facts(fact,bad,forced_snapshots=ss['forced_snapshots'])['status']=='UNKNOWN'
    assert reconcile_stack_facts(fact,events,forced_snapshots=[])['status']=='UNKNOWN'


def test_adjacent_hand_source_fence_never_borrowed():
    rr,ss=fixture()
    rr.append(dict(rr[0],frame_id=10,row_index=0,stable_row_id='new',raw_ocr_text='Hand #2878365445'))
    result,report=join_forced_posts(rr,ss)
    assert report['counts']['UNRESOLVED']==4 and not any(r.get('canonical_event_id') for r in result)


def test_contradictory_blind_role_withholds_ordinary_money():
    rr,ss=fixture()
    rr.append(dict(rr[0],frame_id=10,row_index=1,stable_row_id='wrong_bb',raw_ocr_text='Dealer: a posts Big Blind 100'))
    result,report=join_forced_posts(rr,ss)
    assert report['counts']['CONFLICT']==2
    assert next(r for r in result if r['stable_row_id']=='wrong_bb')['canonical_conflict']
    assert not any(r.get('action_type') in {'SMALL_BLIND','BIG_BLIND'} for r in result)


def test_dirty_intervening_prefix_never_proven():
    rr,ss=fixture()
    sources=[dict(index=0,text='Hand #2878365444',score=.99,source_frame=10,source_ref='header')]
    sources += [dict(index=1,text='Dealer: a calls 100',score=.99,source_frame=10,source_ref='action')]
    sources += [dict(r,index=r['index']+1) for p in ss['forced_snapshots'][0]['posts'] for r in p['source_rows']]
    sources += [dict(index=8,text='Dealer: Dealing cards',score=.99,source_frame=10,source_ref='deal')]
    snapshot,blocker=parse_forced_snapshot(sources,'2878365444',2)
    assert snapshot is None and blocker=='VOLUNTARY_OR_TERMINAL_IN_FORCED_PREFIX'

def test_sidecar_header_has_complete_timeline_export_contract():
    rr,ss=fixture();header=dict(rr[0],source_kind='SIDECAR_HEADER_RECEIPT',ocr_status='OK',
        row_pixel_sha256='a'*64,header_provenance={'proof':'fixture'})
    result,report=join_forced_posts(rr[1:],ss,[header])
    assert all('event_type' in r for r in result if r.get('canonical_stream_version'))
    assert result[0]['event_type']=='HAND_HEADER'


def test_corroborated_physical_prefix_preserves_wrapped_start_and_post_order():
    rr,ss=fixture(include_start=True)
    texts=['Hand #2878365444','Dealer: Starting a new hand','#2878365444',
        'Dealer: a posts Ante 50','Dealer: b posts Ante 50','Dealer: a posts Small',
        'Blind 100','Dealer: b posts Big Blind 200','Dealer: Dealing cards']
    rr=[dict(rr[0],frame_id=10,row_index=i,stable_row_id=f'physical:{i}',raw_ocr_text=text)
        for i,text in enumerate(texts)]
    frozen=copy.deepcopy(rr)
    result,report=join_forced_posts(rr,ss)
    assert [r['stable_row_id'] for r in result]==[r['stable_row_id'] for r in rr]
    assert report['counts']=={'DEDUPLICATED':4}
    records,_=analyze_timeline(result,ocr_status={r['stable_row_id']:'OK' for r in result})
    assert [m['action'] for m in records[0]['observed_hand']['messages']]==[
        'HAND_HEADER','NEW_HAND','ANTE','ANTE','SMALL_BLIND','BIG_BLIND','DEAL']
    assert rr==frozen
