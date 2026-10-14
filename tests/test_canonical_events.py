import copy
import hashlib
import json
from pathlib import Path
import pytest
from parser_core.canonical_events import join_forced_posts, reconcile_stack_facts
from parser_core.poker_pipeline import analyze_timeline
from parser_core.seat_forced_binding import parse_forced_snapshot
from parser_core.canonical_events import prove_candidate_forced_prefix, FORCED_PREFIX_CAPABILITY


@pytest.fixture
def candidate_prefix_source():
    return json.loads((Path(__file__).parent/'fixtures/candidate_forced_prefix_t08_source.json').read_text('utf8'))


def candidate_arguments(source):
    class ExistingWho:
        def for_actor(self,hand,actor):
            return copy.deepcopy(next((p for p in source['who'] if p['actor']==actor),{'status':'UNKNOWN'}))
    return dict(rows=source['rows'],physical=source['physical'],hand=source['record']['observed_hand'],
                actor_provider=ExistingWho(),nick_rows=source['nick_rows'],headers=source['headers'])


def candidate_timeline(source):
    return [copy.deepcopy(r) for m in source['record']['observed_hand']['messages'] for r in m['source_rows']]


def test_complete_candidate_has_only_observed_forced_prefix_capability(candidate_prefix_source):
    s=candidate_prefix_source;before=copy.deepcopy(s)
    proof=prove_candidate_forced_prefix(**candidate_arguments(s))
    assert proof and proof['capability']==FORCED_PREFIX_CAPABILITY
    assert len(proof['posts'])==8 and sum(p['role']=='ANTE' for p in proof['posts'])==6
    assert {(p['actor'],p['amount']) for p in proof['posts'] if p['role']=='SB'}=={('dended',2500)}
    assert {(p['actor'],p['amount']) for p in proof['posts'] if p['role']=='BB'}=={('maximus058',5000)}
    assert all(p['epoch'] is None for p in proof['posts'])
    assert not proof['occupancy_authority'] and not proof['deal_complete_authority']
    assert not proof['start_stack_timing_authority']
    assert 'forced_snapshot_complete' not in proof and 'standard_ante_proven' not in proof
    assert proof['identities']=={p['actor']:p for p in s['who']} and s==before


def test_absent_ordinary_prefix_and_wrapped_bb_project_once(candidate_prefix_source):
    s=candidate_prefix_source;timeline=candidate_timeline(s);before=copy.deepcopy(timeline)
    joined,report=join_forced_posts(timeline,{},candidate_inputs=[candidate_arguments(s)])
    assert report['counts']=={'JOINED':8}
    new=[r for r in joined if r.get('forced_prefix_capability')==FORCED_PREFIX_CAPABILITY]
    assert len(new)==8 and len({r['canonical_event_id'] for r in new})==8
    bb=next(r for r in new if r['event_type']=='BIG_BLIND')
    assert bb['amount']==5000 and len(bb['canonical_provenance']['post']['source_refs'])==2
    assert bb['epoch'] is None and not bb['historical_occupancy_authority']
    assert timeline==before


def test_candidate_partial_overlap_uses_existing_economic_dedup(candidate_prefix_source):
    s=candidate_prefix_source;timeline=candidate_timeline(s)
    full,_=join_forced_posts(timeline,{},candidate_inputs=[candidate_arguments(s)])
    ante=copy.deepcopy(next(r for r in full if r.get('event_type')=='ANTE'))
    joined,report=join_forced_posts(timeline+[ante],{},candidate_inputs=[candidate_arguments(s)])
    unique={r['canonical_event_id'] for r in joined if r.get('canonical_event_id') and not r.get('canonical_duplicate_of')}
    assert len(unique)==8 and report['counts']=={'DEDUPLICATED':1,'JOINED':7}


def test_complete_ordinary_prefix_is_byte_preserved_on_candidate_repeat(candidate_prefix_source):
    s=candidate_prefix_source
    full,_=join_forced_posts(candidate_timeline(s),{},candidate_inputs=[candidate_arguments(s)])
    again,report=join_forced_posts(full,{},candidate_inputs=[candidate_arguments(s)])
    assert again==full and report['counts']=={'CORROBORATED':1}


@pytest.mark.parametrize('mutation',[
    'foreign_hand','foreign_session','foreign_table','cross_epoch','unknown_epoch','missing_marker',
    'foreign_marker','physical_frame','physical_clock','source_gap','source_order','source_pixel',
    'source_geometry','no_deal','low_confidence','missing_who','foreign_who_hand','foreign_nick_hand',
    'nick_pixel','seat_collision','explicit_replacement','explicit_departure','no_sb',
])
def test_candidate_authority_rejects_incomplete_or_conflicting_source(candidate_prefix_source,mutation):
    s=candidate_prefix_source
    if mutation=='foreign_hand':s['rows'][8]['hand_id']='foreign'
    elif mutation=='foreign_session':s['rows'][8]['session_id']='foreign'
    elif mutation=='foreign_table':s['rows'][8]['table_id']=1
    elif mutation=='cross_epoch':s['rows'][8]['epoch']=4
    elif mutation=='unknown_epoch':
        for r in s['rows']:r['epoch']=4
    elif mutation=='missing_marker':s['headers']=[]
    elif mutation=='foreign_marker':
        for r in s['rows']:r['marker_source_frame']=0
    elif mutation=='physical_frame':s['physical']['frame_after']+=1
    elif mutation=='physical_clock':s['physical']['source_pts']+=1
    elif mutation=='source_gap':s['rows'].pop(9)
    elif mutation=='source_order':s['rows'][9]['index']=8
    elif mutation=='source_pixel':s['rows'][9]['pixel_sha256']='0'*64
    elif mutation=='source_geometry':s['rows'][9]['row_bounds'][0]+=1
    elif mutation=='no_deal':s['rows'][-1]['ocr_response']['raw_text']='Dealer: Unknown message'
    elif mutation=='low_confidence':s['rows'][14]['ocr_response']['score']=.9
    elif mutation=='missing_who':s['who'][0]['status']='UNKNOWN'
    elif mutation=='foreign_who_hand':s['who'][0]['hand_id']='foreign'
    elif mutation=='foreign_nick_hand':s['nick_rows'][0]['hand_id']='foreign'
    elif mutation=='nick_pixel':s['nick_rows'][0]['pixel_sha256']='0'*64
    elif mutation=='seat_collision':s['who'][0]['seat']=s['who'][1]['seat']
    elif mutation=='explicit_replacement':s['who'][0]['boundaries'].append({'kind':'REPLACE','frame':1700})
    elif mutation=='explicit_departure':
        row=next(r for m in s['record']['observed_hand']['messages'] for r in m['source_rows'] if r['frame_id']>=1673)
        row['frame_id']=1700;row['raw_ocr_text']='Dealer: dended has left the table'
    elif mutation=='no_sb':s['rows'][14]['ocr_response']['raw_text']='Dealer: Waiting for players'
    assert prove_candidate_forced_prefix(**candidate_arguments(s)) is None
    joined,report=join_forced_posts(candidate_timeline(s),{},candidate_inputs=[candidate_arguments(s)])
    assert not any(r.get('forced_prefix_capability') for r in joined)
    assert report['counts']=={'UNRESOLVED':1}


@pytest.mark.parametrize('mutation',['sb_actor','sb_amount','bb_amount','extra_ante'])
def test_candidate_conflict_rejects_entire_prefix_atomically(candidate_prefix_source,mutation):
    s=candidate_prefix_source;timeline=candidate_timeline(s)
    full,_=join_forced_posts(timeline,{},candidate_inputs=[candidate_arguments(s)])
    target='SMALL_BLIND' if mutation.startswith('sb') else 'BIG_BLIND'
    ordinary=copy.deepcopy(next(r for r in full if r.get('event_type')==target))
    text={'sb_actor':'Dealer: other posts Small Blind 2 500',
          'sb_amount':'Dealer: dended posts Small Blind 2 501',
          'bb_amount':'Dealer: maximus058 posts Big Blind 5 001',
          'extra_ante':'Dealer: other posts Ante 625'}[mutation]
    ordinary.update(raw_ocr_text=text,stable_row_id='ordinary-conflict')
    for field in ('canonical_event_id','canonical_provenance','forced_prefix_capability'):ordinary.pop(field,None)
    joined,report=join_forced_posts(timeline+[ordinary],{},candidate_inputs=[candidate_arguments(s)])
    assert report['counts']=={'CONFLICT':1}
    assert not any(r.get('forced_prefix_capability') for r in joined)


def test_retained_raw_forced_conflict_cannot_hide_as_unknown(candidate_prefix_source):
    s=candidate_prefix_source;timeline=candidate_timeline(s)
    full,_=join_forced_posts(timeline,{},candidate_inputs=[candidate_arguments(s)])
    ordinary=copy.deepcopy(next(r for r in full if r.get('event_type')=='SMALL_BLIND'))
    ordinary.update(stable_row_id='retained-conflict',raw_ocr_text='Dealer: other posts Small Blind 2500',
                    canonical_conflict='FORCED_SOURCES_DISAGREE_OR_IDENTITY_AMBIGUOUS')
    for field in ('canonical_event_id','canonical_provenance','forced_prefix_capability'):ordinary.pop(field,None)
    joined,report=join_forced_posts(timeline+[ordinary],{},candidate_inputs=[candidate_arguments(s)])
    assert report['counts']=={'CONFLICT':1} and not any(r.get('forced_prefix_capability') for r in joined)
    assert next(r for r in joined if r['stable_row_id']=='retained-conflict')['canonical_conflict']


def test_legacy_accepted_summary_is_not_suppressed_by_candidate_overlap(candidate_prefix_source):
    s=candidate_prefix_source;args=candidate_arguments(s)
    proof=prove_candidate_forced_prefix(**args);posts=copy.deepcopy(proof['posts'])
    for p in posts:p['epoch']='legacy-existing-epoch'
    p=posts[0]
    summary=dict(forced_snapshots=[dict(table=p['table'],session=p['session'],hand_id=p['hand_id'],
        forced_snapshot_complete=True,posts=posts)],bound_posts=[p],historical_binding_receipts=[
        dict(table=p['table'],session=p['session'],hand_id=p['hand_id'],actor=p['actor'],seat=p['seat'],
             epoch=p['epoch'],binding_rule='FIRST_EXACT_LABEL_HISTORICAL_EPOCH',
             forced_source_frame=p['source_frame'],forced_source_refs=[p['source_refs']],label_evidence=[])])
    joined,report=join_forced_posts(candidate_timeline(s),summary,candidate_inputs=[args])
    assert not any(r.get('forced_prefix_capability') for r in joined)
    assert sum(r.get('canonical_event_id') is not None for r in joined)==1
    assert any(d.get('blocker')=='EXISTING_SUMMARY_FORCED_AUTHORITY_RETAINED' for d in report['diagnostics'])


def test_repeated_candidate_batch_creates_one_prefix(candidate_prefix_source):
    s=candidate_prefix_source;args=candidate_arguments(s)
    joined,report=join_forced_posts(candidate_timeline(s),{},candidate_inputs=[args,args])
    assert sum(r.get('forced_prefix_capability')==FORCED_PREFIX_CAPABILITY for r in joined)==8
    assert report['counts']=={'JOINED':8,'CORROBORATED':1}


@pytest.mark.parametrize('order',[(0,1,0),(1,0,0)])
def test_candidate_source_conflict_cannot_be_erased_by_later_repeat(candidate_prefix_source,order):
    a=candidate_prefix_source;b=copy.deepcopy(a)
    b['rows'][16]['ocr_response']['raw_text']='5001'
    sources=[candidate_arguments(a),candidate_arguments(b)]
    joined,report=join_forced_posts(candidate_timeline(a),{},candidate_inputs=[sources[i] for i in order])
    assert not any(r.get('forced_prefix_capability') for r in joined)
    assert report['counts']=={'CONFLICT':2}


def test_deferred_admission_updates_existing_records_before_simple_corrections(candidate_prefix_source,tmp_path,monkeypatch):
    from parser_core.simple_saved_policy import SimpleSavedPolicyProvider
    import parser_core.canonical_events as canonical
    s=candidate_prefix_source;record=copy.deepcopy(s['record']);hand=record['observed_hand']
    key=(8,hand['session_id'],hand['hand_id'])
    (tmp_path/'seat_forced_inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in s['rows']+s['nick_rows']),'utf8')
    (tmp_path/'physical_chat_admission.jsonl').write_text(json.dumps(s['physical'])+'\n','utf8')
    monkeypatch.setattr(canonical,'source_bound_headers',lambda source:(s['headers'],{},[]))
    frames=[r['frame_id'] for m in hand['messages'] for r in m['source_rows']]
    provider=SimpleSavedPolicyProvider([hand],{key:(min(frames),max(frames))},{},
        [p['anchor'] for p in s['who']],candidate_source=tmp_path,candidate_records=[record])
    assert record['forced_prefix_admission']['capability']==FORCED_PREFIX_CAPABILITY
    assert record['observed_hand'] is provider.hands[key]
    rows=provider.seat_rows(record['observed_hand'])
    dended=next(r for r in rows if r['seat']=='seat4')
    assert dended['forced']=={'ANTE':625,'SB':2500,'BB':0}
    assert dended['START_STACK'] is None  # No numeric or timing authority is fabricated.
    assert len(provider.forced_prefix_admission)==1
    old=s['record']
    for name in ('logical_hand_analysis.jsonl','complete_hand_analysis.jsonl'):
        (tmp_path/name).write_text(json.dumps(old)+'\n','utf8')
    retained=dict(canonical_event_id='retained-other-hand',raw_ocr_text='unchanged')
    (tmp_path/'canonical_forced_events.jsonl').write_text(json.dumps(retained)+'\n','utf8')
    (tmp_path/'canonical_join_summary.json').write_text(json.dumps({'counts':{'JOINED':1}}),'utf8')
    provider.write_audit(tmp_path)
    for name in ('logical_hand_analysis.jsonl','complete_hand_analysis.jsonl'):
        assert json.loads((tmp_path/name).read_text('utf8'))==record
    events=[json.loads(line) for line in (tmp_path/'canonical_forced_events.jsonl').read_text('utf8').splitlines()]
    assert events[0]==retained and len(events)==9
    assert len({r['canonical_event_id'] for r in events})==9
    provider.write_audit(tmp_path)
    assert events==[json.loads(line) for line in (tmp_path/'canonical_forced_events.jsonl').read_text('utf8').splitlines()]
    summary=json.loads((tmp_path/'canonical_join_summary.json').read_text('utf8'))
    assert summary['counts']=={'JOINED':1}
    assert summary['candidate_forced_prefix_admission']==provider.forced_prefix_admission


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
