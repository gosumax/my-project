"""Read complete bounded source intervals; never select only positive labels."""
import hashlib
import json
import base64
import sqlite3
from collections import Counter,defaultdict
from pathlib import Path
from .stack_timeline_storage import stack_timeline_source_name

FILES=('ordered_timeline.jsonl','physical_rows.jsonl','raw_ocr_results.jsonl',
       'physical_chat_admission.jsonl','hand_header_ocr.jsonl','hand_state_receipts.jsonl',
       'seat_forced_inputs.jsonl','start_stack_capture.jsonl')


def load_contexts(source,windows,*,actor_policy=False):
    source=Path(source)
    contexts=[dict(w,ordinary_prefix_timeline=[],ordinary_prefix_physical_rows=[],
        ordinary_prefix_raw_ocr=[],header_receipts_admitted_and_confirmed=[],hand_state_receipts=[],
        nickname_observations_all_within_window=[],header_interval_records=[],state_interval_receipts=[],
        ordinary_hand_markers_in_capture_window=[],ordinary_rows_in_capture_window=[],native_capture_nick_rows=[])
        for w in windows]
    source_hashes={}
    for name in FILES:
        digest=hashlib.sha256()
        with (source/name).open('rb') as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():continue
                r=json.loads(line)
                rawtable=r.get('table_id',r.get('table'))
                table=int(str(rawtable).lstrip('T')) if rawtable is not None else None
                session=r.get('session_id',r.get('table_session_id'))
                frame=r.get('source_frame',r.get('frame_id',r.get('frame_after')))
                for c in contexts:
                    if (table,session)!=(c['table_id'],c['session_id']) or type(frame) is not int or not c['header_frame']<=frame<=c['capture_end_inclusive']:continue
                    at_start=frame==c['header_frame']
                    if name=='ordered_timeline.jsonl':
                        if r.get('source') in {'FORCED_SNAPSHOT','SIDECAR_HEADER'}:continue
                        c['ordinary_rows_in_capture_window'].append(r)
                        if at_start:c['ordinary_prefix_timeline'].append(r)
                        if r.get('event_type') in {'HAND_HEADER','HAND_ID','NEW_HAND'}:
                            c['ordinary_hand_markers_in_capture_window'].append(r)
                    elif name=='physical_rows.jsonl' and at_start:c['ordinary_prefix_physical_rows'].append(r)
                    elif name=='raw_ocr_results.jsonl' and at_start:c['ordinary_prefix_raw_ocr'].append(r)
                    elif name=='physical_chat_admission.jsonl' and at_start:
                        if 'ordinary_prefix_admission' in c:raise ValueError('MULTIPLE_ADMISSION_RECEIPTS')
                        c['ordinary_prefix_admission']=r
                    elif name=='hand_header_ocr.jsonl':
                        c['header_interval_records'].append(r)
                        if at_start:c['header_receipts_admitted_and_confirmed'].append(r)
                    elif name=='hand_state_receipts.jsonl':
                        c['state_interval_receipts'].append(r)
                        if at_start:c['hand_state_receipts'].append(r)
                    elif name=='seat_forced_inputs.jsonl' and r.get('kind')=='NICK':
                        c['nickname_observations_all_within_window'].append(r)
                    elif name=='start_stack_capture.jsonl' and r.get('field')=='nick':
                        c['native_capture_nick_rows'].append(r)
        source_hashes[name]=digest.hexdigest()
    for c in contexts:
        ids={r['stable_row_id'] for r in c['ordinary_prefix_timeline']}
        for key in ('ordinary_prefix_physical_rows','ordinary_prefix_raw_ocr'):
            c[key]=[r for r in c[key] if r['row_id'] in ids]
    loaded=dict(schema='positive-actor-proof-inputs-v2',scope='complete native bounded windows; no proof authority',
                source_hashes=source_hashes,contexts=contexts)
    if actor_policy:
        from .chat_nick_actor import attach_current_sources
        return attach_current_sources(source,loaded)
    return loaded


def verify_against_current_sources(proof,current_context):
    # The caller re-reads complete files; a proof's retained subset is never
    # used as the current source universe for authority.
    from parser_core.actor_seat_point import prove_actor_point
    return proof==prove_actor_point(current_context,proof.get('actor')) and proof.get('status')=='PROVEN'


class JointStackSourceProvider:
    """Read all available sources in each bounded hand window, then verify.

    Retained proofs are output evidence. They are never the loader's input.
    This route has no occupancy-epoch authority and never changes source rows.
    """
    def __init__(self,source,hands,*,video_sha256,capture_frames=64,sitout_provider=None,actor_policy=False,native_who=None):
        from .actor_seat_point import HEADER,ante_candidates
        from .chat_nick_actor import prove_actor_point, point_matches_hand, SOURCE_KEY
        from .joint_start_stack import prove,matches_current_hand
        self.native_who=native_who
        source=Path(source);windows=[];by_key={};self.proofs={};self.rows=[]
        for hand in hands:
            key=(int(str(hand['table_id']).lstrip('T')),hand['session_id'],hand['hand_id'])
            if key in by_key:raise ValueError('JOINT_DUPLICATE_HAND_CONTEXT')
            by_key[key]=hand
            rows=[r for m in hand['messages'] for r in m['source_rows']]
            headers=[r for r in rows if HEADER.fullmatch(r['raw_ocr_text'].strip()) and
                     HEADER.fullmatch(r['raw_ocr_text'].strip())[1]==hand['hand_id'] and
                     (':R' in str(r.get('stable_row_id')) or ':F' in str(r.get('stable_row_id')))]
            if not headers:continue
            frame=min(r['frame_id'] for r in headers)
            windows.append(dict(table_id=key[0],session_id=key[1],hand_id=key[2],
                header_frame=frame,capture_end_inclusive=frame+capture_frames))
        loaded=load_contexts(source,windows,actor_policy=actor_policy);self.source_hashes=loaded['source_hashes']
        self.source=source;self.windows=windows
        self.contexts={(c['table_id'],c['session_id'],c['hand_id']):c for c in loaded['contexts']}
        captures=defaultdict(list);digest=hashlib.sha256()
        with (source/'start_stack_capture.jsonl').open('rb') as stream:
            for lineno,line in enumerate(stream,1):
                digest.update(line);r=json.loads(line)
                if r['field']!='stack':continue
                for c in loaded['contexts']:
                    if ((r['table_id'],r['session_id'])==(c['table_id'],c['session_id']) and
                            c['header_frame']<=r['source_frame']<=c['capture_end_inclusive']):
                        captures[(c['table_id'],c['session_id'],c['hand_id'],r['seat'])].append(dict(source_line=lineno,raw_record=r))
        if digest.hexdigest()!=self.source_hashes['start_stack_capture.jsonl']:
            raise ValueError('JOINT_CAPTURE_CHANGED_DURING_LOAD')
        database_name=stack_timeline_source_name(source)
        database=source/database_name
        self.source_hashes[database_name]=hashlib.sha256(database.read_bytes()).hexdigest()
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            metadata=[dict(r) for r in db.execute('select key,value from metadata order by key')]
            video=[json.loads(r['value']) for r in metadata if r['key']=='video_sha256']
            if video!=[video_sha256] or not video_sha256:raise ValueError('JOINT_VIDEO_IDENTITY_CONFLICT')
            for c in loaded['contexts']:
                key=(c['table_id'],c['session_id'],c['hand_id'])
                actors=sorted({p['actor'] for p in ante_candidates(c['ordinary_prefix_timeline'])} |
                              {p['actor'] for p in c.get(SOURCE_KEY,{}).get('chat_records',[])})
                for actor in actors:
                    point=prove_actor_point(c,actor,native_who=self.native_who)
                    if point['status']=='PROVEN' and not point_matches_hand(point,by_key[key]):
                        point=dict(status='UNKNOWN',reason='CURRENT_HAND_ACTOR_SOURCE_CONFLICT')
                    audit=dict(table_id=key[0],session_id=key[1],hand_id=key[2],actor=actor,
                               point_status=point['status'],point_reason=point['reason'],status='UNKNOWN')
                    if point['status']!='PROVEN':self.rows.append(audit);continue
                    seat=point['seat'];audit['seat']=seat
                    if sitout_provider is not None and sitout_provider.excludes_stack_work(key,actor):
                        audit.update(status='NOT_REQUIRED_SIT_OUT',reason='SOURCE_BOUND_STATUS_NO_NUMERIC_RECOVERY')
                        self.rows.append(audit);continue
                    obs=[dict(r) for r in db.execute('select * from observations where table_id=? and seat_id=? '
                        'and frame_index<=? and last_seen_frame>=? order by observation_id',
                        (key[0],seat,c['capture_end_inclusive'],c['header_frame']))]
                    states=[]
                    for sid in sorted({r['state_id'] for r in obs}):
                        r=dict(db.execute('select * from visual_states where state_id=?',(sid,)).fetchone())
                        pixels=r.pop('raw_gray');r.update(raw_gray_base64=base64.b64encode(pixels).decode(),
                            raw_gray_sha256=hashlib.sha256(pixels).hexdigest());states.append(r)
                    case=dict(actor=actor,seat=seat,table_id=key[0],session_id=key[1],hand_id=key[2],
                        source_frame_range=[c['header_frame'],c['capture_end_inclusive']],source_context=c,
                        raw_stack_captures=captures[(*key,seat)],raw_sql_observations=obs,raw_sql_visual_states=states)
                    proof=prove(case,metadata,native_who=self.native_who);audit.update(status=proof['status'],reason=proof['reason'])
                    if proof['status']=='PROVEN':
                        if matches_current_hand(proof,by_key[key]):
                            self.proofs.setdefault(key,[]).append(proof)
                            audit.update(calculated_start_stack=proof['calculated_start_stack'])
                        else:audit.update(status='UNKNOWN',reason='CURRENT_NATIVE_HAND_SOURCE_CONFLICT')
                    self.rows.append(audit)
        if self.native_who is not None and not self.native_who.sources_unchanged():
            raise ValueError('JOINT_NATIVE_WHO_SOURCE_CHANGED')
        for name,digest in self.source_hashes.items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('JOINT_SOURCE_CHANGED_DURING_PROOF: '+name)

    def for_hand(self,hand):
        from .joint_start_stack import verify_retained,RULE
        key=(int(str(hand['table_id']).lstrip('T')),hand['session_id'],hand['hand_id'])
        if not hasattr(self,'source'):
            # Historical retained stores supported only the legacy contract.
            # New-policy proofs always require an independently loaded context.
            return [p for p in self.proofs.get(key,[]) if p.get('rule')==RULE and verify_retained(p,hand)]
        def unchanged():
            return all(hashlib.sha256((self.source/n).read_bytes()).hexdigest()==digest
                       for n,digest in self.source_hashes.items()) and (
                           self.native_who is None or self.native_who.sources_unchanged())
        if not unchanged():return []
        result=[p for p in self.proofs.get(key,[]) if verify_retained(p,hand,self.contexts.get(key),native_who=self.native_who)]
        return result if unchanged() else []

    def write_audit(self,output):
        output=Path(output)
        with (output/'joint_start_stack_proofs.jsonl').open('w',encoding='utf-8') as stream:
            for proofs in self.proofs.values():
                for proof in proofs:stream.write(json.dumps(proof,ensure_ascii=False)+'\n')
        (output/'joint_start_stack_audit.json').write_text(json.dumps(dict(
            scope='INDEPENDENT_OWN_FORCED_FIRST_NUMBER_NO_OCCUPANCY_EPOCH',source_hashes=self.source_hashes,
            counts=dict(Counter(r['status'] for r in self.rows)),rows=self.rows),ensure_ascii=False,indent=2),'utf-8')
