"""Post-event logical gate over the existing assembled source hand records.

Never changes capture, OCR, timeline rows, messages, or observed hand boundaries.
Excel keeps every record. Only the downstream complete selection is gated.
"""
import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from parser_core.temporal_overlap import hand_number, START
from parser_core.poker_pipeline import analyze_timeline
from parser_core.poker_hand_assembly import parse_text


def table_key(table):
    text=str(table)
    return int(text[1:]) if text.startswith('T') and text[1:].isdigit() else int(text) if text.isdigit() else text


def context_key(hand):
    return table_key(hand['table_id']), hand['session_id'], hand['hand_id']


def physical_start_corroboration(record):
    """A newly admitted numbered header plus its adjacent physical start text.

    A sidecar can read an already visible initial header long after frame zero.
    Only ordinary admitted new rows can remove that initial-context uncertainty.
    R rows enter OCR exclusively at GLOBAL_NEW_BOTTOM; F rows carry their own
    independent physical-birth receipt. Neither an OCR callback time nor a
    sidecar header alone establishes birth.
    """
    hand=record['observed_hand'];life=record['lifecycle']
    if life['status'] not in {'COMPLETE','END_PARTIAL'}:return []
    def new_source(row):
        rid=str(row.get('stable_row_id') or row.get('row_id') or '')
        return (row.get('origin')=='NEW_CANDIDATE' and row.get('source_kind')!='SIDECAR_HEADER_RECEIPT'
            and row.get('ocr_status')=='OK' and row.get('frame_id',0)>0
            and re.fullmatch(r'[0-9a-f]{64}',str(row.get('row_pixel_sha256') or ''))
            and (re.search(r':R\d+$',rid) or re.search(r':F\d+$',rid)
                and row.get('admission_ref') and row.get('admission_reason')=='PROVEN_PHYSICAL_NEW; OLD_IDENTITY_UNKNOWN'))
    for message in hand['messages']:
        source=message['source_rows']
        if (message['action'] not in {'HAND_HEADER','HAND_ID'} or len(source)!=1
                or not new_source(source[0]) or not set(message['source_row_ids'])<=set(life['start_evidence_refs'])):continue
        header=source[0]
        for following in hand['messages']:
            continuation=following['source_rows']
            number=parse_text(following['raw_text'])['hand_number']
            if (following['action']!='NEW_HAND' or number!=hand['hand_id'] or not continuation
                    or not all(new_source(r) for r in continuation)):continue
            sequence=source+continuation
            if all(r.get('frame_id')==header['frame_id'] and all(r.get(k)==header.get(k)
                    for k in ('table_id','session_id','epoch')) and r.get('row_index')==header.get('row_index',-1)+j
                    for j,r in enumerate(sequence)):
                return [r.get('stable_row_id',r.get('row_id')) for r in sequence]
    return []


def gate_first_context(record):
    evidence=physical_start_corroboration(record)
    if evidence:
        record['lifecycle']['first_context_start_proof']=dict(
            reason='NEW_PHYSICAL_NUMBERED_HEADER_AND_ADJACENT_START',source_row_ids=evidence)
    else:
        record['lifecycle'].update(status='START_PARTIAL',reason='FIRST_VISIBLE_CONTEXT_START_NOT_PROVEN',
            trace=record['lifecycle']['trace']+['START_PARTIAL'])


def classify_lifecycle(records):
    """Annotate only logical metadata; reuse anchored start and terminal contracts."""
    highest={}
    for record in records:
        h=record['observed_hand'];context=(h['table_id'],h['session_id'])
        started=False;ended=False;barrier=False;reason='START_NOT_OBSERVED'
        seen_material=False;trace=['WAIT_HAND_START'];start_refs=[];terminal_refs=[]
        for m in h['messages']:
            source=m['source_rows']
            safe=all(r.get('ocr_status')!='ERROR' and r.get('status') not in {'ERROR','UNREADABLE'} for r in source)
            numbers=[hand_number(str(r.get('raw_text') or r.get('raw_ocr_text') or r.get('text') or '')) for r in source]
            number=next((n for n in numbers if n),None)
            marker_like=('hand #' in m['raw_text'].lower() or m['raw_text'].strip().startswith('#'))
            marker_like = marker_like or any(r.get('source_kind')=='SIDECAR_HEADER_RECEIPT' and
                                              r.get('ocr_status')!='OK' for r in source)
            marker=safe and number is not None and m['action'] in {'HAND_HEADER','HAND_ID','NEW_HAND'}
            if marker:
                old=highest.get(context)
                fresh=old is None or int(number)>int(old)
                repeated=started and number==h['hand_id'] and number==old
                if fresh and not seen_material and number==h['hand_id'] and not barrier:
                    highest[context]=number;started=True;start_refs.extend(m['source_row_ids'])
                    trace.append('IN_PROGRESS');reason='START_OBSERVED_END_UNPROVEN'
                elif not repeated:
                    barrier=True;reason='OLDER_OR_CONFLICTING_HAND_HEADER'
            elif marker_like:
                barrier=True;reason='UNKNOWN_HAND_BOUNDARY'
            elif START.fullmatch(m['normalized_text']):
                # Existing explicit Starting metadata may corroborate an already
                # anchored prefix; it never supplies a numbered start itself.
                if seen_material or not safe:
                    barrier=True;reason='UNNUMBERED_STARTING_BOUNDARY'
            elif m['action'] not in {'TABLE_INFO','BLIND_LEVEL'}:
                seen_material=True
            if started and not barrier and safe and m['action'] in {'WIN','HAND_END'}:
                ended=True;terminal_refs.extend(m['source_row_ids']);trace.append('COMPLETE')
        status=('COMPLETE' if started and ended and not barrier else
                'END_PARTIAL' if started else 'UNKNOWN_UNPROVEN' if barrier else 'START_PARTIAL')
        if status=='COMPLETE':reason='START_AND_OWN_TERMINAL_OBSERVED'
        elif started and not barrier:reason='NEXT_BOUNDARY_WITHOUT_TERMINAL' if h['next_boundary_observed'] else 'EOF_OR_CONTEXT_END_WITHOUT_TERMINAL'
        source=[r for m in h['messages'] for r in m['source_rows']]
        frames=[r['frame_id'] for r in source if r.get('frame_id') is not None]
        record['lifecycle']=dict(status=status,reason=reason,table_id=h['table_id'],session_id=h['session_id'],hand_id=h['hand_id'],
            first_source_frame=min(frames,default=None),last_source_frame=max(frames,default=None),
            evidence_refs=[rid for m in h['messages'] for rid in m['source_row_ids']],
            start_evidence_refs=start_refs,terminal_evidence_refs=terminal_refs,trace=trace+[status])
    # Conflicting duplicate segments cannot lend a COMPLETE context to math.
    states=defaultdict(set)
    for r in records:states[context_key(r['observed_hand'])].add(r['lifecycle']['status'])
    eligible={k for k,v in states.items() if v=={'COMPLETE'}}
    complete=[r for r in records if r['lifecycle']['status']=='COMPLETE' and context_key(r['observed_hand']) in eligible]
    summary=dict(raw_hand_contexts=len(records),statuses=dict(Counter(r['lifecycle']['status'] for r in records)),
        complete_downstream_hands=len(complete),excluded_downstream_hands=len(records)-len(complete),
        excel_policy='ALL_OBSERVED_EVENTS_AND_HANDS',downstream_policy='COMPLETE_ONLY',
        quality_validation_statuses=dict(Counter(r['validation']['status'] for r in complete)))
    return complete,eligible,summary


def write_lifecycle(output, records, summary):
    output=Path(output)
    with (output/'hand_lifecycle.jsonl').open('w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(dict(segment_id=r['segment_id'],**r['lifecycle']),ensure_ascii=False)+'\n')
    with (output/'partial_hand_diagnostics.jsonl').open('w',encoding='utf-8') as f:
        for r in records:
            if r['lifecycle']['status']!='COMPLETE':
                f.write(json.dumps(dict(segment_id=r['segment_id'],**r['lifecycle'],observed_hand=copy.deepcopy(r['observed_hand'])),ensure_ascii=False)+'\n')
    (output/'lifecycle_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')


def source_bound_headers(output):
    """Read admitted, completed source receipts without changing their authority."""
    output=Path(output)
    state_path=output/'hand_state_receipts.jsonl';header_path=output/'hand_header_ocr.jsonl'
    if not state_path.exists() or not header_path.exists():
        return [], {}, []
    return _source_bound_headers(output, state_path, header_path)


def post_excel_contexts(output, observed_records):
    """Read real confirmed sidecar source rows into a separate logical view.

    Nothing is added to the original timeline or Excel. Source frame/position,
    raw header OCR, hash and both receipt references are retained. The existing
    hand assembler/analysis is reused on this additional source-bound view.
    """
    output=Path(output)
    raw_rows=[copy.deepcopy(r) for record in observed_records
              for m in record['observed_hand']['messages'] for r in m['source_rows']]
    frozen=copy.deepcopy(observed_records)
    if any(r.get('canonical_stream_version') for r in raw_rows):
        # The canonical join already consumed these exact source receipts.
        logical=copy.deepcopy(observed_records)
        complete,eligible,summary=classify_lifecycle(logical)
        _,anchors,_=source_bound_headers(output)
        first_frames={}
        for (table,session,number),anchor in anchors.items():
            first_frames[(table,session)]=min(first_frames.get((table,session),anchor[0]),anchor[0])
        first_contexts={k for k,a in anchors.items() if a[0]==first_frames[k[:2]]}
        for r in logical:
            if context_key(r['observed_hand']) in first_contexts:
                gate_first_context(r)
        states=defaultdict(set)
        for r in logical:states[context_key(r['observed_hand'])].add(r['lifecycle']['status'])
        eligible={k for k,v in states.items() if v=={'COMPLETE'}}
        complete=[r for r in logical if context_key(r['observed_hand']) in eligible]
        summary.update(statuses=dict(Counter(r['lifecycle']['status'] for r in logical)),
            complete_downstream_hands=len(complete),excluded_downstream_hands=len(logical)-len(complete),
            header_receipt_source='CANONICAL_STREAM',sidecar_source_rows=sum(r.get('source_kind')=='SIDECAR_HEADER_RECEIPT' for r in raw_rows))
        return logical,complete,eligible,summary
    state_path=output/'hand_state_receipts.jsonl';header_path=output/'hand_header_ocr.jsonl'
    if not state_path.exists() or not header_path.exists():
        logical=copy.deepcopy(observed_records)
        complete,eligible,summary=classify_lifecycle(logical)
        summary.update(header_receipt_source='UNAVAILABLE',sidecar_source_rows=0)
        return logical,complete,eligible,summary
    headers,anchors,invalid=source_bound_headers(output)
    return _logical_contexts(raw_rows, headers, anchors, invalid, observed_records, frozen)


def _source_bound_headers(output, state_path, header_path):
    def position(r):return (table_key(r['table_id']),r['session_id'],r['source_frame'],r['source_span_index'],r.get('source_pts'))
    states={}
    for line,text in enumerate(state_path.read_text('utf-8').splitlines(),1):
        r=json.loads(text)
        if r.get('status')=='HAND_HEADER_CONFIRMED':states[position(r)]=(line,r)
    admitted={};anchors={};headers=[];invalid=[];completed=set()
    with header_path.open(encoding='utf-8') as f:
        for line,text in enumerate(f,1):
            r=json.loads(text)
            if r.get('status')=='ADMITTED':admitted[position(r)]=(line,r);continue
            if 'ocr_response' not in r:continue
            p=position(r);completed.add(p);response=r['ocr_response'];number=hand_number(response.get('raw_text') or '')
            state=states.get(p);admission=admitted.get(p)
            proven=(r.get('status')=='HAND_HEADER_CONFIRMED' and response.get('status')=='OK' and
                    number is not None and state is not None and state[1]['hand_id']==number and
                    admission is not None and admission[1]['pixel_sha256']==r.get('pixel_sha256'))
            # Known stale/context-rejected headers cannot create a new context.
            # Keep unresolved header source as a logical barrier, not a guess.
            if not proven and r.get('status') in {'STALE_HEADER_IGNORED','STALE_SESSION_IGNORED'}:continue
            source_id=f'SIDECAR_HEADER:{p[0]}:{p[1]}:{p[2]}:{p[3]}:PTS={p[4]}'
            row=dict(stable_row_id=source_id,row_id=source_id,table_id=f'T{p[0]:02d}' if isinstance(p[0],int) else p[0],
                session_id=p[1],frame_id=p[2],row_index=p[3],source_pts=p[4],
                time_base=admission[1].get('time_base') if admission else None,
                epoch=None,raw_ocr_text=response.get('raw_text') or '',confidence=response.get('score'),
                row_pixel_sha256=r.get('pixel_sha256'),video_time=None,source_kind='SIDECAR_HEADER_RECEIPT',
                origin='CONFIRMED_HEADER' if proven else 'UNRESOLVED_HEADER',
                ocr_status='OK' if proven else 'ERROR',status='RAW_UNREVIEWED' if proven else 'ERROR',
                header_provenance=dict(header_file=header_path.name,header_line=line,
                    state_file=state_path.name,state_line=state[0] if state else None,
                    admission_line=admission[0] if admission else None,
                    row_bounds=admission[1].get('row_bounds') if admission else None,
                    raw_header_receipt=copy.deepcopy(r),state_receipt=copy.deepcopy(state[1]) if state else None))
            headers.append(row)
            if proven:
                k=(p[0],p[1],number)
                anchors[k]=min(anchors.get(k,(p[2],p[3])),(p[2],p[3]))
            else:invalid.append(source_id)
    # EOF/drain does not manufacture a successful OCR response. An admitted
    # source with missing completion (or orphan state receipt) remains a fence.
    for p in (set(admitted)|set(states))-completed:
        admission=admitted.get(p);state=states.get(p)
        source_id=f'SIDECAR_HEADER:{p[0]}:{p[1]}:{p[2]}:{p[3]}:PTS={p[4]}'
        headers.append(dict(stable_row_id=source_id,row_id=source_id,
            table_id=f'T{p[0]:02d}' if isinstance(p[0],int) else p[0],session_id=p[1],
            frame_id=p[2],row_index=p[3],source_pts=p[4],epoch=None,raw_ocr_text='',
            row_pixel_sha256=admission[1].get('pixel_sha256') if admission else None,
            time_base=admission[1].get('time_base') if admission else None,video_time=None,
            source_kind='SIDECAR_HEADER_RECEIPT',origin='UNRESOLVED_HEADER',ocr_status='ERROR',status='ERROR',
            header_provenance=dict(header_file=header_path.name,header_line=None,
                admission_line=admission[0] if admission else None,
                admission_receipt=copy.deepcopy(admission[1]) if admission else None,
                state_file=state_path.name,state_line=state[0] if state else None,
                state_receipt=copy.deepcopy(state[1]) if state else None,
                blocker='HEADER_COMPLETION_COVERAGE_UNKNOWN')))
        invalid.append(source_id)
    return headers,anchors,invalid


def _logical_contexts(raw_rows, headers, anchors, invalid, observed_records, frozen):
    # Keep source order within each table/session, independent of callback order.
    logical_rows=raw_rows+headers
    logical_rows.sort(key=lambda r:(str(table_key(r['table_id'])),r['session_id'],r['frame_id'],r['row_index'],r['stable_row_id']))
    logical,analysis=analyze_timeline(logical_rows,ocr_status={r['stable_row_id']:r.get('ocr_status','ERROR') for r in logical_rows})
    complete,eligible,summary=classify_lifecycle(logical)
    first_frames={}
    for (table,session,number),anchor in anchors.items():
        first_frames[(table,session)]=min(first_frames.get((table,session),anchor[0]),anchor[0])
    first_contexts={k for k,a in anchors.items() if a[0]==first_frames[k[:2]]}
    for record in logical:
        if context_key(record['observed_hand']) in first_contexts:
            gate_first_context(record)
    complete=[r for r in logical if r['lifecycle']['status']=='COMPLETE']
    states_by_context=defaultdict(set)
    for r in logical:states_by_context[context_key(r['observed_hand'])].add(r['lifecycle']['status'])
    eligible={k for k,v in states_by_context.items() if v=={'COMPLETE'}}
    complete=[r for r in complete if context_key(r['observed_hand']) in eligible]
    summary.update(raw_excel_hand_segments=len(observed_records),raw_excel_event_rows=len(raw_rows),
        raw_hand_contexts=len(logical),statuses=dict(Counter(r['lifecycle']['status'] for r in logical)),
        complete_downstream_hands=len(complete),excluded_downstream_hands=len(logical)-len(complete),
        quality_validation_statuses=dict(Counter(r['validation']['status'] for r in complete)),
        sidecar_source_rows=len(headers),distinct_confirmed_header_contexts=len(anchors),
        unresolved_header_refs=invalid,header_receipt_source='CONFIRMED_SOURCE_BOUND_SIDECAR_AFTER_EXCEL',
        source_row_coverage_exact=analysis['source_rows']==analysis['source_rows_covered'],
        source_time_note='Frame/PTS denotes source observation, not inferred action onset')
    if observed_records!=frozen:raise ValueError('Lifecycle changed original observed records')
    return logical,complete,eligible,summary
