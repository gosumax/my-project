"""Saved post-pixel Smoke19 quality replay; launched only by run_production.py."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import shutil
import threading
import sys
import time
from collections import Counter
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import base64
import cv2
import numpy as np

from parser_core.normalize_chat import normalize, normalize_with_service
from parser_core.service_text_normalizer import physical_service_group
from parser_core.message_assembly import RowFact, group_rows
from parser_core.stack_timeline_storage import resolve_stack_timeline_path, stack_timeline_source_name
from parser_core.poker_pipeline import analyze_timeline
from parser_core.hand_lifecycle import post_excel_contexts, write_lifecycle, context_key, table_key
from parser_core.poker_math_engine import write_start_stack_math
from parser_core.poker_reconstruction_layer import write_reconstruction
from parser_core.seat_forced_binding import SeatForcedBinding, PendingSourceJobs
from parser_core.seat_epoch_state import SeatEpochMachine
from parser_core.parallel_ocr import row_key
from parser_core.canonical_events import canonical_join


def read(path):
    return [json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]


def write(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), 'utf-8')


def _validation_require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _validation_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _validation_empty_transition(root,pinned,transition):
    """One reviewed acquisition hook delta; never waive original source pins."""
    names={'parser_core/start_stack_live.py','parser_core/empty_fast_capture.py',
           'off_empty_dispatch.py','off_empty_bound.py','off_empty_pixels.py'}
    producer=str((root/'parser_core/start_stack_live.py').resolve())
    _validation_require(transition.get('schema')=='poker.saved-empty-runtime-transition.v1',
                        'SAVED_VALIDATION_TRANSITION_SCHEMA')
    closure=transition['runtime_sha256']
    _validation_require(set(closure)==names,'SAVED_VALIDATION_TRANSITION_SCOPE')
    before=transition['before'];old=Path(before['path']).resolve()
    _validation_require(old!=Path(producer) and before['sha256']==pinned.get(producer) and
        _validation_sha(old)==before['sha256'],'SAVED_VALIDATION_TRANSITION_OLD')
    for name,digest in closure.items():
        _validation_require(_validation_sha(root/name)==digest,
                            'SAVED_VALIDATION_TRANSITION_RUNTIME:'+name)
    receipts={}
    for role in ('proof','review'):
        ref=transition[role];path=Path(ref['path']).resolve()
        _validation_require(_validation_sha(path)==ref['sha256'],
                            'SAVED_VALIDATION_TRANSITION_RECEIPT:'+role)
        receipts[role]=json.loads(path.read_text('utf-8'))
    proof=receipts['proof'];review=receipts['review']
    _validation_require(proof.get('verdict')=='PASS' and
        all(proof.get(name)==0 for name in ('decision_mismatches','state_mismatches','source_receipt_mismatches')) and
        proof.get('native_initial_hands')==57 and proof.get('native_joint_hands')==16 and
        proof.get('fast_capture_batches',0)>0 and proof.get('callback_schedules',0)>=3 and
        proof.get('producer_body_equivalence')=='PASS' and
        all(proof.get(name)==0 for name in ('new_ocr_calls','new_solver_calls','new_video_decode_calls')),
        'SAVED_VALIDATION_TRANSITION_PROOF_NOT_PASS')
    _validation_require(all(proof['source_input_sha256'].get(str((root/name).resolve()))==digest
        for name,digest in closure.items()),'SAVED_VALIDATION_TRANSITION_PROOF_RUNTIME')
    _validation_require(proof['source_input_sha256'].get(str(old))==before['sha256'],
                        'SAVED_VALIDATION_TRANSITION_PROOF_OLD')
    _validation_require(review.get('verdict')=='PASS' and review.get('runtime_sha256')==closure and
        review.get('proof_sha256')==transition['proof']['sha256'] and
        review.get('before_sha256')==before['sha256'],
                        'SAVED_VALIDATION_TRANSITION_REVIEW_NOT_PASS')
    # Retain the historical producer pin in the input seals. Only its execution
    # pin advances, with immutable OLD copy and exact lifecycle proof as evidence.
    pinned[producer]=closure['parser_core/start_stack_live.py']
    for path,digest in ((str(old),before['sha256']),*( (str(Path(transition[r]['path']).resolve()),
        transition[r]['sha256']) for r in ('proof','review'))):
        _validation_require(path not in pinned or pinned[path]==digest,'SAVED_VALIDATION_TRANSITION_PIN_CONFLICT')
        pinned[path]=digest
    for name,digest in closure.items():
        path=str((root/name).resolve())
        _validation_require(name=='parser_core/start_stack_live.py' or path not in pinned or pinned[path]==digest,
                            'SAVED_VALIDATION_TRANSITION_EXISTING_PIN_CONFLICT')
        pinned[path]=digest


def _validation_bundle(path, no_sb_review):
    """Data locators only. Never execute code or derive new candidate choices."""
    path=Path(path).resolve()
    raw=path.read_bytes();bundle=json.loads(raw)
    _validation_require(bundle.get('schema')=='poker.saved-hh-validation.v1','SAVED_VALIDATION_SCHEMA')
    roles={'logical','traces','stacks','seats','who','associations','who_updates',
        'sitout_targets','sitout_histories','no_sb_seats','visual','review','reference_hh',
        'reference_text','original_seals'}
    _validation_require(set(bundle['files'])==roles,'SAVED_VALIDATION_ROLES')
    files={role:Path(value['path']).resolve() for role,value in bundle['files'].items()}
    _validation_require(Path(no_sb_review).resolve()==files['review'],'SAVED_VALIDATION_REVIEW_LOCATOR')
    for role,value in bundle['files'].items():
        _validation_require(_validation_sha(files[role])==value['sha256'],f'SAVED_VALIDATION_INPUT_CHANGED:{role}')
    # These are the only intentional differences from the original candidate.
    # Every other old source/config/model/native input must still reproduce.
    root=Path(__file__).resolve().parents[1]
    routing={str((root/name).resolve()) for name in ('run_production.py',
        'scripts/replay_saved_quality.py','production_manifest.json','PRODUCTION.md')}
    original=json.loads(files['original_seals'].read_text('utf-8'))
    _validation_require(len(original)>0,'SAVED_VALIDATION_EMPTY_SOURCE_SEAL')
    pinned={str(Path(p).resolve()):digest for p,digest in original.items() if str(Path(p).resolve()) not in routing}
    if 'empty_runtime_transition' in bundle:
        _validation_empty_transition(root,pinned,bundle['empty_runtime_transition'])
    for p,digest in pinned.items():
        _validation_require(_validation_sha(p)==digest,f'SAVED_VALIDATION_SOURCE_CHANGED:{p}')
    for role in ('logical','seats','who','associations','who_updates','sitout_targets',
                 'sitout_histories','visual','review'):
        _validation_require(pinned.get(str(files[role]))==bundle['files'][role]['sha256'],
                            f'SAVED_VALIDATION_ROLE_NOT_ORIGINAL:{role}')
    # The ALL57 closure omitted two native containers; their historical pins
    # come from the separate SIT_OUT audit. Exact row reproof remains required.
    extra=bundle.get('native_revalidation_inputs',{})
    if extra:
        history_ref=bundle['native_parent_history'];history_path=Path(history_ref['path']).resolve()
        _validation_require(_validation_sha(history_path)==history_ref['sha256'],
                            'SAVED_VALIDATION_NATIVE_HISTORY_CHANGED')
        history=json.loads(history_path.read_text('utf-8'))
        _validation_require(history['parity']=='PASS' and not history['changed'],
                            'SAVED_VALIDATION_NATIVE_HISTORY_NOT_PASS')
        source=Path(bundle['native_source']).resolve()
        allowed={str(source/'start_stack_capture.jsonl'),str(resolve_stack_timeline_path(source))}
        for path,digest in extra.items():
            resolved=str(Path(path).resolve())
            _validation_require(resolved==path and path in allowed and path not in pinned,
                                f'SAVED_VALIDATION_ADDITIONAL_PARENT_INVALID:{path}')
            prior=history['before'].get(path)
            _validation_require(prior is not None and prior==history['after'].get(path) and
                prior['sha256']==digest and prior['bytes']==Path(path).stat().st_size,
                f'SAVED_VALIDATION_NATIVE_HISTORICAL_PARITY:{path}')
            _validation_require(_validation_sha(path)==digest,f'SAVED_VALIDATION_ADDITIONAL_PARENT_CHANGED:{path}')
            pinned[path]=digest
    return bundle,files,pinned,hashlib.sha256(raw).hexdigest()


def _validation_native_seals(source, files, pinned, seat_rows):
    """Require historical seals before providers can consume native parents."""
    from parser_core.current_chat_sources import FILES
    paths={source/name for name in (*FILES,'seat_forced_inputs.jsonl',
                                   'seat_forced_summary.json','run_metadata.json')}
    for review in read(files['review']):
        manual=review['manual_source']
        path=Path(manual['path']).resolve()
        _validation_require(pinned.get(str(path))==manual['sha256'],
                            f'SAVED_VALIDATION_MANUAL_SOURCE_UNSEALED:{path}')
        paths.add(path)
    # The exporter may independently reload a contradictory SIT_OUT status.
    # Numeric/WHO locators otherwise remain retained facts, not fresh acquisition.
    for row in seat_rows:
        for status in row.get('sitout_sources',[]):
            paths.add(Path(status['source']['file']).resolve())
        claim=row.get('forced_allin_stack_claim') or {}
        if row.get('forced_allin_stack_conflict') and claim.get('source'):
            paths.update(Path(claim['source']).resolve()/name for name in FILES)
    for path in paths:
        _validation_require(str(path) in pinned,f'SAVED_VALIDATION_NATIVE_SOURCE_UNSEALED:{path}')


def _validation_finish_seals(initial, bundle_path, no_sb_review, execution):
    _validation_require(_validation_bundle(bundle_path,no_sb_review)==initial,
                        'SAVED_VALIDATION_BUNDLE_CHANGED_DURING_RUN')
    for path,digest in execution.items():
        _validation_require(_validation_sha(path)==digest,
                            f'SAVED_VALIDATION_EXECUTION_SOURCE_CHANGED:{path}')


_TRACE_VALIDATION_TIMINGS=tuple(('derived_final_validation',section,*leaf)
    for section in ('validation','diagnostics') for leaf in (('logic_ms',),('math','elapsed_ms')))
_HH_VALIDATION_TIMINGS=(*_TRACE_VALIDATION_TIMINGS,
    *(('reconstruction',*path) for path in _TRACE_VALIDATION_TIMINGS))


def _validation_value(value,path):
    for key in path:value=value[key]
    return value


def _validation_difference(old, new, timing_paths=()):
    """Strict semantic parity, excluding only audited regenerated stopwatches."""
    def label(path):return '$'+''.join(f'[{k}]' if type(k) is int else '.'+k for k in path)
    for path in timing_paths:
        try:values=(_validation_value(old,path),_validation_value(new,path))
        except (KeyError,TypeError):return label(path)+':MISSING_TIMING'
        if not all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values):
            return label(path)+':INVALID_TIMING'
    def difference(a,b,path=()):
        if path in timing_paths:return None
        if type(a) is not type(b):return label(path)+':TYPE'
        if isinstance(a,dict):
            for key in sorted(set(a)|set(b)):
                if key not in a or key not in b:return label((*path,key))
                found=difference(a[key],b[key],(*path,key))
                if found is not None:return found
        elif isinstance(a,list):
            if len(a)!=len(b):return label(path)+'.length'
            for index,(x,y) in enumerate(zip(a,b)):
                found=difference(x,y,(*path,index))
                if found is not None:return found
        else:return None if a==b else label(path)
        return None
    return difference(old,new)


class _SavedValidationActors:
    """Views of sealed WHO decisions, retaining literal/canonical provenance."""
    def __init__(self, rows, updates):
        from parser_core.simple_saved_policy import hand_key
        self.key=hand_key;self.rows=defaultdict(dict);self.projections={}
        for row in rows:
            key=hand_key(row)
            _validation_require(row['player'] not in self.rows[key],'SAVED_VALIDATION_DUPLICATE_WHO')
            self.rows[key][row['player']]=copy.deepcopy(row)
        for update in updates:
            key=tuple(update['hand']);actor=update['actor']
            _validation_require(actor in self.rows[key],'SAVED_VALIDATION_WHO_UPDATE_UNBOUND')
            self.rows[key][actor].update(new_status='PROVEN',new_seat=update['proof']['seat'],
                                         new_proof=copy.deepcopy(update['proof']))

    def actor_map(self,hand):
        return {actor:row['new_proof']['canonical_actor'] for actor,row in self.rows[self.key(hand)].items()
                if row['new_status']=='PROVEN'}

    def canonical_hand(self,hand):
        result=copy.deepcopy(hand);mapping=self.actor_map(hand)
        for message in result['messages']:
            for field in ('player','actor'):
                if message.get(field) in mapping:
                    message['original_'+field]=message[field];message[field]=mapping[message[field]]
        result['nickname_canonicalization']=mapping
        return result

    def for_hand(self,hand):
        key=self.key(hand);result={};actors=set();groups=defaultdict(list)
        core=('status','canonical_actor','actor','seat','table_id','session_id','hand_id',
              'anchor','identity_interval')
        for row in self.rows[key].values():
            if row['new_status']!='PROVEN':continue
            proof=copy.deepcopy(row['new_proof']);proof['actor']=proof['canonical_actor']
            _validation_require(all(field in proof for field in core) and
                proof['status']=='PROVEN' and self.key(proof)==key,'SAVED_VALIDATION_WHO_CONFLICT')
            if proof['seat'] in result:
                # Match the sealed candidate's source-order last-row projection.
                # Identity and native authority must agree; the selected complete
                # proof remains subject to the full HH/reference comparison.
                previous=result[proof['seat']]
                _validation_require(_validation_difference({k:previous[k] for k in core},
                    {k:proof[k] for k in core}) is None,'SAVED_VALIDATION_WHO_CONFLICT')
            else:
                _validation_require(proof['actor'] not in actors,'SAVED_VALIDATION_WHO_CONFLICT')
            groups[proof['seat']].append(copy.deepcopy(row))
            actors.add(proof['actor']);result[proof['seat']]=proof
        self.projections[key]=[dict(hand=list(key),seat=seat,raw_rows=rows,
            selected_proof=copy.deepcopy(result[seat])) for seat,rows in groups.items() if len(rows)>1]
        return list(result.values())


class _SavedValidationFacts:
    """Retained fact transport; current policy consumes/rechecks each row."""
    def __init__(self, records, seats, actors, no_sb, source, updates, sitouts, no_sb_seats):
        from parser_core.simple_saved_policy import hand_key,forced_correction
        from parser_core.simple_stack_policy import parse_stack,corrected_start_stack
        self.key=hand_key;self.rows=defaultdict(dict);self.actor_sources=actors
        self.no_sb_source=no_sb;self.native_source=source
        for original in seats:
            row=copy.deepcopy(original);key=hand_key(row);first=row['first_numeric_source']
            _validation_require(row['seat'] not in self.rows[key],'SAVED_VALIDATION_DUPLICATE_SEAT')
            if first is not None:
                _validation_require(parse_stack(first['raw_text'])==row['DISPLAYED_STACK'] and
                    row['hand_start']<=first['frame']<=row['hand_end'] and
                    (first['table_id'],first['session_id'],first['seat'])==(row['table_id'],row['session_id'],row['seat']),
                    'SAVED_VALIDATION_NUMERIC_CONTEXT')
            else:_validation_require(row['DISPLAYED_STACK'] is None,'SAVED_VALIDATION_NUMERIC_MISSING')
            _validation_require(corrected_start_stack(row['DISPLAYED_STACK'],row['forced'])==row['START_STACK'],
                                'SAVED_VALIDATION_RETAINED_STACK_MATH')
            self.rows[key][row['seat']]=row
        for update in updates:
            proof=update['proof'];row=self.rows[tuple(update['hand'])][proof['seat']]
            row.update(player=proof['canonical_actor'],WHO_status='PROVEN',identity_proof=copy.deepcopy(proof))
        for record in records:
            hand=actors.canonical_hand(record['observed_hand'])
            for row in self.rows[hand_key(hand)].values():
                own,refs=forced_correction(hand,actor=row['player'],seat=row['seat'])
                row.update(forced=own,forced_sources=refs,START_STACK=corrected_start_stack(row['DISPLAYED_STACK'],own))
        for row in sitouts+no_sb_seats:
            _validation_require(row['seat'] in self.rows[hand_key(row)],'SAVED_VALIDATION_SEAT_OVERRIDE_UNBOUND')
            self.rows[hand_key(row)][row['seat']]=copy.deepcopy(row)

    def seat_rows(self,hand):
        return copy.deepcopy(list(self.rows[self.key(hand)].values()))

    def for_hand(self,hand):
        from parser_core.simple_saved_policy import SimpleSavedPolicyProvider
        return SimpleSavedPolicyProvider.for_hand(self,hand)


def _saved_validation_profile(counts):
    """Fail before any new search/acquisition; count actual validator execution."""
    forbidden={'solve','reconstruct_hand','reconstruct_production_hands','write_reconstruction',
               'saved_stack_observations','saved_sitout_observations'}
    def profile(frame,event,arg):
        if event!='call':return
        module=frame.f_globals.get('__name__','');name=frame.f_code.co_name
        if (module.startswith(('parser_core.poker_reconstruction','parser_core.simple_saved_policy')) and name in forbidden
                or module=='parser_core.poker_reconstruction.state_machine' and name=='apply'
                or module in {'parser_core.tiny_chat_client','parser_core.ram_first_stage'}):
            raise RuntimeError(f'SAVED_VALIDATION_FORBIDDEN_CALL:{module}.{name}')
        if module=='parser_core.reconstruction_validation' and name=='validate_reconstruction':
            counts['validate_reconstruction']+=1
    return profile


def validate_saved_hh(bundle_path,output,*,no_sb_review):
    """Official source-sealed fixed-trace validation. Never invokes reconstruction."""
    started=time.perf_counter()
    root=Path(__file__).resolve().parents[1]
    execution={str((root/name).resolve()):_validation_sha(root/name)
               for name in ('run_production.py','scripts/replay_saved_quality.py')}
    initial=_validation_bundle(bundle_path,no_sb_review);bundle,files,pinned,bundle_sha=initial
    output=Path(output).resolve();source=Path(bundle['native_source']).resolve()
    _validation_require(all(output!=p and output not in p.parents and p not in output.parents
        for p in [source,*files.values()]),'SAVED_VALIDATION_OUTPUT_OVERLAPS_INPUT')
    output.mkdir(parents=True,exist_ok=False)
    counts=Counter();previous=sys.getprofile()
    _validation_require(previous is None,'SAVED_VALIDATION_EXISTING_PROFILER')
    sys.setprofile(_saved_validation_profile(counts))
    try:
        from parser_core.simple_saved_policy import hand_key,LineageActorProvider,saved_nick_observations
        from parser_core.simple_seat_who import SeatIdentityLineage
        from parser_core.legitimate_no_sb import LegitimateNoSBSource
        from parser_core.hidden_loser_evidence import HiddenLoserSource
        from parser_core.reconstruction_validation import validate_reconstruction
        from parser_core.poker_reconstruction_layer import reconstruction_config,reconstruction_output_blockers
        from parser_core.full_hh_profile import reconstruction_settings
        from parser_core.hh_export import export_hand_histories
        records=read(files['logical']);traces=read(files['traces']);stacks=read(files['stacks'])
        seats=read(files['seats']);updates=read(files['who_updates']);reference=read(files['reference_hh'])
        sitouts=json.loads(files['sitout_targets'].read_text('utf-8'))
        for hh in read(files['sitout_histories']):
            sitouts.extend(p['simple_sitout_state'] for p in hh['players'] if p.get('simple_sitout_state'))
        no_sb_seats=read(files['no_sb_seats'])
        _validation_native_seals(source,files,pinned,[])
        n=bundle['expected_complete']
        _validation_require(type(n) is int and n>0 and len(records)==len(traces)==len(reference)==n,
                            'SAVED_VALIDATION_DENOMINATOR')
        segments=[r['segment_id'] for r in records]
        _validation_require(len(set(segments))==n and [r['segment_id'] for r in traces]==segments
            and [r['segment_id'] for r in reference]==segments and
            len({hand_key(r['observed_hand']) for r in records})==n and
            all(r['lifecycle']['status']=='COMPLETE' for r in records),'SAVED_VALIDATION_COMPLETE_ORDER')
        _validation_require(len(stacks)==bundle['expected_stack_rows'] and
            len({(s['segment_id'],s['player']) for s in stacks})==len(stacks) and
            all(s['segment_id'] in set(segments) for s in stacks),'SAVED_VALIDATION_STACK_SEGMENT')
        actors=_SavedValidationActors(read(files['who']),updates)
        bounds={hand_key(r):(r['hand_start'],r['hand_end']) for r in seats}
        state_path=source/'seat_forced_summary.json'
        states=json.loads(state_path.read_text('utf-8'))['state_receipts']
        native_actors=LineageActorProvider(SeatIdentityLineage(saved_nick_observations(source),states),
            bounds,[r['observed_hand'] for r in records],read(files['associations']))
        no_sb=LegitimateNoSBSource(source,native_actors,files['review'])
        hidden=HiddenLoserSource(source,native_actors,visual_source=files['visual'])
        facts=_SavedValidationFacts(records,seats,actors,no_sb,source,updates,sitouts,no_sb_seats)
        # Only the final candidate views reach consumers. Intermediate seat
        # audit locators remain sealed metadata even when a later view replaces
        # their state; do not mistake those locators for reopened native inputs.
        _validation_native_seals(source,files,pinned,
            [row for record in records for row in facts.seat_rows(record['observed_hand'])])
        config=reconstruction_settings(reconstruction_config());results=[];statuses=[];timings=[]
        raw_trace_parity=True
        for record,original in zip(records,traces):
            hand=record['observed_hand'];key=hand_key(hand)
            expected=[s['simple_binding_receipt'] for s in stacks if s['segment_id']==record['segment_id']
                      and s.get('simple_binding_receipt')]
            actual=facts.for_hand(hand)
            identities={p['actor']:p for p in actors.for_hand(hand)}
            _validation_require(all(r['actor'] in identities and r['seat']==identities[r['actor']]['seat'] and
                hand_key(r)==key and next((row.get('WHO_status') for row in facts.seat_rows(hand)
                    if row['seat']==r['seat'] and row['player']==r['actor']),None)=='PROVEN' for r in actual),
                f'SAVED_VALIDATION_STACK_WHO_BINDING:{key}')
            _validation_require(sorted(actual,key=lambda r:r['actor'])==sorted(expected,key=lambda r:r['actor']),
                                f'SAVED_VALIDATION_START_STACK_PARITY:{key}')
            result=copy.deepcopy(original)
            result['derived_final_validation']=validate_reconstruction(hand,result,
                nickname_canonicalization=result.get('nickname_canonicalization',{}),
                hidden_card_source=hidden,no_sb_source=no_sb)
            reasons=reconstruction_output_blockers(record,result,config)
            result.update(output_blockers=reasons,output_eligible=not reasons)
            if not result['derived_final_validation'].get('accepted') or reasons:
                write(output/'rejected_reconstruction.jsonl',[result])
            _validation_require(result['derived_final_validation'].get('accepted') and not reasons,
                                f'SAVED_VALIDATION_NOT_READY:{key}:{reasons}')
            difference=_validation_difference(original,result,_TRACE_VALIDATION_TIMINGS)
            if difference is not None:
                write(output/'trace_mismatch_old.jsonl',[original])
                write(output/'trace_mismatch_new.jsonl',[result])
                raise ValueError(f'SAVED_VALIDATION_TRACE_AUTHORITY_PARITY:{key}:{difference}')
            raw_trace_parity=raw_trace_parity and result==original
            timings.extend(dict(segment_id=record['segment_id'],path='.'.join(path),
                old=_validation_value(original,path),new=_validation_value(result,path))
                for path in _TRACE_VALIDATION_TIMINGS)
            results.append(result)
            validation=result['derived_final_validation']['validation']
            _validation_require(validation['math']['pass'],f'SAVED_VALIDATION_MATH:{key}')
            statuses.append(dict(table=hand['table_id'],hand=hand['hand_id'],COMPLETE=True,READY=True,MATH='PASS'))
        write(output/'poker_reconstruction.jsonl',results)
        write(output/'validation_timing_deltas.jsonl',timings)
        write(output/'who_projection_audit.jsonl',[row for rows in actors.projections.values() for row in rows])
        write(output/'logical_hand_analysis.jsonl',records)
        write(output/'complete_hand_analysis.jsonl',records)
        shutil.copy2(files['stacks'],output/'stack_start_stack.jsonl')
        summary=export_hand_histories(output,records,simple_provider=facts,actor_provider=actors)
        exported=read(output/'hand_histories.jsonl')
        _validation_require(len(exported)==len(reference),'SAVED_VALIDATION_HH_DENOMINATOR')
        for old,new in zip(reference,exported):
            difference=_validation_difference(old,new,_HH_VALIDATION_TIMINGS)
            _validation_require(difference is None,
                f'SAVED_VALIDATION_HH_SEMANTIC_PARITY:{old["segment_id"]}:{difference}')
        _validation_require((output/'hand_histories.txt').read_bytes()==files['reference_text'].read_bytes(),
                            'SAVED_VALIDATION_HH_BYTE_PARITY')
        _validation_require(summary['certified_hh']==summary['ready_hh_text_count']==n and
            not summary['blockers'] and not summary['ready_hh_text_blockers'],'SAVED_VALIDATION_HH_GATE')
        _validation_require(counts['validate_reconstruction']==n,'SAVED_VALIDATION_VALIDATOR_CALL_COUNT')
        # Recheck all sealed inputs before issuing any PASS receipt.
        _validation_finish_seals(initial,bundle_path,no_sb_review,execution)
        write(output/'hand_status.jsonl',statuses)
        receipt=dict(schema='poker.saved-hh-validation.receipt.v1',verdict='PASS',
            route='run_production.py --validate-saved-hh',scope='SEALED_RETAINED_TRACES_AND_FACTS_CURRENT_VALIDATORS_EXPORTER',
            COMPLETE=n,READY=n,MATH_PASS=n,HH=n,HH_SEMANTIC_PARITY=True,HH_BYTE_PARITY=True,
            HH_BYTE_PARITY_FORMAT='hand_histories.txt',RAW_TRACE_JSON_PARITY=raw_trace_parity,
            RAW_HH_JSON_PARITY=exported==reference,
            semantic_parity_excluded_stopwatches=dict(trace=['.'.join(p) for p in _TRACE_VALIDATION_TIMINGS],
                hh=['.'.join(p) for p in _HH_VALIDATION_TIMINGS]),
            START_STACK_AUTHORITY_PARITY=True,DEAL_AUTHORITY_PARITY=True,ROW_PLAYER_CARD_ACTION_IDENTITY_PARITY=True,
            NEW_UNKNOWN=0,NEW_NOT_READY=0,new_solver_calls=0,new_video_decode_calls=0,new_ocr_calls=0,
            actual_calls=dict(counts),source_seals=len(pinned),bundle_sha256=bundle_sha,
            additional_native_parent_sha256=bundle.get('native_revalidation_inputs',{}),
            additional_parent_historical_seal=bundle.get('native_parent_history'),
            no_sb_review_path=str(files['review']),no_sb_review_sha256=_validation_sha(files['review']),
            hh_sha256=_validation_sha(output/'hand_histories.txt'),
            execution_source_sha256={str(Path(path).relative_to(root)):digest for path,digest in execution.items()},
            wall_seconds=time.perf_counter()-started)
        (output/'saved_validation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n','utf-8')
        print(json.dumps(receipt),flush=True)
        return receipt
    except BaseException as error:
        (output/'saved_validation_failure.json').write_text(json.dumps(dict(verdict='FAIL',
            error_type=type(error).__name__,reason=str(error),actual_calls=dict(counts)),indent=2)+'\n','utf-8')
        raise
    finally:
        sys.setprofile(previous)


_VISUAL_FIELDS = (
    'ocr_event_type','cards','raw_vl_response','evidence_status','continuity_status',
    'evidence_context_status','visual_crop_path','visual_crop_sha256','visual_crop_type',
    'identity_status','card_evidence_id','card_row_key','card_row_id','card_crop_pixel_sha256',
    'trigger_row_key','recognition_mode','raw_pixel_response','development_only','rank_mode',
    'rank_ncc_config_sha256','board_sha','board_evidence_id','win_best_five_proof',
    'win_best_five_conflict','unique_card_set_proof','unique_card_set_conflict',
)


def ordinary_replay_rows(original, *, reset_visual=False):
    """Remove generated rows/annotations, never their native source observations."""
    rows=[]
    for source in original:
        if source.get('source_kind')=='SIDECAR_HEADER_RECEIPT':
            continue
        if (source.get('origin')=='FORCED_SNAPSHOT' and
                str(source.get('stable_row_id','')).startswith('CANONICAL_FORCED:') and
                source.get('stable_row_id')==source.get('canonical_event_id')):
            continue
        row=copy.deepcopy(source)
        if row.get('canonical_provenance'):
            for field in ('seat','hand_id','player','amount','action_type','binding_status'):
                row.pop(field,None)
        for field in ('canonical_event_id','canonical_provenance','canonical_stream_version',
                      'canonical_conflict','canonical_duplicate_of'):
            row.pop(field,None)
        if reset_visual:
            projected = any(row.get(k) for k in ('card_row_key','card_evidence_id','board_evidence_id',
                                                 'visual_crop_type','visual_crop_sha256'))
            for field in _VISUAL_FIELDS:
                row.pop(field,None)
            if projected:
                for field in ('hand_id','street','player'):
                    row.pop(field,None)
                candidate=normalize([row['raw_ocr_text']])
                row['event_type']=candidate.event_type
                if row.get('status') not in {'ERROR','UNRESOLVED'} and row.get('origin')!='UNRESOLVED_IDENTITY':
                    row['status']='UNKNOWN' if candidate.event_type=='UNKNOWN_EVENT' else 'RAW_UNREVIEWED'
        rows.append(row)
    return rows


def refresh_saved_card_readings(source, output):
    """Re-run current PIXEL_ONLY producer on exact saved native CHAT crops."""
    from parser_core.visual_tile_queue import VisualTileQueue, card_tile_candidate
    root=Path(__file__).resolve().parents[1]
    source_path=source/'visual_evidence.jsonl'
    source_bytes=source_path.read_bytes()
    physical_path=source/'physical_rows.jsonl'
    physical_bytes=physical_path.read_bytes()
    saved_records=read(source_path)
    parents=[r for r in saved_records if r.get('row_key')]
    diagnostics=[r for r in saved_records if not r.get('row_key')]
    missing_fields={'type','street','player','table_id','table_session_id','frame_id','hand_id',
                    'trigger_row_key','crop_type','crop_path','raw_vl_response','status','reason'}
    if any(set(r)-missing_fields or r.get('reason')!='NO_VALID_ADJACENT_CARD_TILE' or
           r.get('status')!='UNRESOLVED' or r.get('crop_type')!='CHAT_ROW_CARDS' or
           not r.get('trigger_row_key') or r.get('crop_path') is not None or
           r.get('raw_vl_response') is not None for r in diagnostics):
        raise ValueError('SAVED_CARD_UNKNOWN_RECORD_WITHOUT_SOURCE_KEY')
    physical_rows=read(physical_path)
    physical={r['row_key']:r for r in physical_rows}
    if len(physical)!=len(physical_rows) or len({r['row_key'] for r in parents})!=len(parents):
        raise ValueError('SAVED_CARD_SOURCE_IDENTITY_NOT_UNIQUE')
    manifest=json.loads((root/'production_manifest.json').read_text('utf-8'))
    paths=[root/name for name in ['production_manifest.json',*manifest['modules'],*manifest['configs'],
                                  'models/board_glyphs_v1.json']]
    resolved_dependencies=[]
    for dependency in manifest['data_dependencies']:
        config=json.loads((root/dependency['config']).read_text('utf-8'))
        actual=Path(config[dependency['key']]).resolve()
        if hashlib.sha256(actual.read_bytes()).hexdigest()!=dependency['sha256']:
            raise ValueError('SAVED_CARD_DATA_DEPENDENCY_HASH_CONFLICT')
        paths.append(actual)
        resolved_dependencies.append(dict(config=dependency['config'],key=dependency['key'],
            resolved_path=str(actual),sha256=dependency['sha256']))
    card_config=json.loads((root/'configs/card_hybrid_local.json').read_text('utf-8'))
    paths.extend(Path(row['tile_png']) for row in read(Path(card_config['train_manifest'])))
    paths=list(dict.fromkeys(p.resolve() for p in paths))
    before={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    jobs=[]
    fields=('table_id','table_session_id','frame_id','source_span_index','source_pts',
            'time_base_num','time_base_den','row_pixel_sha256','row_id','epoch')
    for parent in parents:
        native=physical[parent['row_key']]
        if (parent.get('crop_type')!='CHAT_ROW_CARDS' or parent.get('recognition_mode')!='PIXEL_ONLY'
                or any(parent.get(k)!=native.get(k) for k in fields)):
            raise ValueError('SAVED_CARD_PHYSICAL_CONTEXT_CONFLICT')
        data=base64.b64decode(parent['crop_png_base64'],validate=True)
        if (hashlib.sha256(data).hexdigest()!=parent['crop_sha256'] or
                parent['raw_pixel_response']['source_sha256']!=parent['crop_sha256']):
            raise ValueError('SAVED_CARD_PNG_HASH_CONFLICT')
        pixels=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED)
        native_pixels=base64.b64decode(native['row_pixels_base64'],validate=True)
        if (pixels is None or pixels.ndim!=2 or pixels.dtype!=np.uint8 or
                list(pixels.shape)!=native['pixel_shape'] or pixels.tobytes()!=native_pixels or
                hashlib.sha256(native_pixels).hexdigest()!=native['row_pixel_sha256']):
            raise ValueError('SAVED_CARD_PIXEL_SOURCE_CONFLICT')
        ok,encoded=cv2.imencode('.png',pixels)
        if not ok or encoded.tobytes()!=data or not card_tile_candidate(pixels):
            raise ValueError('SAVED_CARD_NATIVE_PRODUCER_INPUT_CONFLICT')
        row=SimpleNamespace(table_id=parent['table_id'],session_id=parent['table_session_id'],
            frame_id=parent['frame_id'],source_span_index=parent['source_span_index'],
            video_time=parent['video_time'],source_pts=parent['source_pts'],
            time_base_num=parent['time_base_num'],time_base_den=parent['time_base_den'],
            pixels=pixels,pixel_sha256=parent['row_pixel_sha256'],
            stable_row_id=parent['row_id'],epoch=parent['epoch'])
        if (row_key(row)!=parent['row_key'] or
                row.video_time!=row.source_pts*row.time_base_num/row.time_base_den):
            raise ValueError('SAVED_CARD_SOURCE_CLOCK_CONFLICT')
        jobs.append(row)
    def no_vl(_):
        raise ValueError('SAVED_CARD_REPLAY_REQUIRES_PIXEL_ONLY')
    queue=VisualTileQueue(no_vl,output,workers=1,card_workers=1,image_transport='ram',
        card_hybrid_config=root/'configs/card_hybrid_local.json',
        suit_backup_config=root/'configs/suit_backup_local.json',
        rank_ncc_config=root/'configs/rank_ncc_production.json')
    try:
        for row in jobs:
            if not queue.submit(row):raise ValueError('SAVED_CARD_NATIVE_ADMISSION_CONFLICT')
    finally:
        queue.finish()
    if queue.errors or set(queue.results)!={r['row_key'] for r in parents}:
        raise ValueError('SAVED_CARD_REPLAY_PRODUCER_FAILED: '+repr(queue.errors))
    parent_sha=hashlib.sha256(source_bytes).hexdigest()
    for parent in parents:
        fresh=queue.results[parent['row_key']]
        if any(fresh.get(k)!=parent.get(k) for k in (*fields,'row_key','video_time','crop_sha256','crop_png_base64')):
            raise ValueError('SAVED_CARD_REPLAY_IDENTITY_CHANGED')
        fresh['saved_replay_parent']=dict(file=str(source_path),file_sha256=parent_sha,
            row_key=parent['row_key'],crop_sha256=parent['crop_sha256'])
    after={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if before!=after or source_bytes!=source_path.read_bytes() or physical_bytes!=physical_path.read_bytes():
        raise ValueError('SAVED_CARD_REPLAY_SOURCE_OR_RUNTIME_CHANGED')
    (output/'visual_evidence_source.jsonl').write_bytes(source_bytes)
    queue.write_evidence(output/'visual_evidence.jsonl')
    (output/'saved_card_replay_receipt.json').write_text(json.dumps(dict(
        rule='CURRENT_PIXEL_PRODUCER_FROM_EXACT_SAVED_CHAT_CROPS_V1',
        source_visual_sha256=parent_sha,physical_rows_sha256=hashlib.sha256(physical_bytes).hexdigest(),
        rows=len(parents),source_missing_diagnostics=len(diagnostics),
        diagnostics_policy='OLD_DIAGNOSTICS_PRESERVED_IN_SOURCE_FILE; REGENERATE_FROM_CURRENT_TIMELINE',
        physical_identity_equal=True,source_unchanged=True,
        runtime_sha256=before,resolved_data_dependencies=resolved_dependencies,
        runtime_unchanged=True,worker_errors=queue.errors,
        authority='FRESH_COMPONENT_READINGS_ONLY; CURRENT_CONTEXT_AND_HH_GATES_REQUIRED'),indent=2),'utf-8')
    return {r['row_key']:r for r in read(output/'visual_evidence.jsonl')}


def replay_nickname_binding(source, output, *, full_profile=False, return_binder=False):
    """Run the production admission/binder over immutable saved Nick responses."""
    original=json.loads((source/'seat_forced_summary.json').read_text('utf-8'))
    deals={}
    for r in read(source/'start_stack_live_receipts.jsonl'):
        if r.get('status')=='DEAL_COMPLETE':
            key=(r['table_id'],r['session_id'],r['hand_id'])
            if full_profile and key in deals and deals[key]!=r:
                raise ValueError('CONFLICTING_DEAL_COMPLETE_SOURCE: '+str(key))
            deals[key]=r
    binder=SeatForcedBinding.__new__(SeatForcedBinding)
    flags=json.loads((Path(__file__).resolve().parents[1]/'configs/dealing_start_stack.json').read_text('utf-8'))
    binder.sidecar=SimpleNamespace(lock=threading.RLock(),flags=flags,
        triggers=read(source/'dealing_cards_triggers.jsonl'),
        windows={k:SimpleNamespace(deal=r) for k,r in deals.items()})
    binder.lock=threading.RLock();binder.jobs=PendingSourceJobs();binder.counts=Counter();binder.prefix_candidates=Counter()
    binder.nick_runs={};binder.nick_labels=defaultdict(list);binder.machine=SeatEpochMachine()
    binder.forced={};binder.snapshots=defaultdict(dict);binder.blockers={};binder.state_events=[]
    binder.nick_candidates=defaultdict(dict);binder.seen_deals=set(deals)
    binder.candidate_sizes={};binder.candidate_rows=defaultdict(dict)
    binder.pending_snapshots={};binder.prefix_reuse={};binder.observed_frames={}
    captured_ends={}
    for entry in original.get('observed_hand_ends',[]):
        binder.observed_frames[(entry['table'],entry['session'],entry['hand_id'])]=entry['source_frame']
    if not binder.observed_frames and (source/'start_stack_capture.jsonl').is_file():
        with (source/'start_stack_capture.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                entry=json.loads(line);route=(entry['table_id'],entry['session_id'])
                captured_ends[route]=max(captured_ends.get(route,-1),entry['source_frame'])
    for key in deals:
        if key not in binder.observed_frames and key[:2] in captured_ends:
            binder.observed_frames[key]=captured_ends[key[:2]]
    early_keys={(s['table'],s['session'],s['hand_id']) for s in original['forced_snapshots']
                if s.get('early_prefix_reused')}
    binder.pending_snapshots={k:[] for k in early_keys}
    binder.roster_prefix_binding=full_profile
    binder.roster_forced_rows=defaultdict(dict);binder.roster_nick_rows=defaultdict(list)
    inputs=read(source/'seat_forced_inputs.jsonl')
    binder.snapshot_sizes=Counter((r['table_id'],r['session_id'],r['hand_id']) for r in inputs if r['kind']=='CHAT')
    binder.snapshot_occupied={k:len(r['occupied_seats']) for k,r in deals.items()}
    prepared=[]
    for receipt in inputs:
        if receipt['kind'] not in {'CHAT','CHAT_CANDIDATE','NICK'}:continue
        key=(receipt['table_id'],receipt['session_id'],receipt['hand_id'])
        pixels=cv2.imdecode(np.frombuffer(base64.b64decode(receipt['png_base64']),np.uint8),cv2.IMREAD_UNCHANGED)
        assert pixels is not None and hashlib.sha256(pixels.tobytes()).hexdigest()==receipt['pixel_sha256']
        index=int(receipt['ocr_response']['item_id'].split(':')[-1])
        row=binder.row(key[0],key[1],receipt['source_frame'],receipt['source_pts'],receipt['time_base'],
            index,receipt['roi'][1] if receipt['kind']=='NICK' else receipt['row_bounds'][0],pixels,
            'NICK_ONLY' if receipt['kind']=='NICK' else
            'FORCED_PREFIX_CANDIDATE' if receipt['kind']=='CHAT_CANDIDATE' else 'FORCED_SNAPSHOT')
        meta={k:v for k,v in receipt.items() if k not in {'png_base64','ocr_response','pixel_sha256'}}
        binder.jobs[row_key(row)]=(key,meta)
        if key not in binder.observed_frames:
            binder.observed_frames[key]=max(r['source_frame'] for r in inputs if
                (r['table_id'],r['session_id'],r['hand_id'])==key)
        prepared.append((row,receipt))
    with (output/'nickname_admission_replay.jsonl').open('w',encoding='utf-8') as log:
        binder.log=log
        # Register all pending observations before replaying callback order.
        # A faster later callback cannot lock the name ahead of an earlier source.
        for row,receipt in prepared:
            binder.completed(row,copy.deepcopy(receipt['ocr_response']))
    old_snapshots={(s['table'],s['session'],s['hand_id']):s for s in original['forced_snapshots']}
    for key,snapshot in old_snapshots.items():
        if not snapshot.get('early_prefix_reused'):continue
        from parser_core.seat_forced_binding import parse_forced_snapshot
        candidates=list(binder.candidate_rows[key].values())
        reproduced,reason=parse_forced_snapshot(candidates,key[2],len(deals[key]['occupied_seats']),flags['stack_confidence'])
        if reproduced is None:raise ValueError('SAVED_EARLY_PREFIX_NOT_REPRODUCIBLE: '+str(reason))
        original_row=candidates[0]
        for post in reproduced['posts']:
            post.update(source_frame=original_row['source_frame'],source_pts=original_row['source_pts'],
                time_base=original_row['time_base'])
        reproduced['early_prefix_reused']=True;binder.forced[key]=reproduced
        binder.pending_snapshots.pop(key)
    binder._resolve_nick_labels()
    post_signature=lambda p:(p['actor'],p['role'],p['amount'],tuple(p['source_refs']),p['source_frame'])
    assert set(binder.forced)==set(old_snapshots)
    assert all(Counter(post_signature(p) for p in binder.forced[k]['posts'])==
               Counter(post_signature(p) for p in old_snapshots[k]['posts']) for k in binder.forced)
    # Reuse exactly the exported occupancy/state receipts, not new inferred
    # occupancy from nickname responses. The production epoch algorithm runs.
    machine=SeatEpochMachine()
    for t in read(source/'dealing_cards_triggers.jsonl'):
        machine.hand(t['table_id'],t['session_id'],t['hand_id'],t['source_frame'])
    for e in original['state_receipts']:
        machine.observe(e['table'],e['session'],e['seat'],e['kind'],e['frame'],actor=e.get('actor'),
            hand_id=e.get('hand_id'),evidence_status=e['evidence_status'],source_ref=e['source_ref'],source_index=e['source_index'])
    machine.build();binder.machine=machine
    assert machine.audit==original['state_receipts']
    bound=[];proofs=[];blockers=[];roster_proofs=[]
    for key,snapshot in binder.forced.items():
        if key not in deals:continue
        # Its physical post signature was reproduced above. Keep the immutable
        # source snapshot schema; replay bookkeeping is not raw provenance.
        snapshot=old_snapshots[key]
        roster_inputs=None;roster_proof=None
        if full_profile:
            from parser_core.nickname_roster_binding import prove_roster_binding, binding_receipt
            roster_inputs=dict(context=list(key),snapshot=dict(snapshot,table=key[0],session=key[1],hand_id=key[2]),
                deal=deals[key],forced_rows=list(binder.roster_forced_rows[key].values()),
                nick_rows=binder.roster_nick_rows[key],
                hand_triggers=[t for t in read(source/'dealing_cards_triggers.jsonl') if (t['table_id'],t['session_id'])==key[:2]],
                state_receipts=[e for e in machine.audit if (e['table'],e['session'])==key[:2]],
                unresolved_events=[e for e in original['unresolved_state_events'] if (e['table'],e['session'])==key[:2]],
                capture_frames=flags['capture_frames'],confidence=flags['stack_confidence'])
            roster_proof=prove_roster_binding(roster_inputs)
            roster_proofs.append(dict(table=key[0],session=key[1],hand_id=key[2],
                **{k:v for k,v in roster_proof.items() if k not in {'source_inputs','label_evidence'}}))
        for actor in snapshot['actors']:
            posts=[p for p in snapshot['posts'] if p['actor']==actor]
            receipt,reason=binder.historical_forced_binding(key,actor,posts,original['unresolved_state_events'])
            if receipt is None and roster_proof and roster_proof['status']=='PROVEN':
                receipt=binding_receipt(roster_proof,actor,posts)
            if receipt is None and roster_inputs:
                from parser_core.nickname_actor_binding import prove_actor_binding, binding_receipt as actor_receipt
                receipt=actor_receipt(prove_actor_binding(roster_inputs,actor),actor,posts)
            if receipt:
                proofs.append(receipt)
                metadata={'binding_rule':receipt['binding_rule']} if (
                    'roster_proof' in receipt or 'actor_prefix_proof' in receipt) else {}
                bound.extend(dict(p,table=key[0],session=key[1],hand_id=key[2],seat=receipt['seat'],epoch=receipt['epoch'],
                                  **metadata) for p in posts)
            else:blockers.append(dict(table=key[0],session=key[1],hand_id=key[2],actor=actor,blocker=reason))
    summary=copy.deepcopy(original)
    summary.update(bound_posts=bound,historical_binding_receipts=proofs,historical_binding_blockers=blockers,
                   forced_bets_bound=len(bound),saved_binding_replay=True,
                   observed_hand_ends=[dict(table=k[0],session=k[1],hand_id=k[2],source_frame=binder._hand_end_frame(k))
                                       for k in binder.observed_frames if k in deals],
                   original_summary_sha256=hashlib.sha256((source/'seat_forced_summary.json').read_bytes()).hexdigest())
    if full_profile:summary['roster_binding_proofs']=roster_proofs
    (output/'seat_forced_summary_source.json').write_bytes((source/'seat_forced_summary.json').read_bytes())
    (output/'seat_forced_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    signature=lambda p:(p['table'],p['session'],p['hand_id'],p['actor'],p['seat'],p['epoch'],p['role'],p['amount'],tuple(p['source_refs']))
    assert {signature(p) for p in original['bound_posts']}.issubset({signature(p) for p in bound})
    receipt=dict(bound_before=len(original['bound_posts']),bound_after=len(bound),occupancy_receipt_parity=True,
                saved_callback_order_preserved=True,physical_forced_snapshot_parity=True,full_profile=full_profile)
    return (receipt,binder) if return_binder else receipt


def replay_stack_frames(source,output,binder):
    """Rebuild current single-frame facts from retained native OCR, without OCR."""
    from parser_core.start_stack_live import first_visible_stack_triple
    from parser_core.start_stack_visibility import first_visible_proof
    from parser_core.first_stack_ocr import direct_numeric_policy
    live=json.loads((source/'start_stack_live_summary.json').read_text('utf-8'))
    bounds={(r['table_id'],r['session_id'],r['hand_id']):r for r in live['hands']}
    readings=defaultdict(dict)
    for row in read(source/'start_stack_live_receipts.jsonl'):
        if row.get('stage')!='STACK_OCR' or row.get('status')=='SOURCE_CONTEXT_UNKNOWN':continue
        key=(row['table_id'],row['session_id'],row['hand_id'],row['seat'])
        frame=row['source_frame']
        if frame in readings[key] and readings[key][frame]!=row:
            raise ValueError('SAVED_STACK_DUPLICATE_SOURCE_CONFLICT')
        readings[key][frame]=row
    summary=json.loads((output/'seat_forced_summary.json').read_text('utf-8'))
    states={};policy=direct_numeric_policy()
    for key,window in bounds.items():
        if key not in binder.sidecar.windows:continue
        w=binder.sidecar.windows[key];w.closed=window.get('first_stack_changing_observation_frame');w.results={}
        start=min(t['source_frame'] for t in binder.sidecar.triggers if
            (t['table_id'],t['session_id'],t['hand_id'])==key)
        following=[t['source_frame'] for t in binder.sidecar.triggers if
            (t['table_id'],t['session_id'])==key[:2] and t['hand_id']!=key[2] and t['source_frame']>start]
        boundary=min(following,default=None);deal=window.get('deal_complete_frame')
        state=dict(numeric_triples={},canonical_attempts={},source_readings={})
        states[key]=state
        if type(deal) is not int or (boundary is not None and deal>=boundary):
            w.deal=None;continue
        for seat in w.deal['occupied_seats']:
            rows=readings[(*key,seat)];state['source_readings'][seat]=rows
            attempt=first_visible_stack_triple(rows,.85,final=True,start_frame=deal+1,
                allow_scaled=True,numeric_policy=policy,single_frame=True)
            if attempt is None:continue
            proof=first_visible_proof(rows,attempt['frames'],start_frame=deal+1,confidence=.85,
                numeric_policy=policy,qualified_single=True,clean_window_end_frame=w.closed)
            attempt['first_visible_proof']=proof
            if proof['status']!='PROVEN':attempt.update(status='UNKNOWN',blocker=proof['reason'])
            if boundary is not None and attempt['frames'][-1]>=boundary:
                attempt.update(status='UNKNOWN',blocker='LATE_HEADER_SOURCE_CONTRADICTION')
            state['canonical_attempts'][seat]=attempt
            if attempt['status']=='PASS':state['numeric_triples'][seat]=attempt
    shutil.copy2(source/'start_stack_facts.jsonl',output/'start_stack_facts_source.jsonl')
    shutil.copy2(source/'start_stack_readings.jsonl',output/'start_stack_readings.jsonl')
    binder.output=output;binder.closed=False;binder.state_events=[]
    binder.seen_prefix_candidates=set(binder.candidate_rows)
    binder.stack_live=SimpleNamespace(states=states)
    binder.ocr=SimpleNamespace(finish=lambda:None,errors=[])
    # Immutable source audit already includes resolved seat events. Replay it
    # once, retaining unresolved facts, then use the exact live fact producer.
    with (output/'start_stack_readings.jsonl').open('a',encoding='utf-8') as log:
        binder.sidecar.logs={'start_stack_readings':log}
        binder.finish(retained_unresolved_events=summary['unresolved_state_events'])
    facts=read(output/'start_stack_facts.jsonl')
    receipt=dict(rule='CURRENT_SINGLE_STACK_FRAME_FROM_RETAINED_NATIVE_OCR_V1',facts=len(facts),
        numeric_frames_required=1,confidence=.85,new_ocr=False,new_video=False,
        selected=sum(r['displayed_stack_status']=='VALID_FIXED_SEAT_ROI' for r in facts),
        source_facts_sha256=hashlib.sha256((source/'start_stack_facts.jsonl').read_bytes()).hexdigest(),
        live_summary_sha256=hashlib.sha256((source/'start_stack_live_summary.json').read_bytes()).hexdigest())
    (output/'saved_stack_frame_replay_receipt.json').write_text(json.dumps(receipt,indent=2),'utf-8')
    return receipt


def metrics(records, reconstruction, timeline, math, snapshots):
    complete = [r for r in records if r.get('lifecycle', {}).get('status') == 'COMPLETE']
    observed={(table_key(r['table_id']),r['session_id'],r['hand_id'],r['player'],
               r['canonical_provenance']['post']['role'],r['amount'],tuple(r['canonical_provenance']['post']['source_refs']))
              for r in timeline if r.get('canonical_event_id') and not r.get('canonical_duplicate_of') and not r.get('canonical_conflict')}
    full_prefix={(s['table'],s['session'],s['hand_id']) for s in snapshots if s.get('forced_snapshot_complete') and
                 all((s['table'],s['session'],s['hand_id'],p['actor'],p['role'],p['amount'],tuple(p['source_refs'])) in observed for p in s['posts'])}
    return dict(COMPLETE=len(complete), validation=dict(Counter(r['validation']['status'] for r in complete)),
        reconstruction=dict(Counter(r['status'] for r in reconstruction)),
        VALID=sum(r['fully_validated'] for r in complete), CERTIFIED=sum(r['certified_hh'] for r in complete),
        errors=dict(Counter(i['code'] for r in complete for i in r['validation']['issues'])),
        raw_UNKNOWN=sum(r['event_type']=='UNKNOWN_EVENT' for r in timeline),
        semantic_UNKNOWN=sum(m['action']=='UNKNOWN_EVENT' for r in complete for m in r['observed_hand']['messages']),
        forced=dict(Counter(r.get('action_type') for r in timeline if r.get('canonical_event_id'))),
        complete_forced_prefix=sum(context_key(r['observed_hand']) in full_prefix for r in complete),
        both_blinds_observed=sum({'SMALL_BLIND','BIG_BLIND'}.issubset({m['action'] for m in r['observed_hand']['messages']}) for r in complete),
        stack_math_statuses=dict(Counter(r['status'] for r in math)),
        stack_snapshot_candidates=sum(r.get('snapshot_arithmetic_status')=='PASS' or r['status']=='PASS' for r in math),
        stack_certified=sum(bool(r.get('start_stack_certified')) for r in math),
        stack_candidates_joined=sum(len(r.get('joined_stack_snapshot_candidates',{})) for r in reconstruction),
        stack_joined_into_solver=sum(len(r.get('joined_start_stacks',{})) for r in reconstruction))


def main(source, output, *, player_database=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError('Replay output must be separate from immutable source run')
    output.mkdir(parents=True, exist_ok=False)
    matrix_path=source/'full_hh_production_matrix.json'
    full_profile=matrix_path.exists() and json.loads(matrix_path.read_text('utf-8')).get('profile')=='FULL_HH_PRODUCTION'
    names=['ordered_timeline.jsonl','raw_ocr_results.jsonl','physical_rows.jsonl',
        'complete_hand_analysis.jsonl','logical_hand_analysis.jsonl','poker_reconstruction.jsonl',
        'start_stack_math.jsonl','start_stack_facts.jsonl','seat_forced_summary.json',
        'canonical_forced_events.jsonl','canonical_join_summary.json','hand_header_ocr.jsonl','hand_state_receipts.jsonl']
    names+=['seat_forced_inputs.jsonl','start_stack_live_receipts.jsonl','dealing_cards_triggers.jsonl']
    full_files=['full_hh_production_matrix.json','run_metadata.json','visual_evidence.jsonl'] if full_profile else []
    if full_profile and (source/'hand_header_admission.jsonl').exists():
        full_files.append('hand_header_admission.jsonl')
    names+=full_files
    if full_profile:names+=[stack_timeline_source_name(source)]
    if full_profile:names+=['start_stack_capture.jsonl','physical_chat_admission.jsonl',
                           'start_stack_live_summary.json','start_stack_readings.jsonl']
    hashes={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names}
    for name in ['start_stack_facts.jsonl','seat_forced_summary.json','canonical_forced_events.jsonl',
                 'canonical_join_summary.json','hand_header_ocr.jsonl','hand_state_receipts.jsonl',
                 'seat_forced_inputs.jsonl','start_stack_live_receipts.jsonl','dealing_cards_triggers.jsonl']:
        shutil.copy2(source/name,output/name)
    for name in full_files+(['physical_rows.jsonl'] if full_profile else []):
        if name!='visual_evidence.jsonl':shutil.copy2(source/name,output/name)
    binding_result=replay_nickname_binding(source,output,full_profile=full_profile,return_binder=full_profile)
    binding_replay,binder=binding_result if full_profile else (binding_result,None)
    if full_profile:
        shutil.copy2(source/'start_stack_live_summary.json',output/'start_stack_live_summary_source.json')
        replay_stack_frames(source,output,binder)
    original=read(source/'ordered_timeline.jsonl')
    # Rejoin derived canonical events from the same immutable ordinary rows.
    # Original canonical rows/report remain available in the source run.
    ordinary=ordinary_replay_rows(original,reset_visual=full_profile)
    timeline=copy.deepcopy(ordinary)
    raw=read(source/'raw_ocr_results.jsonl');ocr={r['row_id']:r['raw_ocr_response']['status'] for r in raw}
    for row in timeline:
        if row.get('canonical_stream_version') or not full_profile and row['event_type']=='BOARD':
            continue
        c,service=normalize_with_service([row['raw_ocr_text']]);row['event_type']=c.event_type
        row['normalized_text']=service.normalized_text
        if service.changes:row['service_normalizations']=list(service.changes)
        else:row.pop('service_normalizations',None)
        row['notes_error']=c.reason or ''
        if row['status']=='UNKNOWN' and c.event_type!='UNKNOWN_EVENT':row['status']='RAW_UNREVIEWED'
    for table in sorted({r['table_id'] for r in timeline}):
        rr=[r for r in timeline if r['table_id']==table and not r.get('canonical_stream_version')]
        facts=[RowFact(r['stable_row_id'],r['session_id'],str(r['epoch']),r['frame_id'],r['row_index'],
                       r['raw_ocr_text'],None,None,r['origin']) for r in rr]
        by_id={r['stable_row_id']:r for r in rr}
        for i,m in enumerate(group_rows(facts),1):
            source_group=[by_id[fact.row_id] for fact in m.rows]
            candidate,service=normalize_with_service([r.text for r in m.rows],
                apply_service=physical_service_group(source_group))
            for fact in m.rows:
                by_id[fact.row_id].update(message_id=f'{table}:M{i:06d}',message_event_type=candidate.event_type,
                                         message_status=m.completeness,message_reason=m.reason or '')
                if service.changes:by_id[fact.row_id]['message_service_normalizations']=list(service.changes)
                else:by_id[fact.row_id].pop('message_service_normalizations',None)
    identity_fields=['stable_row_id','table_id','session_id','frame_id','row_index','raw_ocr_text','row_pixel_sha256',
                     'source_pts','time_base','video_time','epoch']
    assert [[r.get(k) for k in identity_fields] for r in ordinary]==[[r.get(k) for k in identity_fields] for r in timeline]
    timeline,_=canonical_join(output,timeline)
    ordinary_ids={r['stable_row_id'] for r in ordinary}
    assert len(ordinary_ids)==len(ordinary)
    rejoined_ordinary=[r for r in timeline if r['stable_row_id'] in ordinary_ids]
    assert [[r.get(k) for k in identity_fields] for r in ordinary]==[[r.get(k) for k in identity_fields] for r in rejoined_ordinary]
    old_event_ids={r['canonical_event_id'] for r in original if r.get('canonical_event_id')}
    new_event_ids={r['canonical_event_id'] for r in timeline if r.get('canonical_event_id')}
    assert old_event_ids.issubset(new_event_ids)
    if full_profile:
        from parser_core.full_hh_profile import project_current_card_evidence
        visual_readings=refresh_saved_card_readings(source,output)
        card_projection=project_current_card_evidence(timeline,visual_readings)
        (output/'full_hh_card_projection.json').write_text(
            json.dumps(card_projection,indent=2),'utf-8')
        write(output/'visual_evidence.jsonl',list(visual_readings.values()))
    full_options={'join_show_description':True} if full_profile else {}
    records, summary=analyze_timeline(timeline,ocr_status=ocr,**full_options)
    logical, complete, eligible, lifecycle=post_excel_contexts(output,records,**full_options)
    write_lifecycle(output,logical,lifecycle)
    if full_profile:
        from parser_core.table_service_events import service_view
        (output/'table_service_events.json').write_text(json.dumps(
            service_view([r['observed_hand'] for r in logical]),ensure_ascii=False,indent=2),'utf-8')
    write(output/'ordered_timeline.jsonl',timeline);write(output/'hand_analysis.jsonl',records)
    write(output/'logical_hand_analysis.jsonl',logical);write(output/'complete_hand_analysis.jsonl',complete)
    (output/'hand_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    write_start_stack_math(output,eligible_contexts=eligible,first_visible_contract=full_profile)
    hh=None
    if full_profile:
        from parser_core.poker_reconstruction_layer import reconstruction_config
        from parser_core.full_hh_profile import reconstruction_settings
        from parser_core.stack_evidence import StackEvidenceProvider
        from parser_core.hh_export import export_hand_histories
        metadata=json.loads((source/'run_metadata.json').read_text('utf-8'))
        provider=StackEvidenceProvider(resolve_stack_timeline_path(source),video_sha256=metadata['source_video_sha256'])
        from parser_core.simple_saved_policy import SimpleSavedPolicyProvider
        provider.simple_sources=SimpleSavedPolicyProvider.from_saved(source,logical,player_database=player_database)
        provider.actor_sources=provider.simple_sources.actor_sources
        reconstruction=write_reconstruction(output,complete,config=reconstruction_settings(reconstruction_config()),
            stack_provider=provider,stack_start_stack=True)
        hh=export_hand_histories(output,logical,simple_provider=provider.simple_sources,
                                 actor_provider=provider.actor_sources)
    else:
        reconstruction=write_reconstruction(output,complete)
    snapshots=json.loads((source/'seat_forced_summary.json').read_text('utf-8'))['forced_snapshots']
    before=metrics(read(source/'logical_hand_analysis.jsonl'),read(source/'poker_reconstruction.jsonl'),original,read(source/'start_stack_math.jsonl'),snapshots)
    after=metrics(logical,read(output/'poker_reconstruction.jsonl'),timeline,read(output/'start_stack_math.jsonl'),snapshots)
    report=dict(scope='SAVED_POST_PIXEL_REPLAY_NO_DECODE_NO_OCR_NO_NEW_CAPTURE',source=str(source),
        before=before,after=after,ordinary_source_rows_identity_order_raw_hash_parity=True,
        original_canonical_event_ids_retained=True,new_canonical_event_ids=sorted(new_event_ids-old_event_ids),
        source_input_hashes=hashes,reconstruction_summary=reconstruction,
        binding_replay=binding_replay,full_profile=full_profile,hh=hh,
        performance_code_changed=False,frozen_vendor_changed=False,verdict='QUALITY_PARTIAL')
    assert hashes=={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names}
    (output/'quality_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    if full_profile:(output/'replay_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(dict(before=before,after=after),ensure_ascii=False,indent=2))
    return report
