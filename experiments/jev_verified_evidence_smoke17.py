"""Isolated saved-source Jev evidence receipts and deterministic offline arms.

Preparation uses stdlib only. No production, edge gate, OCR or video execution.
API calls are orchestrated separately through the installed Jev connector.
"""
import argparse
import base64
from collections import Counter
import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/edge_hand_attribution_window_simulation_20261001_v2'
AUDIT=ROOT/'outputs/remaining28_fixed_window_audit_20261001_v2'
SOURCE=ROOT/'runs/production_smoke17_20261001_013909'
PRELIM=ROOT/'outputs/jev_verified_evidence_smoke17_20261001'
OUT=ROOT/'outputs/jev_structured_evidence_smoke17_20261001_v2'

def read(p):return [json.loads(l) for l in p.read_text('utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def action_context(hand,record,index):
    if index<0:return dict(scope='AGGREGATE_HAND',player=None,street=None,prior_contribution=None,
                          constraints_trusted=False,compatible_semantics=[],unknown=['No single action index'])
    message=hand['messages'][index];step=record['observed_ledger']['steps'][index];before=step['before']
    previous_street=max([i for i,m in enumerate(hand['messages'][:index]) if m['action'] in {'FLOP','TURN','RIVER'}],default=-1)
    own=Decimal(before['street_contributions_by_player'].get(message.get('player'),'0'))
    required=Decimal(before['current_bet'])-own
    # Never turn a default-ADDED bookkeeping prefix into an observed fact.
    ambiguous_prior=[i for i,m in enumerate(hand['messages'][:index]) if i>previous_street and m['action']=='CALL'
        and Decimal(record['observed_ledger']['steps'][i]['before']['street_contributions_by_player'].get(m['player'],'0'))!=0]
    prefix_errors=[{k:b[k] for k in ['code','message_index','source_row_ids']} for b in record['fully_valid_blocking_checks']
        if ((0<=b['message_index']<index and b['code']!='AMOUNT_SEMANTICS_UNPROVEN') or
            (b['message_index']==index and b['code'] in {'ROW_IDENTITY_UNRESOLVED','PLAYER_IDENTITY_CONFLICT',
                'ACTION_AFTER_FOLD','ACTION_AFTER_HAND_END','ACTION_AFTER_ALLIN'}) or
            (b['message_index']<0 and b['code'] in {'INCOMPLETE_START','HAND_BOUNDARY_CONFLICT'}))]
    trusted=bool(hand['start_observed'] and not hand['boundary_conflicts'] and not ambiguous_prior and not prefix_errors)
    candidate_effects={}
    if message['action'] in {'CALL','BET'} and message['amount'] is not None:
        amount=Decimal(message['amount'])
        for meaning,delta in [('ADDED',amount),('TOTAL',amount-own)]:
            legal=delta>0 and (delta<=required and (delta==required or message.get('all_in',False))
                if message['action']=='CALL' else Decimal(before['current_bet'])==0)
            candidate_effects[meaning]=dict(added_amount=str(delta),resulting_street_contribution=str(own+delta),
                agrees_with_local_state_constraints=legal if trusted else None)
    board=[]
    for m in hand['messages'][:index]:
        if m['action'] in {'FLOP','TURN','RIVER'}:board=m.get('cards',[])
    return dict(message_index=index,player=message.get('player'),action_candidate=message['action'],
        ocr_amount_candidate=message.get('amount'),all_in_candidate=message.get('all_in',False),
        street=before['street'],board_pipeline_candidate=board,
        previous_street_actions=list(range(previous_street+1,index)),prior_player_street_contribution=str(own),
        current_bet=str(before['current_bet']),amount_to_call=str(required),
        next_neighbor_message=index+1 if index+1<len(hand['messages']) else None,
        previous_neighbor_message=index-1 if index else None,
        prefix_conflicts=prefix_errors,ambiguous_prior_call_indices=ambiguous_prior,
        constraints_trusted=trusted,scope='CONDITIONAL_DERIVED_LOCAL_STATE_NOT_OBSERVED_STACK_OR_POT',
        pot_observed=None,stack_observed=None,candidate_effects=candidate_effects,
        compatible_semantics=[k for k,v in candidate_effects.items() if v['agrees_with_local_state_constraints'] is True])


def structured_state(hand,record,context,ocr,visual,targets):
    row_ids=[rid for m in hand['messages'] for rid in m['source_row_ids']]
    refs={rid:i for i,rid in enumerate(row_ids)}
    observations=[]
    for rid in row_ids:
        row=ocr[rid]
        observations.append(dict(row_ref=refs[rid],frame=row['frame_id'],timestamp_seconds=
            row['source_pts']*row['time_base_num']/row['time_base_den'],epoch=row['epoch'],
            raw_ocr_response=row['raw_ocr_response'],pixel_sha256=row['row_pixel_sha256']))
    known_visual=[dict(source_row_id=v['row_id'],card_evidence_id=v.get('card_evidence_id'),
        frame=v['frame_id'],timestamp_seconds=v.get('video_time'),epoch=v['epoch'],
        status=v['status'],cards_reader_output=v.get('cards'),crop_sha256=v.get('crop_sha256'),
        response_source_sha256=v.get('response_source_sha256'),identity_status=v.get('identity_status'),
        continuity_status=v.get('continuity_status'),context_reason=v.get('context_reason'))
        for v in visual if v.get('row_id') is not None and v['table_session_id']==hand['session_id'] and
        (v['row_id'] in refs or (hand['hand_id'] is not None and v.get('hand_id')==hand['hand_id']))]
    for key,target in targets.items():target['derived_action_context']=action_context(hand,record,target['blocker']['message_index'])
    return dict(context=context,
        OBSERVED=dict(row_dictionary=row_ids,ocr_rows=observations,available_card_reader_receipts=known_visual,
            hand_header_texts=[m['raw_text'] for m in hand['messages'] if m['action']=='HAND_HEADER'],
            modality='Pipeline text/fact records only; no Jev OCR/pixel authority'),
        DERIVED=dict(immutable_fixed_interval=context['fixed_interval'],candidate_hand_number=hand['hand_id'],
            ordered_actions=[dict(message_index=i,row_refs=[refs[rid] for rid in m['source_row_ids']],
                parsed_action=m['action'],parsed_player=m.get('player'),parsed_amount=m.get('amount'),
                source_type='PIPELINE_PARSED_FROM_OBSERVED_OCR',all_in=m.get('all_in',False),
                board_or_hole_cards=m.get('cards',[]),track_origins=[r.get('origin') for r in m['source_rows']])
                for i,m in enumerate(hand['messages'])],
            disputed_action_contexts={key:target['derived_action_context'] for key,target in targets.items()},
            existing_blockers=[{k:b[k] for k in ['code','message_index','source_row_ids','detail']} for b in record['fully_valid_blocking_checks']],
            rules=['ADDED means added_amount equals OCR amount; TOTAL means resulting street contribution equals OCR amount.',
                'Ordinary CALL must add current_bet minus previous player contribution; observed all-in permits a smaller positive increment.',
                'Unopened-street BET must be positive and current_bet zero.',
                'Numeric state from default-ADDED bookkeeping is conditional if earlier ambiguous CALL or unresolved prefix exists.',
                'Prefix pot imbalance before payout is not a failed conservation check. Final delta alone never proves amount semantics.']),
        CANDIDATES=dict(amount_interpretations=['ADDED','TOTAL','UNRESOLVED'],
            effects_by_question={key:target['derived_action_context'].get('candidate_effects',{}) for key,target in targets.items()},
            nonsemantic_options=['SUPPORTED','CONTRADICTED','UNRESOLVED']),
        UNKNOWN=dict(observed_pot=None,observed_stacks=None,global_call_bet_ui_contract=None,
            unobserved_actions_amounts_cards_wins_ends='Must remain unknown; never invent',
            uncertain_event_identity='Retain all existing critical identity blockers; equal text/hash is not unique event proof',
            conditional_prefixes='See constraints_trusted and prefix_conflicts per action',
            visual_limitation='No images supplied to Jev; reader outputs retain their own RAW_UNREVIEWED/UNRESOLVED statuses'))

def prepare():
    if OUT.exists():raise ValueError('Fresh output required')
    frozen=json.loads((AUDIT/'input_hashes.json').read_text('utf-8'))
    protected={p:sha(ROOT/p) for p in frozen}
    if protected!=frozen:raise ValueError('Frozen hash drift')
    production=json.loads((ROOT/'production_manifest.json').read_text('utf-8'))['hashes']
    if any(sha(ROOT/p)!=h for p,h in production.items()):raise ValueError('Production drift')
    hands=json.loads((BASE/'CORE_FIXED_WINDOWS.assembled_hands.json').read_text('utf-8'))
    attr=read(AUDIT/'all28_attribution.jsonl')
    physical={r['row_id']:r for r in read(SOURCE/'physical_rows.jsonl')}
    ocr={r['row_id']:r for r in read(SOURCE/'raw_ocr_results.jsonl')}
    visual=read(SOURCE/'visual_evidence.jsonl')
    records=read(BASE/'CORE_FIXED_WINDOWS.hands.jsonl')
    OUT.mkdir();(OUT/'requests').mkdir();(OUT/'receipts').mkdir();(OUT/'sources').mkdir()
    write(OUT/'input_hashes_before.json',protected)
    write(OUT/'production_hashes_before.json',production)
    ranked=sorted(attr,key=lambda r:(r['hand_index']!=4,not r['math_checks']['pass'],len(r['all_blockers']),r['hand_index']))
    plan=[]
    for rank,r in enumerate(ranked):
        hand=hands[r['hand_index']];ids=[rid for m in hand['messages'] for rid in m['source_row_ids']]
        sources=[]
        for rid in ids:
            p=physical[rid];o=ocr[rid]
            if p['row_pixel_sha256']!=o['row_pixel_sha256']:raise ValueError('OCR pixel binding mismatch')
            pixels=base64.b64decode(p['row_pixels_base64'],validate=True)
            if hashlib.sha256(pixels).hexdigest()!=p['row_pixel_sha256']:raise ValueError('Pixels hash mismatch')
            sources.append(dict(source_row_id=rid,physical=copy.deepcopy(p),raw_ocr=copy.deepcopy(o)))
        context=dict(table_id=hand['table_id'],session_id=hand['session_id'],hand_id=hand['hand_id'],
                     hand_index=r['hand_index'],fixed_interval=r['fixed_interval'],start_row_id=r['start_row_id'])
        source_file=f'sources/hand_{r["hand_index"]:02d}.json'
        write(OUT/source_file,dict(context=context,sources=sources,assembled_hand=hand))
        questions={};targets={}
        # Every disputed amount gets an independent typed question and source binding.
        for b in r['blocker_groups']['amount_semantics']:
            mi=b['message_index'];m=hand['messages'][mi];key=f'semantics_m{mi}'
            questions[key]=dict(type='choice',instructions=dict(target_source_row_ids=b['source_row_ids'],
                question='Does supplied observed UI/OCR context establish whether this exact action amount is ADDED (increment of street contribution) or TOTAL (resulting street contribution)? Require observable format/contrast or independent before/after contribution evidence. Do not use final pot balance, general poker habits, or a prior model-derived contract as proof. If both agree or context does not discriminate, choose UNRESOLVED. Do not infer a global rule.'),
                criteria=dict(ADDED='Observed UI or independent source witness discriminates incremental amount from total.',
                              TOTAL='Observed UI or independent source witness discriminates resulting total from incremental amount.',
                              UNRESOLVED='The supplied observations cannot discriminate or have missing/conflicting evidence.'))
            targets[key]=dict(kind='AMOUNT_SEMANTICS',blocker=copy.deepcopy(b),action=m['action'],
                              allowed_evidence_row_ids=ids)
        groups=[('identity_provenance','identity/event multiplicity and hand ownership, independent of equal text/pixels'),
                ('action_state','actual chronology and legality of the flagged action, including genuine after WIN/end or after fold'),
                ('completeness','missing observed action/start/end/amount'),
                ('card_evidence','missing independently recognized, correctly bound card evidence'),
                ('payout','payout/side-pot entitlement and completeness based on observed records')]
        for group,topic in groups:
            blockers=r['blocker_groups'][group]
            if not blockers:continue
            # Per-hand representative first blocker; all hand blockers are supplied as context.
            b=blockers[0];key=f'evidence_{group}'
            questions[key]=dict(type='choice',instructions=dict(target_source_row_ids=b['source_row_ids'],
                blocker_code=b['code'],question=f'Does the supplied source context already establish a concrete fact that contradicts this blocker about {topic}? Identify only whether the exact source evidence can justify reconsideration. No imagined amounts/actions/cards/WIN, no text-equality identity merging, no boundary changes, and no judgement that the hand is valid. Confidence cannot replace observation. Missing independent witnesses mean UNRESOLVED.'),
                criteria=dict(SUPPORTED='Observed independent evidence concretely contradicts the blocker premise; a source fact is available to test deterministically.',
                              CONTRADICTED='Observed context supports the blocker premise; the proposed resolution is contradicted.',
                              UNRESOLVED='No sufficiently independent source witness or unresolved/conflicting context.'))
            targets[key]=dict(kind=group,blocker=copy.deepcopy(b),allowed_evidence_row_ids=ids)
        state=dict(context=context,source_format='PokerDom English dealer CHAT; this is not an externally certified CALL/BET UI specification',
            modality='TEXT_ONLY_JEV_NO_PIXEL_INSPECTION',
            visual_context_available='Exact 16px-high CHAT row pixels stored in source receipt. Wider table/bet/stack before-after views are not present in this packet. Raw pixel hashes are provenance, not semantic proof.',
            ordered_messages=[dict(message_index=i,source_row_ids=m['source_row_ids'],raw_text=m['raw_text'],
                parsed_action_candidate=m['action'],parsed_player_candidate=m['player'],parsed_amount_candidate=m['amount'],
                cards_candidate=m.get('cards'),source_rows=[dict(source_row_id=row['stable_row_id'],frame_id=row['frame_id'],
                    video_time=row.get('video_time'),origin=row.get('origin'),epoch=row.get('epoch'),
                    row_pixel_sha256=row.get('row_pixel_sha256'),raw_ocr_text=row.get('raw_ocr_text'),
                    raw_ocr_response=ocr[row['stable_row_id']]['raw_ocr_response']) for row in m['source_rows']])
                for i,m in enumerate(hand['messages'])],
            all_existing_blockers=[{k:v for k,v in b.items() if k in {'code','detail','message_index','source_row_ids','domain'}} for b in r['all_blockers']],
            missing_information='No new UI documentation, no observed stack/bet before-after measurements, no continuous capture witness certifying uncertain event identity. Do not invent these.')
        request=dict(model='jev-latest',state=state,questions=questions)
        request['state']=structured_state(hand,records[r['hand_index']],context,ocr,visual,targets)
        for key,q in questions.items():
            if targets[key]['kind']=='AMOUNT_SEMANTICS':
                q['instructions']['question']='Use OBSERVED, DERIVED, CANDIDATES and UNKNOWN separately. For this exact action, choose ADDED or TOTAL only if exactly one candidate agrees with the trusted local observed/derived constraints. Prior street contribution, amount to call, previous actions and source refs are supplied. Equal numeric effects do not discriminate semantics. Conditional prefixes, conflicting evidence or missing required inputs mean UNRESOLVED. Do not infer missing stack/pot or a global UI rule; do not use final pot balance or model knowledge as proof of OCR/pixels.'
                q['criteria']['ADDED']='Only the ADDED candidate agrees with trusted local state constraints (positive action increment equal to OCR amount).'
                q['criteria']['TOTAL']='Only the TOTAL candidate agrees with trusted local state constraints (resulting street contribution equal to OCR amount).'
            else:
                q['instructions']['question']+=' Treat OBSERVED as input readings, DERIVED as conditional calculations, CANDIDATES as hypotheses and UNKNOWN as missing facts. Never use Jev as OCR/pixel confirmation.'
        name=f'{rank+1:02d}_hand_{r["hand_index"]:02d}'
        write(OUT/'requests'/f'{name}.json',request)
        write(OUT/'requests'/f'{name}.binding.json',dict(context=context,source_file=source_file,
            source_file_sha256=sha(OUT/source_file),question_targets=targets))
        plan.append(dict(name=name,rank=rank+1,hand_index=r['hand_index'],hand_id=hand['hand_id'],
            table_id=hand['table_id'],blockers=len(r['all_blockers']),questions=len(questions)))
    write(OUT/'plan.json',plan)
    print(json.dumps(dict(prepared_hands=len(plan),questions=sum(p['questions'] for p in plan),first=plan[0])))

def verify_inputs():
    before=json.loads((OUT/'input_hashes_before.json').read_text('utf-8'))
    after={p:sha(ROOT/p) for p in before}
    if after!=before:raise ValueError('Protected input changed')
    write(OUT/'input_hashes_after.json',after)
    return len(after)


def checked_answer(question,answer):
    if answer.get('type')!='choice':raise ValueError('Unexpected answer type')
    options=question['criteria'];prob=answer['probabilities']
    if set(prob)!=set(options) or answer['choice'] not in options:raise ValueError('Out-of-contract choice')
    if any(not isinstance(v,(int,float)) or not 0<=v<=1 for v in prob.values()):raise ValueError('Invalid probabilities')
    if abs(sum(prob.values())-1)>1e-6:raise ValueError('Probability sum')
    if not 0<=answer['confidence']<=1:raise ValueError('Invalid confidence')
    return answer


def stable_result(value):
    if isinstance(value,dict):return {k:stable_result(v) for k,v in value.items() if k not in {'elapsed_ms','logic_ms'}}
    if isinstance(value,list):return [stable_result(v) for v in value]
    return value


def verify_dispatch(raw,request_path,binding_path,source_path):
    if raw['request_file'].replace('\\','/')!=request_path.relative_to(OUT).as_posix():raise ValueError('Foreign request receipt')
    if raw['request_sha256']!=sha(request_path) or raw['binding_sha256']!=sha(binding_path) or raw['source_sha256']!=sha(source_path):
        raise ValueError('Dispatch hash mismatch')


def review_proposal(target,answer):
    """A typed judgement without an observed witness is not a source fact.

    This experiment receives only choices/probabilities. It does not provide a
    positive repair translator: a positive hypothesis requires a separate,
    independently checked source witness before any future candidate change.
    Neither confidence nor a deterministic template written by us is evidence.
    """
    ctx=target.get('derived_action_context',{})
    if (target['kind']=='AMOUNT_SEMANTICS' and answer['choice'] in {'ADDED','TOTAL'} and
            ctx.get('constraints_trusted') and ctx.get('compatible_semantics')==[answer['choice']]):
        return dict(status='SUPPORTED',admitted_fact=dict(kind='ACTION_SCOPED_AMOUNT_SEMANTICS',
            message_index=ctx['message_index'],action=ctx['action_candidate'],semantics=answer['choice'],
            source_row_ids=target['blocker']['source_row_ids'],source_type='JEV_VERIFIED_DERIVED',
            proof=copy.deepcopy(ctx)),jev_choice=answer['choice'],confidence=answer['confidence'],
            evidence_basis='Exactly one candidate satisfies independent trusted prefix/action constraints; not final pot balance.')
    return dict(status='UNRESOLVED',admitted_fact=None,jev_choice=answer['choice'],
                confidence=answer['confidence'],source_row_ids=target['blocker']['source_row_ids'],
                evidence_basis='Exact source-bound OCR/context supplied; no independent discriminating witness returned.',
                reason='JEV_ABSTAINED' if answer['choice']=='UNRESOLVED' else 'POSITIVE_OR_NEGATIVE_MODEL_JUDGEMENT_REQUIRES_INDEPENDENT_SOURCE_WITNESS')


def finalize():
    if (OUT/'summary.json').exists():raise ValueError('Preserve previous finalization')
    plan=json.loads((OUT/'plan.json').read_text('utf-8'));receipts=[];facts=[];tokens=Counter();choices=Counter()
    hands=json.loads((BASE/'CORE_FIXED_WINDOWS.assembled_hands.json').read_text('utf-8'))
    old=read(BASE/'CORE_FIXED_WINDOWS.hands.jsonl')
    fingerprints={}
    for item in plan:
        name=item['name'];request_path=OUT/'requests'/f'{name}.json';binding_path=OUT/'requests'/f'{name}.binding.json'
        raw_path=OUT/'receipts'/f'{name}.raw.json'
        request=json.loads(request_path.read_text('utf-8'));binding=json.loads(binding_path.read_text('utf-8'))
        raw=json.loads(raw_path.read_text('utf-8'));source_path=OUT/binding['source_file']
        if sha(source_path)!=binding['source_file_sha256']:raise ValueError('Source receipt drift')
        verify_dispatch(raw,request_path,binding_path,source_path)
        response=raw['response']
        if response.get('isError'):raise ValueError('API error: not evidence')
        data=response.get('structuredContent')
        if data is None:
            blocks=[b['text'] for b in response['content'] if b['type']=='text']
            if len(blocks)!=1:raise ValueError('Ambiguous response')
            data=json.loads(blocks[0])
        if set(data['answers'])!=set(request['questions']):raise ValueError('Missing/extra answers')
        source=json.loads(source_path.read_text('utf-8'));ids={r['source_row_id'] for r in source['sources']}
        if source['context']!=binding['context'] or request['state']['context']!=binding['context']:
            raise ValueError('Foreign context')
        reviews={}
        for key,q in request['questions'].items():
            target=binding['question_targets'][key]
            if not set(target['blocker']['source_row_ids'])<=ids:raise ValueError('Foreign source witness')
            fresh_context=action_context(hands[item['hand_index']],old[item['hand_index']],target['blocker']['message_index'])
            if target['derived_action_context']!=fresh_context:raise ValueError('Derived source constraint drift')
            a=data['answers'][key];choices[a.get('choice','INVALID_RESPONSE')]+=1
            try:
                checked_answer(q,a)
                decision=review_proposal(target,a);answer_validation='VALIDATED'
            except ValueError as error:
                decision=dict(status='UNRESOLVED',admitted_fact=None,jev_choice=a.get('choice'),
                    confidence=a.get('confidence'),source_row_ids=target['blocker']['source_row_ids'],
                    reason='MALFORMED_NONADMISSIBLE_API_ANSWER',validation_error=str(error),
                    evidence_basis='Raw answer preserved; no probabilities normalized and no fact admitted.')
                answer_validation='MALFORMED_NONADMISSIBLE'
            reviews[key]=dict(target=target,answer=a,answer_validation=answer_validation,review=decision)
            fact=reviews[key]['review']['admitted_fact']
            if fact:
                if raw.get('hash_binding_method')!='PINNED_BEFORE_API':raise ValueError('Positive fact lacks dispatch-time pin')
                facts.append(dict(hand_index=item['hand_index'],receipt_name=name,question=key,**fact))
        fingerprints.update({str(p.relative_to(OUT)):sha(p) for p in [request_path,binding_path,raw_path,source_path]})
        receipt=dict(context=binding['context'],request_file=str(request_path.relative_to(OUT)),
            request_sha256=sha(request_path),source_file=binding['source_file'],source_sha256=sha(source_path),
            raw_response_file=str(raw_path.relative_to(OUT)),raw_response_sha256=sha(raw_path),
            question_answers=reviews,model=data['model'],usage=data['usage'],
            started=raw['started'],completed=raw['completed'],hash_binding_method=raw.get('hash_binding_method'),
            status='SUPPORTED' if any(v['review']['admitted_fact'] for v in reviews.values()) else 'UNRESOLVED',
            admitted_facts=[f for f in facts if f['receipt_name']==name])
        tokens.update(data['usage']);receipts.append(receipt)
        write(OUT/'receipts'/f'{name}.reviewed.json',receipt)
    write(OUT/'receipts.json',receipts);write(OUT/'evidence_hashes.json',fingerprints)
    write(OUT/'admitted_facts.json',facts)
    # Deterministic validation only: no assembler/gate/replay/Jev resolver call.
    sys.path.insert(0,str(ROOT))
    from parser_core.poker_logic_engine import validate_hand
    candidate=copy.deepcopy(hands)
    if facts:raise ValueError('Positive facts require independently reviewed complete hand-wide semantic coverage before a candidate translator is enabled')
    if candidate!=hands:raise ValueError('Unexpected candidate mutation')
    arms={};metrics={}
    for arm,arm_hands in [('CORE_FIXED_WINDOWS',hands),('CORE_FIXED_WINDOWS_JEV_VERIFIED_EVIDENCE',candidate)]:
        results=[]
        for i,hand in enumerate(arm_hands):
            checked=validate_hand(hand)
            if stable_result(checked)!=stable_result(old[i]['validator_result']):raise ValueError(f'Validator receipt drift in hand {i}')
            results.append(dict(hand_index=i,table_id=hand['table_id'],session_id=hand['session_id'],
                hand_id=hand['hand_id'],fixed_interval=hand['edge_start_evidence'],validator_result=checked))
        arms[arm]=results
        metrics[arm]=dict(candidates=len(results),logic_pass=sum(r['validator_result']['logic_pass'] for r in results),
            math_pass=sum(r['validator_result']['math']['pass'] for r in results),
            fully_valid=sum(r['validator_result']['status']=='FULLY_VALID_HAND' for r in results))
        write(OUT/f'{arm}.results.json',results)
    if stable_result(arms['CORE_FIXED_WINDOWS'])!=stable_result(arms['CORE_FIXED_WINDOWS_JEV_VERIFIED_EVIDENCE']):raise ValueError('Unexpected arm difference')
    protected=verify_inputs()
    write(OUT/'candidate_hands.json',candidate)
    summary=dict(verdict='JEV_EVIDENCE_INSUFFICIENT',jev_calls=len(receipts),questions=sum(p['questions'] for p in plan),
        response_choices=dict(choices),usage=dict(tokens),admitted_facts=len(facts),metrics=metrics,
        blockers_removed={},changed_hands=[],new_fully_valid_hands=[],regressed_hands=[],
        remaining_blockers=sum(len(r['validator_result']['certification_blockers']) for r in arms['CORE_FIXED_WINDOWS']),
        production_code_changed=False,protected_hashes_unchanged=protected,original_rows=8146,
        fixed_intervals_unchanged=True,source_rows_unchanged=True,no_dedup=True,no_rewrite=True,
        image_writes=0,ocr_calls=0,video_decode_calls=0,edge_gate_calls=0,validator_calls=58,
        positive_admission_translator='NO_FIX_APPLIED_NO_ELIGIBLE_FACTS_IN_CURRENT_FIXTURE',
        coverage='All 28 nonvalid hands; every amount-semantics blocker separately, one first blocker per other present domain. Not an individual Jev verdict on all 573 blockers.',
        visual_limit='Jev text-only. Saved CHAT pixels linked by hash, T06 visually inspected by root; no wider UI before/after view supplied.')
    preliminary=[json.loads(p.read_text('utf-8')) for p in (PRELIM/'receipts').glob('*.raw.json')]
    all_raw=[json.loads(p.read_text('utf-8')) for p in (OUT/'receipts').glob('*.raw.json')]
    summary.update(preliminary_calls=len(preliminary),structured_api_attempts=len(all_raw),
        structured_successful_calls=sum(not r['response'].get('isError',False) for r in all_raw),
        structured_transport_errors=sum(r['response'].get('isError',False) for r in all_raw),
        total_api_attempts=len(preliminary)+len(all_raw),total_successful_calls=len(preliminary)+len(receipts),
        malformed_nonadmissible_answers=sum(v['answer_validation']=='MALFORMED_NONADMISSIBLE'
            for r in receipts for v in r['question_answers'].values()),
        packet_format=['OBSERVED','DERIVED','CANDIDATES','UNKNOWN'],
        semantic_unique_trusted_candidates=sum(t['kind']=='AMOUNT_SEMANTICS' and t['derived_action_context'].get('constraints_trusted') and
            len(t['derived_action_context'].get('compatible_semantics',[]))==1
            for p in (OUT/'requests').glob('*.binding.json') for t in json.loads(p.read_text('utf-8'))['question_targets'].values()),
        request_hash_limit='First structured request hash reconstructed post-call from unchanged tool-dispatch file; no positive admission permitted. Other28 attempts pinned before API.')
    write(OUT/'summary.json',summary)
    first=receipts[0]
    lines=['# Jev verified evidence Smoke17 — offline experiment','',
        '## 1. JEV CALLS / RECEIPTS','',f'{summary["total_api_attempts"]} total API attempts:28 preliminary calls before packet-format clarification, then29 structured attempts (28 successful,1 HTTP400). Main arm:28 structured hand receipts /135 typed answers. Largest failed request retried once with lossless prefix-conflict references; exact HTTP400 cause unknown. Source facts/questions/candidates unchanged (compact_request_parity.json). Model jev-1.13.0. Main successful usage '+json.dumps(dict(tokens))+'.',
        'Exact requests, raw responses, reviewed receipts, question-to-source bindings and source payloads saved separately. Each reviewed receipt links request/response/source SHA256, table/session/fixed interval, raw OCR, frame/time and source IDs. Jev returns choices/probabilities, not a prose rationale or source witness. Coverage: '+summary['coverage'],'',
        'Packets separate OBSERVED readings/reader status, DERIVED parse/state and conditional calculations, CANDIDATES amount effects, UNKNOWN missing facts. No image API input. Previous street actions and neighbors link by message indices; adjacency alone does not certify an event. Prefix conflict/ambiguity disables trusted local constraints. '+summary['request_hash_limit'],'',
        f'Malformed/nonadmissible answers: {summary["malformed_nonadmissible_answers"]}. Two raw payout distributions sum to0.99; retained as returned, not normalized, no admission permitted. These are not valid evidence receipts for a positive fact.','',
        '## 2. T06 CALL/BET VERIFICATION','',
        'Hand #2876718617; table T06; session table_6f37cc14274a733e7ed2ad35; immutable start R000002; frames0→2565. R000020 frame185/time6.1667: `Dealer: billmurray calls 10 750`; R000033 frame2323/time77.4333: `Dealer: billmurray bets 10 956`. Root directly inspected saved 16×224 crops; text matches OCR. No prior/resulting contribution display or amount-format UI definition is visible in those strips.',
        '| Target | Jev choice | Confidence | Evidence status |','|---|---|---:|---|']
    for key,value in first['question_answers'].items():
        if value['target']['kind']=='AMOUNT_SEMANTICS':
            lines.append('|'+','.join(value['review']['source_row_ids'])+f'|{value["answer"]["choice"]}|{value["answer"]["confidence"]}|UNRESOLVED|')
    lines += ['', 'DERIVED CALL constraints: previous contribution0/current bet10750/required10750; both ADDED and TOTAL add10750. FLOP BET: previous contribution0/current bet0; both add10956. Each pair has two compatible candidates, so neither is uniquely established. Observed pot/stack remain UNKNOWN; default-ADDED bookkeeping is not relabeled as observed pot/stack. No global CALL/BET rule transferred. Existing validator consumes hand-wide semantics; a one-action receipt cannot certify other unproved actions. BET TOTAL cannot bypass its unchanged check.','',
        '## 3. NEW EVIDENCE','', 'New model receipts are available. Independently supported new semantic/identity/action/card/payout facts admitted:0. Do not equate a new receipt with a new proven source fact.','',
        '## 4. UNRESOLVED EVIDENCE','',f'All {summary["questions"]} Jev choices are '+json.dumps(dict(choices))+'. Exact target facts remain unresolved. No missing amounts/actions/cards/WIN or hand ends invented. The main source limitation is absent discriminating UI observations / continuous identity witnesses, not a validator threshold.','',
        '## 5. BLOCKERS REMOVED','',f'0 removed; {summary["remaining_blockers"]} baseline certification blockers remain (573 in the28 nonvalid hands). No probability-only repair translator exists in this arm.','',
        '## 6. BEFORE / AFTER METRICS','', '| Arm | Candidates | Logic PASS | Math PASS | Fully valid |','|---|---:|---:|---:|---:|']
    for arm,m in metrics.items():lines.append(f'|{arm}|{m["candidates"]}|{m["logic_pass"]}|{m["math_pass"]}|{m["fully_valid"]}|')
    lines += ['', 'Both arms revalidated by existing validate_hand; semantic results exactly match saved baseline, excluding only elapsed_ms/logic_ms runtime timing. No global semantics recomputation.','',
        '## 7. NEW FULLY VALID HANDS','', 'None. No source→Jev→admitted fact→new Fully valid chain exists. The previously valid hand remains valid.','',
        '## 8. REGRESSIONS','', 'None; no candidate row/message/hand changed. Genuine after-end/after-fold, conflicting numbers, uncertain epochs/identity/boundaries, missing start/end/cards/actions/amounts and branch conflicts retain their baseline checks. Tests validate rejection of unsupported model choices; no positive repair correctness is claimed.','',
        '## 9. PARITY','',f'All {protected} protected input hashes unchanged, including production manifest/code/configs and8146-row source. Baseline and candidate hand fields, source rows, message order and fixed interval metadata exactly equal.976 admitted source row instances retained. Jev evidence is separate; image writes0, video decode0, OCR0, edge-gate calls0. Production pipeline not launched.','',
        '## 10. TESTS','', 'See tests.json / verification.json for independent bounded checks. Deterministic validator calls58 (29 per arm); no production validator edits.','',
        '## 11. PRODUCTION CODE CHANGED: NO','', 'Only new isolated experiment/test/artifact files. No production, edge-gate, validator, config or manifest edits.','',
        '## 12. VERDICT','', '**JEV_EVIDENCE_INSUFFICIENT**. Fully valid remains1. This verdict applies to this source context and text-only Jev evaluation; additional independent visual/UI/identity evidence could change the result.',
        'Official modality reference: https://docs.typesafe.ai/concepts/state (text only; images/audio/video unsupported).','',
        '| Rank | Hand index | Table | Hand | Questions | Receipt status |','|---:|---:|---|---|---:|---|']
    for p in plan:lines.append(f'|{p["rank"]}|{p["hand_index"]}|{p["table_id"]}|{p["hand_id"] or "UNKNOWN"}|{p["questions"]}|UNRESOLVED|')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','verify','finalize'])
    a=parser.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='finalize':finalize()
    else:print(verify_inputs())
