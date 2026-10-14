"""Retained Smoke18 physical snapshots; no generated pixels, OCR or video."""
import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from parser_core.legitimate_no_sb import LegitimateNoSBSource, PHYSICAL_PREFIX_RULE, prove_no_sb
from parser_core.simple_saved_policy import (LineageActorProvider, forced_correction,
    hand_key, saved_hand_bounds, saved_nick_observations)
from parser_core.simple_seat_who import SeatIdentityLineage
from parser_core.poker_logic_engine import validate_hand

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT/'runs/smoke18_ready_current_20261013_205122/pipeline'


@pytest.fixture(scope='module')
def source():
    records = [json.loads(line) for line in (NATIVE/'logical_hand_analysis.jsonl').read_text('utf8').splitlines() if line.strip()]
    hands = [r['observed_hand'] for r in records]
    states = json.loads((NATIVE/'seat_forced_summary.json').read_text('utf8'))['state_receipts']
    actors = LineageActorProvider(SeatIdentityLineage(saved_nick_observations(NATIVE),states),
        saved_hand_bounds(records),hands)
    result = LegitimateNoSBSource(NATIVE,actors)
    result._load()
    return result


def hand(source,number='2875902711'):
    return source.actor_provider.canonical_hand(next(h for k,h in source.actor_provider.hands.items() if k[2] == number))


def rows(h):
    end = next(i for i,m in enumerate(h['messages']) if m['action'] == 'DEAL')
    return [r for m in h['messages'][:end+1] for r in m['source_rows']]


@pytest.mark.parametrize('number,frame,count',[('2875902772',161,12),('2875902711',35,11)])
def test_actual_single_snapshot_prefix_authorizes_without_manual_receipt(source,number,frame,count):
    h = hand(source,number); before = copy.deepcopy(h)
    proof = prove_no_sb(h,source)
    assert proof, source.last_rejection
    assert proof['authority'] == PHYSICAL_PREFIX_RULE
    assert proof['chat_completeness']['frame'] == frame
    assert len(proof['prefix_source_row_ids']) == count
    assert proof['sb_status'] == 'NOT_POSTED' and proof['posted_sb_total'] == 0
    assert proof['button_actor'] is None and proof['next_button_position'] == 'UNKNOWN'
    assert proof['occupancy_authority'] is False and proof['synthetic_post_authority'] is False
    assert 'manual_source' not in proof and 'review_source' not in proof
    assert h == before
    for actor in proof['participants']:
        forced,refs = forced_correction(h,actor=actor,no_sb_source=source)
        assert forced == dict(ANTE=700,SB=0,BB=6000 if actor == proof['bb_actor'] else 0)
        assert all(r['role'] != 'SB' for r in refs)


def cloned_packet(source,h):
    clone = copy.copy(source)
    clone.native = dict(source.native); clone.physical = dict(source.physical)
    clone.inventory = dict(source.inventory)
    clone.inventory['packets'] = dict(source.inventory['packets'])
    key = hand_key(h); frame = rows(h)[0]['frame_id']
    pid,packet = next((pid,p) for pid,p in source.inventory['packets'].items()
                     if (p['table_id'],p['session_id'],p['header_frame']) == (*key[:2],frame))
    packet = copy.deepcopy(packet); clone.inventory['packets'][pid] = packet
    return clone,packet['retained_event_admission_receipts'][0]


@pytest.mark.parametrize('mutation',[
    'missing_admission','unproven_birth','missing_packet','line_gap','compact_layout',
    'edge_offset','clipped_header','clipped_deal','tight_hash','padded_hash','clock',
    'width','duplicate_admission','failed_frame','native_sb','derived_sb',
    'unknown_prefix','no_end_boundary','no_positive_next_header','low_confidence',
])
def test_incomplete_or_contradictory_physical_prefix_stays_unknown(source,mutation):
    h = hand(source); clone,receipt = cloned_packet(source,h)
    rr = rows(h); key = hand_key(h); index = rr[0]['row_index']
    if mutation == 'missing_admission':receipt['admissions'].pop(3)
    elif mutation == 'unproven_birth':receipt['proof']['status'] = 'UNKNOWN'
    elif mutation == 'missing_packet':clone.inventory['packets'] = {}
    elif mutation == 'line_gap':receipt['new_spans'][index+2] = [232,242]
    elif mutation == 'compact_layout':receipt['new_spans'][index+1] = [164,175]
    elif mutation == 'edge_offset':receipt['geometry']['top_edge_spans'] = [[0,9]]
    elif mutation == 'clipped_header':receipt['new_spans'][index] = [0,10]
    elif mutation == 'clipped_deal':receipt['new_spans'][-1] = [367,377]
    elif mutation == 'tight_hash':receipt['admissions'][0]['row_pixel_sha256'] = '0'*64
    elif mutation == 'padded_hash':
        route = (rr[0]['table_id'],rr[0]['session_id'],rr[0]['frame_id'],index,rr[0]['stable_row_id'])
        clone.physical[route] = dict(clone.physical[route],row_pixel_sha256='0'*64)
    elif mutation == 'clock':receipt['source_pts'] += 1
    elif mutation == 'width':receipt['geometry']['shape'][1] += 1
    elif mutation == 'duplicate_admission':receipt['admissions'].append(copy.deepcopy(receipt['admissions'][0]))
    elif mutation == 'failed_frame':clone.inventory['frame_failures'] = [dict(table_id=key[0],session_id=key[1],frame=rr[0]['frame_id'])]
    elif mutation == 'native_sb':
        extra = dict(rr[0],frame_id=rr[0]['frame_id']+1,raw_ocr_text='Dealer: vadrou posts small blind 3000')
        clone.native[('T08',key[1],extra['frame_id'],0,'extra-contradiction')] = extra
    elif mutation == 'derived_sb':h['messages'].append(dict(h['messages'][3],action='SMALL_BLIND'))
    elif mutation == 'unknown_prefix':h['messages'][3]['action'] = 'UNKNOWN'
    elif mutation == 'no_end_boundary':h['next_boundary_observed'] = False
    elif mutation == 'no_positive_next_header':clone.inventory['positive_headers'] = [r for r in source.inventory['positive_headers'] if r['frame'] <= rr[0]['frame_id']]
    else:rr[0]['confidence'] = .5
    assert prove_no_sb(h,clone) is None, mutation


def test_t08_no_sb_preserves_observed_pot(source):
    h = hand(source)
    result = validate_hand(h,no_sb_source=source)
    assert result['status'] != 'INCOMPLETE_START'
    assert result['math']['pass'] and result['math']['delta'] == '0'
    assert result['no_sb_authority']['sb_status'] == 'NOT_POSTED'
    assert sum(int(m['amount']) for m in h['messages'] if m['action'] == 'WIN') == 16200


def capture_source(tmp_path,source):
    from scripts.run_smoke14_ram import write_capture_metadata
    target=tmp_path/'captured';target.mkdir()
    for path in source.hashes:
        shutil.copyfile(path,target/Path(path).name)
    metadata=json.loads((target/'run_metadata.json').read_text('utf8'))
    metadata['status']='RUNNING'
    (target/'run_metadata.json').write_text(json.dumps(metadata),'utf8')
    snapshot=write_capture_metadata(target,metadata)
    return target,metadata,snapshot


def test_persisted_physical_proof_survives_final_status_and_fresh_revalidation(tmp_path,source):
    target,metadata,snapshot=capture_source(tmp_path,source)
    states=json.loads((target/'seat_forced_summary.json').read_text('utf8'))['state_receipts']
    old=source.actor_provider
    actors=LineageActorProvider(SeatIdentityLineage(saved_nick_observations(target),states),
        old.bounds,old.hands.values(),old.associations,old.manual_reviews)
    current=LegitimateNoSBSource(target,actors)
    hands=[actors.canonical_hand(next(h for k,h in actors.hands.items() if k[2]==number))
           for number in ('2875902772','2875902711')]
    proofs=[prove_no_sb(h,current) for h in hands]
    assert all(proofs),current.last_rejection
    capture_bytes=snapshot.read_bytes()
    metadata['status']='PARTIAL_VISUAL'
    (target/'run_metadata.json').write_text(json.dumps(metadata),'utf8')
    assert [prove_no_sb(h,current) for h in hands]==proofs
    reopened=LegitimateNoSBSource(target,actors)
    assert [prove_no_sb(h,reopened) for h in hands]==proofs,reopened.last_rejection
    for proof in proofs:
        assert str(snapshot) in proof['source_sha256']
        assert str(target/'run_metadata.json') not in proof['source_sha256']
        assert len(proof['source_sha256'])==len(source.hashes)==9
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha
                   for p,sha in proof['source_sha256'].items())
    assert snapshot.read_bytes()==capture_bytes
    snapshot.write_bytes(capture_bytes+b'\n')
    assert prove_no_sb(hands[0],reopened) is None
    assert reopened.last_rejection=='NO_SB_SOURCE_CHANGED'


@pytest.mark.parametrize('change',['malformed','video_scope','identity_scope'])
def test_capture_metadata_never_falls_back_on_invalid_snapshot(tmp_path,source,change):
    target,metadata,snapshot=capture_source(tmp_path,source)
    if change=='malformed':snapshot.write_text('{broken','utf8')
    else:
        key='source_video_sha256' if change=='video_scope' else 'source_id'
        metadata[key]='0'*64
        snapshot.write_text(json.dumps(metadata),'utf8')
    candidate=LegitimateNoSBSource(target,source.actor_provider)
    assert prove_no_sb(hand(source),candidate) is None
    if change!='malformed':assert candidate.last_rejection=='NO_SB_CAPTURE_METADATA_SCOPE_CONFLICT'
