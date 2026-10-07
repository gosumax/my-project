"""Offline continuity overlay on the frozen edge gate; never imported by production.

Relationships are sidecar evidence. Boundaries, message assembly and every source
field are frozen. Nonidentity window vetoes outrank physical-row continuity.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

from parser_core.edge_gate import _identity, _safe, _confirmed_end, classify_edges, confirmed_hand_starts
from parser_core.poker_hand_assembly import assemble_messages


def contexts(rows, baseline):
    """Attribute the exact v1 predicates; assert agreement with frozen output."""
    groups = defaultdict(list)
    result = {}
    summaries = {(s['table_id'], s['session_id']): s for s in baseline.sessions}
    for pos, row in enumerate(rows):
        key = (row.get('table_id'), row.get('session_id'))
        groups[key].append(pos)
    for key, positions in groups.items():
        summary = summaries.get(key)
        if not summary or not summary['input_order_identity_valid']:
            for p in positions:
                result[p] = dict(reasons=['SESSION_OR_SOURCE_ORDER_IDENTITY_INVALID'], interval=None)
            continue
        source = [rows[p] for p in positions]
        messages = assemble_messages(source)
        starts = sorted(confirmed_hand_starts(messages))
        windows = [messages[b:starts[n+1] if n+1 < len(starts) else len(messages)]
                   for n, b in enumerate(starts)]
        repeated = set()
        seen = {}
        for n, window in enumerate(windows):
            for number in {m['hand_number'] for m in window
                           if m['action'] in {'HAND_HEADER', 'HAND_ID', 'NEW_HAND'} and m['hand_number']}:
                if number in seen:
                    repeated.update((seen[number], n))
                else:
                    seen[number] = n
        byid = {_identity(rows[p]): p for p in positions}
        for message in messages:
            for row in message['source_rows']:
                result[byid[_identity(row)]] = dict(reasons=['MISSING_CONFIRMED_HAND_START'], interval=None)
        for n, (window, hand) in enumerate(zip(windows, summary['hands'])):
            numbers = {m['hand_number'] for m in window
                       if m['action'] in {'HAND_HEADER', 'HAND_ID', 'NEW_HAND'} and m['hand_number']}
            epochs = {r.get('epoch') for m in window for r in m['source_rows']}
            reasons = []
            if n in repeated: reasons.append('REPEATED_NUMBER_WINDOWS')
            if len(numbers) > 1: reasons.append('CONFLICTING_HAND_NUMBERS')
            if len(epochs) > 1: reasons.append('EPOCH_DISCONTINUITY')
            if sum(m['action'] == 'NEW_HAND' for m in window) > 1: reasons.append('MULTIPLE_NEW_HAND_MESSAGES')
            if any(m['action'] == 'NEW_HAND' and not _safe(m) for m in window):
                reasons.append('UNSAFE_NEW_HAND_BOUNDARY')
            expected = 'UNRESOLVED_EDGE' if reasons else 'KEEP' if hand['end_reason'] else 'PARTIAL_END'
            assert expected == hand['edge_status']
            if not reasons: reasons = ['PROVEN_WINDOW_EDGES'] if hand['end_reason'] else ['MISSING_PROVEN_HAND_END']
            interval = dict(start_frame=hand['start_frame'], start_row_id=hand['start_row_id'],
                            start_row_index=rows[byid[hand['start_row_id']]]['row_index'],
                            next_start_frame=summary['hands'][n+1]['start_frame'] if n+1 < len(windows) else None,
                            end_reason=hand['end_reason'], window_last_frame=window[-1]['frame_id'],
                            hand_number_candidate=hand['hand_number_candidate'])
            for message in window:
                for row in message['source_rows']:
                    p = byid[_identity(row)]
                    assert baseline.rows[p]['edge_status'] == expected
                    result[p] = dict(reasons=list(reasons), interval=interval,
                        window_messages=window,
                        unsafe_new_hand_messages=[[_identity(r) for r in m['source_rows']]
                            for m in window if m['action']=='NEW_HAND' and not _safe(m)])
    return result


@dataclass
class Simulation:
    rows: list
    attribution: list
    chains: list
    relationships: list


def simulate(rows, relationships, baseline=None, row_context=None):
    baseline = baseline or classify_edges(rows)
    row_context = row_context or contexts(rows, baseline)
    output = copy.deepcopy(baseline.rows)
    byid = {_identity(row): row for row in rows}
    if len(byid) != len(rows):
        raise ValueError('Ambiguous source IDs cannot be linked')
    adjacency = defaultdict(list)
    admitted = []
    pairs = set()
    for rel in relationships:
        # Fail closed even if a caller injects an unreviewed/similar-text pair.
        if rel.get('visual_label') != 'VISUALLY_SAME' or rel.get('relationship') != 'VALIDATED_SAME_PHYSICAL_OCCURRENCE':
            continue
        old, new = rel['old_track_id'], rel['new_track_id']
        if old not in byid or new not in byid or old == new or (old, new) in pairs:
            raise ValueError('Missing, reflexive or duplicate relationship endpoint')
        pairs.add((old, new))
        for side, rid in [('old', old), ('new', new)]:
            r = byid[rid]
            if (r['table_id'], r['session_id'], r['frame_id'], r['row_pixel_sha256']) != (
                    rel['table_id'], rel['session_id'], rel[side+'_first_frame'], rel[side+'_source_pixel_hash']):
                raise ValueError('Relationship/source provenance mismatch')
        if rel['old_first_frame'] > rel['new_first_frame']:
            raise ValueError('Reversed relationship')
        admitted.append(copy.deepcopy(rel))
        adjacency[old].append((new, rel))
        adjacency[new].append((old, rel))
    components, component_by_id = [], {}
    for root in adjacency:
        if root in component_by_id: continue
        nodes, queue, links = set(), [root], {}
        while queue:
            node = queue.pop()
            if node in nodes: continue
            nodes.add(node)
            for other, rel in adjacency[node]:
                links[rel['pair_id']] = rel
                queue.append(other)
        node_rows = [byid[node] for node in nodes]
        indegree, outdegree = Counter(), Counter()
        for rel in links.values():
            indegree[rel['new_track_id']] += 1
            outdegree[rel['old_track_id']] += 1
        conflicts = []
        if len(links) != len(nodes)-1 or max(indegree.values(), default=0) > 1 or max(outdegree.values(), default=0) > 1:
            conflicts.append('BRANCH_OR_CYCLE')
        if len({r['epoch'] for r in node_rows}) != 1: conflicts.append('EPOCH_DISCONTINUITY')
        if any(rel.get('boundary_frames') for rel in links.values()): conflicts.append('VALIDATION_BOUNDARY_CONFLICT')
        chain = dict(track_ids=sorted(nodes, key=lambda node: (byid[node]['frame_id'], byid[node]['row_index'])),
                     pair_ids=sorted(links), first_seen=min(r['frame_id'] for r in node_rows),
                     last_seen=max(rel['new_last_frame'] for rel in links.values()), conflicts=conflicts)
        # last_seen is a frozen observed lifetime, never extrapolated to EOF.
        chain['last_seen'] = max(max(rel['old_last_frame'], rel['new_last_frame']) for rel in links.values())
        components.append(chain)
        for node in nodes: component_by_id[node] = chain
    attribution = []
    for p, row in enumerate(rows):
        rid = _identity(row)
        context = row_context[p]
        chain = component_by_id.get(rid)
        used = []
        ownership = 'NO_VALIDATED_CONTINUITY'
        first, last = row['frame_id'], row['frame_id']
        if chain:
            first, last = chain['first_seen'], chain['last_seen']
            # Record an exact validated path from earliest observation to this ID.
            root = chain['track_ids'][0]
            queue, paths = deque([root]), {root: []}
            while queue and rid not in paths:
                node = queue.popleft()
                for other, rel in adjacency[node]:
                    if other not in paths:
                        paths[other] = paths[node] + [rel['pair_id']]
                        queue.append(other)
            used = paths[rid]
            interval = context['interval']
            start = (interval['start_frame'], interval['start_row_index']) if interval else None
            earliest = min((byid[node]['frame_id'], byid[node]['row_index']) for node in chain['track_ids'])
            current = (row['frame_id'], row['row_index'])
            boundary_veto = any(reason in context['reasons'] for reason in {
                'REPEATED_NUMBER_WINDOWS', 'CONFLICTING_HAND_NUMBERS', 'MULTIPLE_NEW_HAND_MESSAGES',
                'UNSAFE_NEW_HAND_BOUNDARY', 'EPOCH_DISCONTINUITY', 'SESSION_OR_SOURCE_ORDER_IDENTITY_INVALID'})
            if chain['conflicts']:
                ownership = 'UNRESOLVED_CHAIN_CONFLICT'
            elif start is None:
                ownership = 'MISSING_CONFIRMED_HAND_START'
            elif earliest < start <= current:
                ownership = 'PREVIOUS_HAND_TAIL_SUPPORTED'
            elif boundary_veto:
                ownership = 'UNRESOLVED_BOUNDARY_OWNERSHIP'
            else:
                ownership = 'CURRENT_WINDOW_CONTINUITY_SUPPORTED'
        # Attribution first; then only an unsafe NEW_HAND observation proven
        # to belong before the fixed start may cease vetoing the current window.
        after = list(context['reasons'])
        attribution.append(dict(sequence_id=row.get('sequence_id'), stable_row_id=rid,
            old_edge_status=baseline.rows[p]['edge_status'], new_edge_status=output[p]['edge_status'],
            reason_before=context['reasons'], reason_after=after, hand_interval=context['interval'],
            relationship_path_used=used, continuity_component_pair_ids=chain['pair_ids'] if chain else [],
            old_stable_id=chain['track_ids'][0] if chain else None, new_stable_id=rid,
            physical_first_seen_frame=first, physical_last_seen_frame=last,
            lifetime_scope='FROZEN_CAUSAL_REVIEW_CUTOFF' if chain else 'SOURCE_ONSET_ONLY',
            ownership_evidence=ownership, chain_conflicts=chain['conflicts'] if chain else [],
            duplicate_poker_event_suppressed=False,
            veto_retained=baseline.rows[p]['edge_status']=='UNRESOLVED_EDGE'))
    id_position = {_identity(row): p for p,row in enumerate(rows)}
    windows = defaultdict(list)
    for p,a in enumerate(attribution):
        interval=a['hand_interval']
        if interval:
            windows[(rows[p]['table_id'],rows[p]['session_id'],interval['start_row_id'])].append(p)
    for positions in windows.values():
        context=row_context[positions[0]]
        # Numbers, epoch discontinuities, multiple starts and source conflicts
        # remain absolute vetoes. SAME never certifies an unsafe new boundary.
        if context['reasons'] != ['UNSAFE_NEW_HAND_BOUNDARY']:
            continue
        if any(attribution[p]['chain_conflicts'] for p in positions):
            continue
        unsafe=context['unsafe_new_hand_messages']
        marker_ids={rid for message in unsafe for rid in message}
        # Only source identity failures may be discharged. OCR errors, missing
        # text and unrelated notes remain blockers even with a SAME proof.
        if any(byid[rid].get('origin')!='UNRESOLVED_IDENTITY' or
               byid[rid].get('status')!='UNRESOLVED' or
               byid[rid].get('notes_error','') not in {None,'','TYPE_NOT_RECOGNIZED'} or
               not byid[rid].get('raw_ocr_text') or
               not all(rel.get('old_source_ocr_ok') and rel.get('new_source_ocr_ok')
                       for _,rel in adjacency[rid]) for rid in marker_ids):
            continue
        if not unsafe or not all(
                attribution[id_position[rid]]['ownership_evidence']=='PREVIOUS_HAND_TAIL_SUPPORTED'
                and attribution[id_position[rid]]['relationship_path_used'] for message in unsafe for rid in message):
            continue
        tail_positions={p for p in positions if attribution[p]['ownership_evidence']=='PREVIOUS_HAND_TAIL_SUPPORTED'}
        proofs=[dict(source_row_ids=message,
                     relationship_paths=[attribution[id_position[rid]]['relationship_path_used'] for rid in message])
                for message in unsafe]
        end_reason=context['interval']['end_reason']
        if end_reason and end_reason!='NEXT_CONFIRMED_HAND_START':
            current_messages=[m for m in context['window_messages'] if not any(
                id_position[_identity(r)] in tail_positions for r in m['source_rows'])]
            # Never borrow a previous-hand payout/end. Never manufacture an end
            # when the frozen gate did not have one.
            end_reason=_confirmed_end(current_messages)
        status='KEEP' if end_reason else 'PARTIAL_END'
        for p in positions:
            a=attribution[p]
            a['window_continuity_proofs']=proofs
            a['effective_end_reason']=end_reason
            if p in tail_positions:
                root=a['old_stable_id']
                owner_pos=id_position[root]
                owner_status=baseline.rows[owner_pos]['edge_status']
                # Physical-row continuity is insufficient to admit a duplicate
                # downstream poker-event row; KEEP owners remain quarantined.
                new_status=owner_status if owner_status in {'PARTIAL_START','PARTIAL_END','UNRESOLVED_EDGE'} else 'UNRESOLVED_EDGE'
                a['reason_after']=['PREVIOUS_HAND_TAIL']+row_context[owner_pos]['reasons']
                if owner_status=='KEEP': a['reason_after'].append('DOWNSTREAM_EVENT_MAPPING_UNPROVEN')
                a['physical_owner_interval']=row_context[owner_pos]['interval']
            else:
                new_status=status
                a['reason_after']=['VALIDATED_PREVIOUS_HAND_TAIL_REMOVES_UNSAFE_MARKER_VETO',
                                   'PROVEN_WINDOW_EDGES' if status=='KEEP' else 'MISSING_PROVEN_HAND_END']
            output[p]['edge_status']=new_status
            a['new_edge_status']=new_status
            a['veto_retained']=new_status=='UNRESOLVED_EDGE'
    return Simulation(output, attribution, components, admitted)
