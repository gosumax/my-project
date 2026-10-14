"""Narrow native NO_SHOW capability; never infer an opponent's cards or rank."""
from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import cv2

from .actor_seat_point import source_time
from .current_chat_sources import FILES as CHAT_FILES, load_current_chat, observation_source_rows
from .native_nick_who import file_hash, validate_native_nick
from .poker_card_engine import best_rank
from .poker_hand_assembly import make_message
from .simple_saved_policy import LineageActorProvider, hand_key, saved_nick_observations
from .simple_seat_who import SeatIdentityLineage
from .visual_cards import associate_visual_evidence, project_chat_card_events
from .win_tail_evidence import _frozen, _png, current_win_context, verified_win_tail

RULE = 'AUTHORIZED_HIDDEN_LOSER_HOLE_CARDS_V1'
COMPARISON = 'NOT_REQUIRED_EXPLICIT_NO_SHOW'
_WIN = re.compile(r'(?:Dealer|Дилер)\s*:\s*(.+?) wins \d', re.I)
_SHOW = re.compile(r'(?:Dealer|Дилер)\s*:\s*(.+?) shows(?:\s|$)', re.I)


def _rows(path):
    with Path(path).open(encoding='utf8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _stamp(path):
    s = path.stat()
    return s.st_size, s.st_mtime_ns, s.st_ctime_ns


class HiddenLoserSource:
    """Independent sealed source parents plus the existing current SIMPLE WHO.

    Construct after the source writer is drained. The optional visual file is a
    current producer receipt file, useful when revalidating saved raw sources.
    No caller-supplied authority flag or cached hand annotation is consumed.
    """
    def __init__(self, source, actor_provider, *, visual_source=None):
        self.source = Path(source)
        self.actor_provider = actor_provider
        self.visual_source = Path(visual_source) if visual_source else self.source/'visual_evidence.jsonl'
        self.loaded = False
        self.failure = None
        self.last_rejection = None

    def _reject(self, reason):
        self.last_rejection = reason
        return None

    def _load(self):
        if self.failure:
            raise ValueError(self.failure)
        if self.loaded:
            if any(_stamp(p) != stamp for p, stamp in self.stamps.items()):
                raise ValueError('HIDDEN_SOURCE_PARENT_CHANGED')
            return
        actor = self.actor_provider
        if not isinstance(actor, LineageActorProvider):
            raise ValueError('CURRENT_SIMPLE_WHO_PROVIDER_REQUIRED')
        paths = [self.source/name for name in (*CHAT_FILES, 'seat_forced_inputs.jsonl', 'seat_forced_summary.json')]
        paths.append(self.visual_source)
        self.hashes = {str(p):file_hash(p) for p in paths}
        self.stamps = {p:_stamp(p) for p in paths}
        chat = load_current_chat(self.source, {n:self.hashes[str(self.source/n)] for n in CHAT_FILES})
        self.checkpoints = chat['context_checkpoints']
        self.frame_failures = chat['frame_failures']
        self.packets = {(p['table_id'],p['session_id'],p['header_frame']):p for p in chat['packets'].values()}
        self.owners = {}
        for item in chat['retained_row_ownership']:
            if item['source_join_status'] == 'SOURCE_OBSERVATION_PROVEN':
                self.owners[(item['table_id'],item['session_id'],item['frame'],item['row_id'])] = item['hand_id']
        self.native = {}
        self.clocks = defaultdict(dict)
        for packet in self.packets.values():
            for row, physical, response, pixels in observation_source_rows(packet):
                key = (row['table_id'],row['session_id'],row['frame_id'],row['stable_row_id'])
                if key in self.native:
                    raise ValueError('DUPLICATE_NATIVE_SOURCE_ROW')
                self.native[key] = row, physical, response
                clock = self.clocks[(physical['table_id'],physical['table_session_id'])]
                stamp = source_time(physical['source_pts'],
                    [physical['time_base_num'],physical['time_base_den']])
                prior = clock.get(physical['frame_id'])
                if prior is not None and prior != stamp:
                    raise ValueError('NATIVE_CHAT_FRAME_CLOCK_CONFLICT')
                clock[physical['frame_id']] = stamp
        for clock in self.clocks.values():
            ordered = [clock[f] for f in sorted(clock)]
            if ordered != sorted(set(ordered)):
                raise ValueError('NATIVE_CHAT_CLOCK_ORDER_CONFLICT')
        observations = saved_nick_observations(self.source)
        states = json.loads((self.source/'seat_forced_summary.json').read_text('utf8')).get('state_receipts', [])
        self.fresh_actors = LineageActorProvider(SeatIdentityLineage(observations, states),
            actor.bounds, actor.hands.values(), actor.associations, actor.manual_reviews,
            native_source=actor.native_source,player_registry=actor.player_registry)
        self.nick_rows = {}
        # Keep only proof-anchor parents, not a second complete pixel inventory.
        needed = {p['anchor']['source_line'] for hand in actor.hands.values()
                  for p in self.fresh_actors.for_hand(hand) if p.get('anchor',{}).get('source_line')}
        with (self.source/'seat_forced_inputs.jsonl').open(encoding='utf8') as stream:
            for line, text in enumerate(stream, 1):
                if line in needed:
                    self.nick_rows[line] = json.loads(text)
        visual = list(_rows(self.visual_source))
        self.visual = {r['row_key']:r for r in visual}
        if len(self.visual) != len(visual):
            raise ValueError('DUPLICATE_VISUAL_SOURCE_ROW')
        if any(file_hash(p) != self.hashes[str(p)] for p in paths):
            raise ValueError('HIDDEN_SOURCE_PARENT_CHANGED_DURING_LOAD')
        self.loaded = True

    def _native_row(self, row, hand):
        table, session, number = hand_key(hand)
        key = (f'T{table:02d}',session,row['frame_id'],row['stable_row_id'])
        native, physical, response = self.native[key]
        if self.owners.get((table,session,row['frame_id'],row['stable_row_id'])) != number:
            raise ValueError('CURRENT_HAND_NATIVE_OWNERSHIP_NOT_PROVEN')
        fields = ('table_id','session_id','frame_id','row_index','epoch','stable_row_id',
                  'raw_ocr_text','confidence','row_pixel_sha256','video_time')
        if any(row.get(k) != native.get(k) for k in fields):
            raise ValueError('NATIVE_SOURCE_FIELDS_CHANGED')
        if (response.get('status') != 'OK' or row.get('ocr_status') != 'OK'
                or row.get('origin') == 'UNRESOLVED_IDENTITY'
                or row.get('status') in {'ERROR','UNRESOLVED'}
                or row.get('canonical_conflict') or row.get('canonical_duplicate_of')):
            raise ValueError('FAILED_OR_CONFLICTING_SOURCE_ROW')
        for field in ('hand_id','hand_number'):
            if row.get(field) is not None and str(row[field]) != number:
                raise ValueError('FOREIGN_HAND_SOURCE_ROW')
        if row.get('source_pts') is not None and row['source_pts'] != physical['source_pts']:
            raise ValueError('SOURCE_CLOCK_CHANGED')
        if row.get('time_base') is not None and row['time_base'] != [physical['time_base_num'],physical['time_base_den']]:
            raise ValueError('SOURCE_CLOCK_CHANGED')
        return dict(row_id=row['stable_row_id'],frame=row['frame_id'],row_index=row['row_index'],
            pixel_sha256=row['row_pixel_sha256'],source_pts=physical['source_pts'],
            time_base=[physical['time_base_num'],physical['time_base_den']])

    def _message(self, message, hand):
        rows = message['source_rows']
        if not rows or message['source_row_ids'] != [r['stable_row_id'] for r in rows]:
            raise ValueError('MESSAGE_SOURCE_SET_CHANGED')
        refs = [self._native_row(r, hand) for r in rows]
        for a,b in zip(rows, rows[1:]):
            if (a['frame_id'] != b['frame_id'] or a['epoch'] != b['epoch']
                    or b['row_index'] != a['row_index']+1):
                raise ValueError('PHYSICAL_CONTINUATION_NOT_PROVEN')
        parsed = make_message(rows, require_pixel_cards=True,
                              pixel_sources={r['stable_row_id']:r for r in rows})
        aliases = self.actor_provider.actor_map(hand)
        actor = aliases.get(parsed.get('player'),parsed.get('player'))
        if parsed['action'] != message['action'] or actor != message.get('player'):
            raise ValueError('NATIVE_ACTION_OR_ACTOR_CHANGED')
        if message['action'] == 'WIN' and str(parsed.get('amount')) != str(message.get('amount')):
            raise ValueError('NATIVE_PAYOUT_AMOUNT_CHANGED')
        return refs

    def _who(self, actor, hand, frames):
        old = self.actor_provider.for_actor(hand, actor)
        fresh = self.fresh_actors.for_actor(hand, actor)
        table,session,number = hand_key(hand)
        for proof in (old,fresh):
            if (proof.get('status') != 'PROVEN' or proof.get('table_id') != table
                    or proof.get('session_id') != session or str(proof.get('hand_id')) != number
                    or proof.get('canonical_actor',proof.get('actor')) != actor):
                raise ValueError('CURRENT_WHO_NOT_PROVEN')
            low,high = proof['identity_interval']
            if min(frames) < low or high is not None and max(frames) > high:
                raise ValueError('STALE_WHO_INTERVAL')
        if old.get('seat') != fresh.get('seat'):
            raise ValueError('CURRENT_WHO_SEAT_CONFLICT')
        anchor = fresh['anchor']
        checkpoints = [c for c in self.checkpoints if c['table_id']==table
                       and c['session_id']==session and c['frame']<=anchor['frame']]
        if not checkpoints or max(checkpoints,key=lambda c:c['frame'])['hand_id'] != number:
            raise ValueError('CURRENT_WHO_ANCHOR_HEADER_NOT_PROVEN')
        native = self.nick_rows[anchor['source_line']]
        if (Path(anchor['source_file']).resolve() != (self.source/'seat_forced_inputs.jsonl').resolve()
                or native['table_id'] != table or native['session_id'] != session
                or native['hand_id'] != number or native['seat'] != fresh['seat']
                or native['source_frame'] != anchor['frame']):
            raise ValueError('NATIVE_WHO_CONTEXT_CONFLICT')
        stamp, _ = validate_native_nick(native)
        clock = dict(self.clocks[(table,session)])
        prior = clock.get(anchor['frame'])
        if prior is not None and prior != stamp:
            raise ValueError('NATIVE_WHO_FRAME_CLOCK_CONFLICT')
        clock[anchor['frame']] = stamp
        ordered = [clock[f] for f in sorted(clock)]
        if ordered != sorted(set(ordered)):
            raise ValueError('NATIVE_WHO_CLOCK_ORDER_CONFLICT')
        response = native['ocr_response']
        values = dict(raw_text=response.get('raw_text'),confidence=response.get('score'),
            ocr_status=response.get('status'),item_id=response.get('item_id'),
            pixel_sha256=native['pixel_sha256'],source_pts=native['source_pts'],
            time_base=native['time_base'],roi=native['roi'])
        if any(anchor.get(k) != value for k,value in values.items()) or response.get('status') != 'OK':
            raise ValueError('NATIVE_WHO_ANCHOR_CHANGED')
        return copy.deepcopy(fresh)

    def authorize(self, hand, *, players, nets, payouts, board, hole, ledger_known, math):
        """Grant comparison exemption only inside one exact native main-pot result."""
        try:
            self.last_rejection = None
            live = {n for n,p in players.items() if p['state'] != 'FOLDED'}
            missing = {n for n in live if len(hole.get(n,[])) != 2}
            if not missing:
                return None
            if (not ledger_known or not math['pass'] or hand.get('excluded_message_indices')
                    or not hand.get('start_observed') or not hand.get('end_observed')
                    or hand.get('boundary_conflicts') or len(board) != 5 or len(set(board)) != 5
                    or len(payouts) != 1):
                return self._reject('RESULT_ACCOUNTING_OR_BOARD_NOT_PROVEN')
            winner,paid = next(iter(payouts.items()))
            if winner not in live or winner in missing or paid <= 0 or any(n not in players for n in missing):
                return self._reject('KNOWN_WINNER_REQUIRED')
            if any(p['state'] == 'ALL_IN' for p in players.values()):
                return self._reject('ALL_IN_POT_OWNERSHIP_OUTSIDE_NARROW_CONTRACT')
            layers=[]; previous=Decimal(0)
            for level in sorted({n for n in nets.values() if n>0}):
                contributors={n for n,value in nets.items() if value>=level}
                eligible=contributors & live
                if len(contributors)<2 or eligible != live or winner not in eligible:
                    return self._reject('SIDE_POT_OWNERSHIP_NOT_UNIQUE_MAIN_POT')
                layers.append(dict(amount=str((level-previous)*len(contributors)),eligible=sorted(eligible)))
                previous=level
            if not layers or sum((Decimal(p['amount']) for p in layers),Decimal(0)) != paid:
                return self._reject('EXACT_MAIN_POT_AMOUNT_NOT_PROVEN')
            winner_rank=best_rank(board+hole[winner])
            for actor,cards in hole.items():
                if actor in live and actor!=winner and best_rank(board+cards)>=winner_rank:
                    return self._reject('KNOWN_OPPONENT_CONTRADICTS_UNIQUE_WINNER')
            self._load()
            messages=hand['messages']; allrows=[r for m in messages for r in m['source_rows']]
            positions=[(r['frame_id'],r['row_index']) for r in allrows]
            if positions != sorted(set(positions)):
                return self._reject('HAND_SOURCE_ORDER_INVALID')
            table,session,number=hand_key(hand)
            # Assembly may stop at the first WIN. A failed suffix can still
            # hide a competing payout until the next independently bound hand.
            next_hand = min((c['frame'] for c in self.checkpoints
                if c['table_id']==table and c['session_id']==session
                and c['frame']>positions[0][0] and c['hand_id'] not in (None,number)),
                default=float('inf'))
            if any(f['table_id']==table and f['session_id']==session
                   and positions[0][0]<=f['frame']<next_hand
                   for f in self.frame_failures):
                return self._reject('FAILED_CURRENT_HAND_SOURCE_FRAME')
            wins=[m for m in messages if m['action']=='WIN']
            if len(wins)!=1 or wins[0]['player']!=winner or Decimal(wins[0]['amount'])!=paid:
                return self._reject('CONFLICTING_NATIVE_PAYOUT')
            win=wins[0]
            if not re.search(r' from the main pot with ',win['raw_text'],re.I):
                return self._reject('EXPLICIT_MAIN_POT_RESULT_REQUIRED')
            win_refs=self._message(win,hand)
            aliases=self.actor_provider.actor_map(hand)
            native_wins=[]
            for (t,s,frame,rid),(row,_,_) in self.native.items():
                if t!=f'T{table:02d}' or s!=session or self.owners.get((table,s,frame,rid))!=number:
                    continue
                text=row['raw_ocr_text'].strip()
                if _WIN.match(text): native_wins.append(rid)
                show=_SHOW.match(text)
                if show and aliases.get(show[1],show[1]) in missing:
                    return self._reject('NATIVE_SHOW_CONFLICTS_WITH_HIDDEN_STATE')
            if native_wins != [win['source_rows'][0]['stable_row_id']]:
                return self._reject('NATIVE_PAYOUT_SET_NOT_EXACT')
            timeline=copy.deepcopy(allrows); visual=copy.deepcopy(self.visual)
            associate_visual_evidence(timeline,visual)
            project_chat_card_events(timeline,visual)
            byid={r['stable_row_id']:r for r in timeline}
            for message in messages:
                if message['action'] not in {'FLOP','TURN','RIVER','SHOW'}: continue
                self._message(message,hand)
                rebuilt=make_message([byid[r['stable_row_id']] for r in message['source_rows']],
                    require_pixel_cards=True,pixel_sources=byid)
                if rebuilt.get('cards') != message.get('cards') or not rebuilt.get('cards'):
                    return self._reject('CURRENT_NATIVE_CARD_PROJECTION_NOT_REPRODUCIBLE')
                head=message['source_rows'][0]
                key=head.get('card_row_key')
                reading=visual[key]
                expected={'FLOP':3,'TURN':4,'RIVER':5,'SHOW':2}[message['action']]
                trigger_key=f"{table}:{head['frame_id']}:{head['row_index']}"
                if (reading.get('projected_card_count') != expected
                        or reading.get('trigger_row_key') != trigger_key
                        or reading.get('cards') != rebuilt['cards']
                        or reading['hybrid'].get('cards') != rebuilt['cards']
                        or reading['hybrid'].get('reason') is not None
                        or reading.get('row_id') not in message['source_row_ids']):
                    return self._reject('FRESH_NATIVE_CARD_PROJECTION_REQUIRED')
                _frozen(reading)
                _png(reading['crop_png_base64'],reading['crop_sha256'],reading['row_pixel_sha256'])
            win_index=allrows.index(win['source_rows'][0])
            native_win=verified_win_tail(win['source_rows'],context=current_win_context(allrows,win_index))
            if (native_win is None or native_win['cards'] != win.get('win_best_five_cards')
                    or not set(native_win['cards']).issubset(set(board+hole[winner]))
                    or best_rank(native_win['cards']) != winner_rank
                    or Decimal(native_win['amount']) != paid
                    or aliases.get(native_win['actor'],native_win['actor']) != winner):
                return self._reject('NATIVE_WIN_BEST_FIVE_NOT_PROVEN')
            river_index=max(i for i,m in enumerate(messages) if m['action']=='RIVER')
            win_index=messages.index(win)
            proofs={}
            for actor in sorted(missing):
                hidden=[(i,m) for i,m in enumerate(messages) if m['action']=='NO_SHOW' and m.get('player')==actor]
                if len(hidden)!=1 or actor==winner or any(m['action']=='SHOW' and m.get('player')==actor for m in messages):
                    return self._reject('EXACT_EXPLICIT_NO_SHOW_REQUIRED')
                index,message=hidden[0]
                if not river_index<index<win_index:
                    return self._reject('NO_SHOW_OUTSIDE_SHOWDOWN_CONTEXT')
                refs=self._message(message,hand)
                # WHO binds this native hidden-card observation. Earlier forced
                # posts do not require continuous seat occupancy authority here.
                frames=[r['frame_id'] for r in message['source_rows']]
                proofs[actor]=dict(comparison=COMPARISON,hole_cards=None,
                    native_no_show=refs,who=self._who(actor,hand,frames))
            winner_frames=[r['frame_id'] for m in messages
                           if m.get('player')==winner and m['action'] in {'SHOW','WIN'}
                           for r in m['source_rows']]
            winner_who=self._who(winner,hand,winner_frames)
            self._load()  # Reject parents changed while proving this capability.
            return dict(rule=RULE,table_id=table,session_id=session,hand_id=number,
                hidden_losers=proofs,winner=winner,winner_who=winner_who,payout=str(paid),
                native_win=win_refs,pot_layers=layers,source_sha256=dict(self.hashes),
                scope='ONLY_INDEPENDENT_COMPARISON_FOR_EXPLICIT_HIDDEN_LOSERS',
                hole_cards_authority=False,WHO_authority=False,START_STACK_authority=False,
                board_authority=False,pot_authority=False,side_pot_authority=False)
        except (KeyError,ValueError,TypeError,IndexError,AttributeError,OSError,ArithmeticError,cv2.error) as exc:
            return self._reject(type(exc).__name__+':'+str(exc))
