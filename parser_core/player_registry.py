"""Human-confirmed OCR spellings. Similarity never writes or resolves aliases."""
import copy
import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from .nickname_policy import NICK_CONFIDENCE_MIN

DEFAULT_DATABASE = Path(__file__).resolve().parents[1] / 'data' / 'players.sqlite'
AUTHORITY = 'USER_CONFIRMED_PLAYER_REGISTRY'
NATIVE_NICK_CONFIRMATION_MIN = NICK_CONFIDENCE_MIN


class CurrentHandReviewAssociation(dict):
    """In-process capability created only after current manual source reproof."""


def actor_observation_has_name(row, actor):
    text = row.get('raw_ocr_text')
    if not isinstance(text, str):
        return False
    prefix = re.match(r'^(?:Dealer|Дилер)\s*:\s*', text, re.I)
    if prefix is not None:
        return re.match(re.escape(actor) + r'\s', text[prefix.end():]) is not None
    # An OCR service delimiter can have become part of the literal parsed actor.
    # This checks that exact original actor only; no semantic rewrite is made.
    return bool(re.match(r'^(?:Dealer|Дилер)\s*;\s*\S', actor, re.I) and
                re.match(re.escape(actor) + r'\s', text))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def _name(value):
    if (not isinstance(value, str) or not value.strip() or len(value) > 100 or
            any(ord(c) < 32 for c in value)):
        raise ValueError('Введите непустой ник длиной до 100 символов')
    return value.strip()


def hand_nick_conflicts(lineage,route,bounds,aliases,registry,canonical=None,other_actors=()):
    """Separate a competing player from uncertain OCR once at HH assembly.

    Reuse the existing lineage admission and its two-frame replacement rule.
    Unmatched low-confidence OCR remains raw UNKNOWN evidence, not a player.
    """
    from .simple_seat_who import nick_constraint,nick_matches
    low,high=bounds;seen=set();uncertain=[];strong={}
    for segment in lineage.segments[route]:
        for row in segment['observations']+segment['ignored']:
            if id(row) in seen or not low<=row['frame']<=high:continue
            seen.add(id(row));label=nick_constraint(row['raw_text'])
            if not label:continue
            if any(nick_matches(label,actor) for actor in other_actors):
                return 'CURRENT_REVIEW_OTHER_OBSERVED_PLAYER',[row],uncertain
            if row['raw_text'] in aliases:continue
            known=registry.resolve(row['raw_text'],'NICK')
            if known:
                if canonical is None or known['canonical']!=canonical:
                    return 'CURRENT_REVIEW_OTHER_REGISTERED_PLAYER',[row],uncertain
                continue
            if registry.aliases.get(('NICK',row['raw_text'])):
                return 'CURRENT_REVIEW_AMBIGUOUS_REGISTERED_PLAYER',[row],uncertain
            confidence=row.get('confidence')
            if (label['kind']=='EXACT' and type(confidence) in (int,float) and math.isfinite(confidence) and
                NATIVE_NICK_CONFIRMATION_MIN<=confidence<=1):
                strong.setdefault(label['text'],[]).append(row)
            else:
                uncertain.append(row)
    for rows in strong.values():
        if len({row['frame'] for row in rows})>=2:
            return 'CURRENT_REVIEW_UNSELECTED_STABLE_NICK',rows,uncertain
        uncertain.extend(rows)
    return None,[],uncertain


class PlayerRegistry:
    """Read-only snapshot by default; only an explicit review creates the DB."""
    def __init__(self, path=DEFAULT_DATABASE):
        self.path = Path(path).resolve()
        self.aliases = {}
        self.players = {}
        self.reviews = {}
        if self.path.exists():
            with sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True) as conn:
                conn.row_factory = sqlite3.Row
                if conn.execute('PRAGMA user_version').fetchone()[0] != 1:
                    raise ValueError('Unsupported player registry schema')
                self.players = {r['id']: r['canonical'] for r in conn.execute('SELECT * FROM players')}
                self.reviews = {r['id']: json.loads(r['record']) for r in conn.execute('SELECT * FROM reviews')}
                superseded = {rid for r in self.reviews.values() for rid in r.get('supersedes', ())}
                self.active_reviews = {rid:r for rid,r in self.reviews.items() if rid not in superseded}
                for r in conn.execute('SELECT * FROM aliases ORDER BY id'):
                    if r['review_id'] in superseded:
                        continue
                    self.aliases.setdefault((r['kind'], r['raw']), {}).setdefault(r['player_id'], []).append(r['review_id'])
        if not hasattr(self, 'active_reviews'):
            self.active_reviews = {}
        self.revision = digest(dict(players=self.players, reviews=self.reviews))

    def resolve(self, raw, kind):
        candidates = self.aliases.get((kind, raw), {})
        if len(candidates) != 1:
            return None
        player = next(iter(candidates))
        return dict(player_id=player, canonical=self.players[player],
                    review_ids=sorted(set(candidates[player])), registry_revision=self.revision)

    def confirm(self, case, canonical, selected, *, source_review_id=None, restrict_ui_aliases=False):
        """Called only by the local review UI after source/case verification."""
        canonical = _name(canonical)
        choices = {r['id']: r for r in case['observations']}
        if (not isinstance(selected, list) or len(selected) != len(set(selected)) or
                any(key not in choices for key in selected)):
            raise ValueError('Invalid review observation selection')
        explicit=list(selected)
        canonical_ids=[o['id'] for o in case['observations'] if o['text']==canonical]
        observations = [o for o in case['observations'] if o['id'] in set(explicit+canonical_ids)]
        if not any(r['kind'] == 'NICK' for r in observations):
            raise ValueError('Подтвердите хотя бы один NICK-кроп')
        if any(r.get('bound_other_seat') for r in observations):
            raise ValueError('CHAT actor already belongs to another physical seat')
        prior=None
        if source_review_id is not None:
            prior=self.active_reviews.get(source_review_id)
            if (prior is None or prior['canonical']!=canonical or prior['case_id']!=case['case_id'] or
                prior['context']!=case['context'] or prior['source_pins']!=case['source_pins'] or
                not (set(explicit)<=set(prior.get('selected_explicit',[o['id'] for o in prior['observations']]))
                     if restrict_ui_aliases else
                     set(explicit)==set(prior.get('selected_explicit',[o['id'] for o in prior['observations']])))):
                raise ValueError('Canonical-source adaptation must retain the original human decision')
        elif restrict_ui_aliases:
            raise ValueError('UI alias restriction requires the original human review')
        record = dict(authority=AUTHORITY, case_id=case['case_id'], canonical=canonical,
                      context=case['context'], identity_interval=case['identity_interval'],
                      review_scope=case.get('review_scope','INTERVAL'),
                      review_scope_key=case.get('review_scope_key'),hand_bounds=case.get('hand_bounds'),
                      binding_blockers=case.get('binding_blockers',[]),
                      source_pins=case['source_pins'],
                      selected_explicit=explicit,canonical_observations=canonical_ids,
                      canonical_sources_derived_from_review=source_review_id,
                      ui_alias_scope_restricted=restrict_ui_aliases,
                      original_selected_explicit=(prior.get('selected_explicit',[o['id'] for o in prior['observations']])
                                                  if restrict_ui_aliases else None),
                      human_confirmed_at=(prior.get('human_confirmed_at') or prior['confirmed_at']) if prior else None,
                      supersedes=[rid for rid, r in self.reviews.items() if r['case_id'] == case['case_id'] or
                          case.get('review_scope_key') is not None and r.get('review_scope_key')==case['review_scope_key']],
                      observations=[{k: v for k, v in r.items() if k != 'image'} for r in observations],
                      confirmed_at=datetime.now(timezone.utc).isoformat())
        review_id = digest(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            version = conn.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise ValueError('Unsupported player registry schema')
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS players(id INTEGER PRIMARY KEY, canonical TEXT NOT NULL UNIQUE);
                CREATE TABLE IF NOT EXISTS reviews(id TEXT PRIMARY KEY, record TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS aliases(id INTEGER PRIMARY KEY, player_id INTEGER NOT NULL,
                    kind TEXT NOT NULL, raw TEXT NOT NULL, review_id TEXT NOT NULL,
                    UNIQUE(player_id,kind,raw,review_id));
                PRAGMA user_version=1;
            ''')
            conn.execute('INSERT OR IGNORE INTO players(canonical) VALUES (?)', (canonical,))
            player = conn.execute('SELECT id FROM players WHERE canonical=?', (canonical,)).fetchone()[0]
            conn.execute('INSERT INTO reviews VALUES (?,?)',
                         (review_id, json.dumps(record, ensure_ascii=False, sort_keys=True)))
            labels = {(r['kind'], r['text']) for r in observations} | {('NICK', canonical), ('CHAT', canonical)}
            conn.executemany('INSERT OR IGNORE INTO aliases(player_id,kind,raw,review_id) VALUES (?,?,?,?)',
                             [(player, kind, raw, review_id) for kind, raw in sorted(labels)])
        fresh = PlayerRegistry(self.path)
        conflicts = [dict(kind=kind, raw=raw) for kind, raw in sorted(labels)
                     if fresh.resolve(raw, kind) is None]
        return dict(review_id=review_id, canonical=canonical, conflicts=conflicts,
                    database=str(self.path), revision=fresh.revision)

    def _current_review(self, lineage, hand, bounds, source, review_id, record, pin_cache):
        """Reprove a human's explicit whole-hand spelling decision, not similarity."""
        from .simple_seat_who import nick_matches, nick_observation_constraint
        table=int(str(hand['table_id']).removeprefix('T'))
        key=(table,hand['session_id'],str(hand['hand_id']))
        context=record['context'];seat=context['seat'];route=(*key[:2],seat)
        low,high=bounds[key]
        def reject(reason):
            return None,None,dict(table_id=table,session_id=key[1],hand_id=key[2],seat=seat,
                                  reason=reason,review_id=review_id)
        if record.get('hand_bounds')!=[low,high] or record.get('identity_interval')!=[low,high]:
            return reject('CURRENT_REVIEW_HAND_BOUNDS_CHANGED')
        current_paths={name:str((source/name).resolve()) for name in
                       ('seat_forced_inputs.jsonl','physical_rows.jsonl')}
        reviewed_paths={Path(name).name:name for name in record['source_pins']}
        if (len(record['source_pins'])!=2 or set(reviewed_paths)!=set(current_paths) or
            len({Path(name).parent.resolve() for name in record['source_pins']})!=1):
            return reject('CURRENT_REVIEW_SOURCE_SET_CONFLICT')
        for name,sha in record['source_pins'].items():
            if name not in pin_cache:
                pin_cache[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
            if pin_cache[name]!=sha:
                return reject('CURRENT_REVIEW_SOURCE_CHANGED')
        rebound=set(record['source_pins'])!=set(current_paths.values())
        current_pins={}
        for name in current_paths.values():
            if name not in pin_cache:
                pin_cache[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
            current_pins[name]=pin_cache[name]
        # A new run directory is not a new human decision. Only the exact same
        # physical CHAT context can reuse it; selected native parents and every
        # current occupant/conflict guard below must still be proved again.
        if rebound and current_pins[current_paths['physical_rows.jsonl']]!=record['source_pins'][reviewed_paths['physical_rows.jsonl']]:
            return reject('CURRENT_REVIEW_SOURCE_CHANGED')
        selected_nicks=[o for o in record['observations'] if o['kind']=='NICK']
        selected_chats=[o for o in record['observations'] if o['kind']=='CHAT']
        if not selected_nicks or not selected_chats:
            return reject('CURRENT_REVIEW_NICK_OR_CHAT_MISSING')
        for observation in selected_nicks+selected_chats:
            identity=self.resolve(observation['text'],observation['kind'])
            if not identity or identity['canonical']!=record['canonical']:
                return reject('CURRENT_REVIEW_CONFLICTING_REGISTERED_PLAYER')
        anchors=[];rebound_nicks=[]
        for o in selected_nicks:
            ref=o['source']
            if ref['file']!=reviewed_paths['seat_forced_inputs.jsonl']:
                return reject('CURRENT_REVIEW_NICK_PARENT_CONFLICT')
            reviewed_line=ref['line']
            matches=[r for r in lineage.observations if
                (int(str(r['table_id']).removeprefix('T')),r['session_id'],r['seat'])==route and
                str(r.get('hand_id'))==key[2] and r.get('source_file')==current_paths['seat_forced_inputs.jsonl'] and
                (rebound or r.get('source_line')==reviewed_line) and r['frame']==ref['frame'] and
                r['raw_text']==o['text'] and r.get('pixel_sha256')==ref['pixel_sha256'] and
                r.get('roi')==ref['roi'] and r.get('source_pts')==ref.get('source_pts') and
                r.get('time_base')==ref.get('time_base')]
            if len(matches)!=1 or not low<=matches[0]['frame']<=high or nick_observation_constraint(matches[0]) is None:
                return reject('CURRENT_REVIEW_NICK_PARENT_CONFLICT')
            row=matches[0];segment=lineage.for_seat(*route,row['frame'])
            if segment is None or ref.get('native_interval')!=[segment['start'],segment['end']]:
                return reject('CURRENT_REVIEW_NATIVE_INTERVAL_CHANGED')
            anchors.append(row)
            if rebound:
                rebound_nicks.append(dict(reviewed_source=copy.deepcopy(ref),current_source={
                    k:row.get(k) for k in ('source_file','source_line','frame','raw_text',
                        'pixel_sha256','roi','source_pts','time_base')}))
        aliases={o['text'] for o in selected_chats}
        own=[m for m in hand['messages'] if (m.get('player') or m.get('actor')) in aliases]
        for o in selected_chats:
            ref=o['source']
            parents=[r for m in own if (m.get('player') or m.get('actor'))==o['text']
                for r in m.get('source_rows',()) if r.get('stable_row_id')==ref['row_id'] and
                r.get('frame_id')==ref['frame'] and r.get('row_pixel_sha256')==ref['pixel_sha256'] and
                r.get('raw_ocr_text')==ref['raw_text'] and actor_observation_has_name(r,o['text'])]
            if len(parents)!=1 or ref['file']!=reviewed_paths['physical_rows.jsonl']:
                return reject('CURRENT_REVIEW_CHAT_PARENT_CONFLICT')
        for m in own:
            if not m.get('source_rows') or any(
                    int(str(r.get('table_id',0)).removeprefix('T'))!=table or r.get('session_id')!=key[1] or
                    not isinstance(r.get('frame_id'),int) or not low<=r['frame_id']<=high
                    for r in m['source_rows']):
                return reject('CURRENT_REVIEW_CHAT_CONTEXT_CONFLICT')
        for role in ('ANTE','SMALL_BLIND','BIG_BLIND','FOLD'):
            if len({m.get('player') or m.get('actor') for m in own if m.get('action')==role})>1:
                return reject('CURRENT_REVIEW_MULTIPLE_ACTORS_SAME_SINGLE_ACTION')
        if any(low<=b['frame']<=high and b['kind']!='STABLE_NEW_NICK' for b in lineage.boundaries[route]):
            return reject('CURRENT_REVIEW_PHYSICAL_CLEAR_OR_REPLACEMENT')
        nick_aliases={o['text'] for o in selected_nicks}
        first=min(r['frame'] for r in anchors)
        other_actors={m.get('player') or m.get('actor') for m in hand['messages']} - aliases - {None}
        if other_actors & nick_aliases:
            return reject('CURRENT_REVIEW_OTHER_OBSERVED_PLAYER')
        reason,_,uncertain=hand_nick_conflicts(lineage,route,(low,high),nick_aliases,
            self,record['canonical'],other_actors)
        if reason:return reject(reason)
        for other_route,segments in lineage.segments.items():
            if other_route[:2]!=key[:2] or other_route==route:
                continue
            if any(s['start']<=high and (s['end'] is None or s['end']>=low) and
                   any(nick_matches(s['label'],a) for a in aliases|nick_aliases) for s in segments):
                return reject('CURRENT_REVIEW_ACTOR_AT_OTHER_SEAT')
        anchor=min(anchors,key=lambda r:r['frame'])
        native_intervals=[[s['start'],s['end']] for s in lineage.segments[route] if
                          s['start']<=high and (s['end'] is None or s['end']>=low)]
        evidence=dict(current_hand_review=review_id,authority='USER_CONFIRMED_CURRENT_HAND_SEAT',
            review_scope_key=record.get('review_scope_key'),source_pins=current_pins,
            registry_revision=self.revision,hand_bounds=[low,high],native_intervals=native_intervals,
            native_nick_sources=copy.deepcopy(anchors),
            unmatched_uncertain_nick_candidates=[{k:r.get(k) for k in
                ('raw_text','confidence','frame','source_file','source_line','pixel_sha256')}
                for r in uncertain],
            early_chat_before_first_nick=any(r['frame_id']<first for m in own for r in m['source_rows']),
            continuity_authority='EXPLICIT_CURRENT_HAND_HUMAN_REVIEW_NOT_PIXEL_CONTINUITY')
        if rebound:
            evidence['source_rebinding']=dict(rule='SAME_PHYSICAL_CHAT_EXACT_REVIEWED_NATIVE_PARENTS_V1',
                reviewed_source_pins=copy.deepcopy(record['source_pins']),current_source_pins=current_pins,
                physical_chat_byte_parity=True,native_parents=rebound_nicks)
        association=CurrentHandReviewAssociation(table_id=table,session_id=key[1],hand_id=key[2],seat=seat,
            aliases=sorted(aliases|nick_aliases),authority='USER_CONFIRMED_CURRENT_HAND_SEAT',
            anchor=anchor,hand_bounds=[low,high],registry_evidence=evidence)
        override=dict(table=table,session=key[1],hand=key[2],seat=seat,manual_confirmed=True,
            manual_ground_truth=record['canonical'],authority='USER_CONFIRMED_CURRENT_HAND_SEAT',registry_evidence=evidence)
        return association,override,None

    def hand_evidence(self, lineage, hands, bounds, *, source=None):
        """Reuse exact reviewed spellings through current fixed-seat evidence.

        No similarity, transitive merging, unknown alias, multiple interval or
        multiple-seat fallback. Existing lineage boundary semantics stay intact.
        """
        from .simple_seat_who import nick_matches, nick_observation_constraint
        associations, overrides, rejected = [], [], []
        source=Path(source).resolve() if source is not None else None
        pin_cache={}
        for hand in hands:
            table = int(str(hand['table_id']).removeprefix('T'))
            session, number = hand['session_id'], str(hand['hand_id'])
            low, high = bounds[(table, session, number)]
            messages = hand['messages']
            actors = {m.get('player') or m.get('actor') for m in messages} - {None}
            current=[]
            if source is not None:
                for review_id,record in self.active_reviews.items():
                    if record.get('review_scope')!='HAND_SEAT':
                        continue
                    context=record.get('context',{})
                    if (not isinstance(context,dict) or
                        (context.get('table_id'),context.get('session_id'),str(context.get('hand_id')))!=(table,session,number)):
                        continue
                    try:
                        a,m,error=self._current_review(lineage,hand,bounds,source,review_id,record,pin_cache)
                    except (KeyError,TypeError,ValueError,OSError):
                        a,m=None,None
                        error=dict(table_id=table,session_id=session,hand_id=number,
                            reason='CURRENT_REVIEW_SOURCE_INVALID',review_id=review_id)
                    if error:rejected.append(error)
                    else:current.append((a,m))
            handled=set()
            for a,m in current:
                if any(other is not a and (set(a['aliases'])&set(other['aliases']) or a['seat']==other['seat'])
                       for other,_ in current):
                    rejected.append(dict(table_id=table,session_id=session,hand_id=number,
                        reason='CURRENT_REVIEW_CONFLICTING_DECISIONS',aliases=a['aliases']))
                    continue
                associations.append(a);overrides.append(m);handled.update(a['aliases'])
            clusters = {}
            for actor in actors-handled:
                identity = self.resolve(actor, 'CHAT')
                if identity:
                    clusters.setdefault(identity['player_id'], []).append(actor)
            for player, aliases in clusters.items():
                intervals = []
                for route, segments in lineage.segments.items():
                    if route[:2] != (table, session):
                        continue
                    for segment in segments:
                        if segment['start'] > high or segment['end'] is not None and segment['end'] < low:
                            continue
                        current = [r for r in segment['observations'] if low <= r['frame'] <= high]
                        if any((self.resolve(r.get('raw_text'), 'NICK') or {}).get('player_id') == player
                               for r in current):
                            intervals.append((route, segment, current))
                reason = None
                if len(aliases) > 1:
                    reason = 'REGISTRY_MULTIPLE_CHAT_SPELLINGS_REQUIRE_CURRENT_REVIEW'
                if len(intervals) != 1:
                    reason = 'REGISTRY_MULTIPLE_OR_MISSING_PHYSICAL_INTERVALS'
                else:
                    route, segment, current = intervals[0]
                    overlapping = [s for s in lineage.segments[route] if s['start'] <= high and
                                   (s['end'] is None or s['end'] >= low)]
                    if len(overlapping) != 1 or any(
                            (self.resolve(r.get('raw_text'), 'NICK') or {}).get('player_id') != player
                            for r in current):
                        reason = 'REGISTRY_CONFLICTING_OCCUPANT'
                    if any(low <= r['frame'] <= high and nick_observation_constraint(r) is not None and
                           (self.resolve(r.get('raw_text'), 'NICK') or {}).get('player_id') != player
                           for r in segment['ignored']):
                        reason = 'REGISTRY_UNREVIEWED_NICK_ANOMALY'
                    own = [m for m in messages if (m.get('player') or m.get('actor')) in aliases]
                    if any(not m.get('source_rows') for m in own):
                        reason = 'REGISTRY_CHAT_SOURCE_MISSING'
                    frames = [r.get('frame_id') for m in own for r in m.get('source_rows', ())]
                    for message in own:
                        for row in message.get('source_rows', ()):
                            if (str(row.get('table_id')).removeprefix('T').lstrip('0') != str(table) or
                                    row.get('session_id') != session or not row.get('stable_row_id') or
                                    not isinstance(row.get('row_pixel_sha256'), str) or
                                    len(row['row_pixel_sha256']) != 64):
                                reason = 'REGISTRY_CHAT_SOURCE_CONTEXT_CONFLICT'
                    if not frames or any(not isinstance(f, int) or not low <= f <= high or
                            f < segment['start'] or segment['end'] is not None and f > segment['end']
                            for f in frames):
                        reason = 'REGISTRY_CHAT_OUTSIDE_PHYSICAL_INTERVAL'
                    if any(other_route[:2] == (table, session) and other_route != route and
                           any(s['start'] <= high and (s['end'] is None or s['end'] >= low) and
                               any(nick_matches(s['label'], a) for a in aliases) for s in segments)
                           for other_route, segments in lineage.segments.items()):
                        reason = 'REGISTRY_LITERAL_ACTOR_AT_OTHER_SEAT'
                    for role in ('ANTE', 'SMALL_BLIND', 'BIG_BLIND', 'FOLD'):
                        if len({m.get('player') or m.get('actor') for m in own if m.get('action') == role}) > 1:
                            reason = 'REGISTRY_MULTIPLE_ACTORS_SAME_SINGLE_ACTION'
                if reason:
                    rejected.append(dict(table_id=table, session_id=session, hand_id=number,
                                         aliases=sorted(aliases), reason=reason))
                    continue
                nick_aliases = sorted({r['raw_text'] for r in current})
                canonical = self.players[player]
                reviews = sorted({rid for kind, values in [('NICK', nick_aliases), ('CHAT', aliases)]
                                  for raw in values for rid in self.resolve(raw, kind)['review_ids']})
                receipt = dict(authority=AUTHORITY, database=str(self.path), registry_revision=self.revision,
                               player_id=player, review_ids=reviews,
                               identity_interval=[segment['start'], segment['end']],
                               current_nick_sources=copy.deepcopy(current), chat_source_frames=frames,
                               lineage_rule='UNCHANGED_EXISTING_FIXED_SEAT_INTERVAL')
                associations.append(dict(table_id=table, session_id=session, hand_id=number,
                    seat=route[2], aliases=sorted(set(aliases + nick_aliases)), authority=AUTHORITY,
                    registry_evidence=receipt))
                overrides.append(dict(table=table, session=session, hand=number, seat=route[2],
                    manual_ground_truth=canonical, manual_confirmed=True, authority=AUTHORITY,
                    registry_evidence=receipt))
        return associations, overrides, rejected
