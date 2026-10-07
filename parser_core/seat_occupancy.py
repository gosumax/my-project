"""Independent fixed-seat occupancy from exact, source-continuous label evidence.

This establishes a current seat mask only. It grants no past actor/epoch binding,
forced-debit authority, clean action interval, or START_STACK authority.
"""
import re

OVERLAYS = {'ANTE', 'SB', 'BB', 'FOLD', 'CHECK', 'CALL', 'BET', 'RAISE',
            'ALL-IN', 'ALL IN', 'SITOUT', 'SIT OUT'}
OVERLAY_PATTERN = re.compile(r'(?:(?:ANTE|SB|BB|FOLD|CHECK|CALL|BET|RAISE)(?:\s+\d[\d\s]*)?'
                             r'|SIT[\s-]*OUT|ALL[\s-]*IN)[.!●]?', re.I)


class SeatOccupancyEvidence:
    def __init__(self, confidence=.95, *, reject_empty_fragments=False):
        self.confidence = confidence
        self.reject_empty_fragments = reject_empty_fragments
        self.sources = {}
        self.readings = {}
        self.proofs = {}
        self.conflicts = set()

    def observe(self, key, seat, frame, pixel_sha256):
        route = (*key, seat)
        previous = self.sources.get(route)
        continuous = previous and previous['frame'] == frame - 1 and previous['sha256'] == pixel_sha256
        self.sources[route] = dict(frame=frame, sha256=pixel_sha256,
                                   first=previous['first'] if continuous else frame)
        if not continuous:
            self.proofs.pop(route, None)
            self.conflicts.discard(route)

    def completed(self, key, seat, frame, pixel_sha256, response, *, pts, time_base):
        route = (*key, seat)
        latest = self.sources.get(route)
        if not latest or frame < latest['first'] or pixel_sha256 != latest['sha256']:
            return
        text = (response.get('raw_text') or '').strip()
        score = response.get('score')
        empty = text.upper() == 'OPEN SEAT'
        literal = bool(text and text.upper() not in OVERLAYS and not OVERLAY_PATTERN.fullmatch(text)
                       and not (self.reject_empty_fragments and text.upper() in {'PENSEAT','PEN SEAT'})
                       and '...' not in text and '…' not in text
                       and not any(s in text for s in ('\ufffd', '?'))
                       and re.search(r'[A-Za-zА-Яа-я]', text))
        valid = (response.get('status') == 'OK' and isinstance(score, (int, float))
                 and not isinstance(score, bool) and score >= self.confidence and (empty or literal))
        old = self.readings.get(route, [])
        if valid and any(r['pixel_sha256'] == pixel_sha256 and r['raw_text'] != text
                         and r['frame'] >= latest['first'] for r in old):
            self.conflicts.add(route)
        if route in self.conflicts:
            self.proofs.pop(route, None)
            return
        observation = dict(frame=frame, source_pts=pts, time_base=list(time_base),
                           pixel_sha256=pixel_sha256, raw_text=text, confidence=score)
        contiguous = old and old[-1]['frame'] == frame-1 and old[-1]['raw_text'] == text
        readings = (old + [observation])[-3:] if contiguous and valid else [observation] if valid else []
        self.readings[route] = readings
        if (len(readings) == 3 and latest and latest['first'] <= readings[0]['frame']
                and latest['sha256'] == pixel_sha256
                and all(r['pixel_sha256'] == pixel_sha256 for r in readings)):
            self.proofs[route] = dict(state='EMPTY' if empty else 'OCCUPIED',
                                      actor=None if empty else text, observations=list(readings))
        elif not valid:
            self.proofs.pop(route, None)

    def receipt(self, key, frame, seats):
        evidence = {}
        for seat in seats:
            route = (*key, seat)
            latest, proof = self.sources.get(route), self.proofs.get(route)
            if (not latest or not proof or latest['frame'] != frame
                    or latest['first'] > proof['observations'][0]['frame']):
                return None
            evidence[seat] = dict(proof, current_source_frame=frame,
                                  current_pixel_sha256=latest['sha256'],
                                  exact_pixel_continuity_from=latest['first'])
        occupied = [s for s,r in evidence.items() if r['state'] == 'OCCUPIED']
        actors = [evidence[s]['actor'] for s in occupied]
        if len(occupied) < 2 or len(actors) != len(set(actors)):
            return None
        return dict(table_id=key[0], session_id=key[1], hand_id=key[2], source_frame=frame,
                    occupied_seats=occupied, empty_seats=[s for s in seats if s not in occupied],
                    seats=evidence, evidence_status='PROVEN',
                    proof='ALL_FIXED_SEATS_EXACT_LABEL_TRIPLES_AND_CONTINUOUS_IDENTICAL_PIXELS',
                    authority='CURRENT_OCCUPANCY_ONLY; NO_HISTORICAL_IDENTITY_OR_STACK_AUTHORITY')

    def current_labels(self, key, frame, seats):
        """Only currently continuous label proofs, including partial seat sets."""
        result={}
        for seat in seats:
            route=(*key,seat);latest=self.sources.get(route);proof=self.proofs.get(route)
            if (latest and proof and latest['frame']==frame and
                    latest['first']<=proof['observations'][0]['frame']):
                result[seat]=dict(proof,table_id=key[0],session_id=key[1],hand_id=key[2],
                    source_frame=frame,current_pixel_sha256=latest['sha256'])
        return result
