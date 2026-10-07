"""Causal relationship evidence only. Never merge/edit source rows or poker gates."""
from __future__ import annotations
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import numpy as np
from parser_core.chat_rows import (align_upward, canonical_row, match_row_blocks,
    match_rows, padded_row_span, physical_spans, row_distance)

SAME = 'SAME_PHYSICAL_TRACK_PROVEN'
DIFFERENT = 'DIFFERENT_PHYSICAL_TRACK_PROVEN'
AMBIGUOUS = 'AMBIGUOUS'


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: str
    table_id: str
    session_id: str
    first_frame: int
    last_frame: int
    first_ms: float | None
    last_ms: float | None
    row_index: int
    y1: int
    pixel_hash: str
    raw_text: str
    normalized_text: str
    confidence: float | None
    visible_spans: tuple[tuple[int,int], ...]
    source_frame: int

    def __post_init__(self):
        object.__setattr__(self,'visible_spans',tuple(tuple(span) for span in self.visible_spans))


@dataclass(frozen=True)
class MotionProof:
    evidence_frames: tuple[int,...]
    before_hash: str
    after_hash: str
    aligned: bool
    shift_px: int
    alignment_score: float | None
    target_new_index: int | None
    candidate_new_index: int
    target_unique: bool
    exact_target_content: bool
    neighbor_anchors: tuple[tuple[int,int,str,str], ...]
    pixel_similarity: float | None
    old_y: int
    new_y: int
    local_loss_bridge: bool = False
    newly_exposed: bool = False

    def __post_init__(self):
        object.__setattr__(self,'evidence_frames',tuple(self.evidence_frames))
        object.__setattr__(self,'neighbor_anchors',tuple(tuple(a) for a in self.neighbor_anchors))


@dataclass(frozen=True)
class LinkResult:
    old_track_id: str
    new_track_id: str
    link_status: str
    link_reason: str
    decision_frame: int
    evidence_frames: tuple[int,...]


def digest(pixels):
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def _content(gray, span):
    # Same scrollbar exclusion as registered row matching; original stays intact.
    row = gray[span[0]:span[1]]
    return row[:,:-12] if row.shape[1]>=50 else row


def _exact(a,sa,b,sb):
    x,y=_content(a,sa),_content(b,sb)
    return x.shape==y.shape and np.array_equal(x,y)


def motion_proof(before, after, old_index, new_index, before_ids, after_ids,
                 before_frame, after_frame, *, bridge=None, bridge_frame=None):
    """Reuse registered alignment, row assignment and contiguous block logic.

    At least two exact neighbor anchors accompany an exact target raster.
    Similarity alone is diagnostic; thresholds are existing tracker thresholds.
    """
    raw_before,raw_after=digest(before),digest(after)
    old_all,new_all=physical_spans(before),physical_spans(after)
    old_spans,new_spans=old_all[1:],new_all[1:]
    if len(before_ids)!=len(old_spans) or len(after_ids)!=len(new_spans):
        raise ValueError('Viewport spans/track inventory mismatch')
    if not (0<=old_index<len(old_spans) and 0<=new_index<len(new_spans)):
        raise ValueError('Target not physically visible')
    alignment=align_upward(before,after)
    matches=match_rows(before,old_spans,after,new_spans,alignment.shift_px) if alignment.proven else {}
    blocks=match_row_blocks([canonical_row(before,s) for s in old_spans],
                           [canonical_row(after,s) for s in new_spans])
    mapped=next((n for n,o in matches.items() if o==old_index),None)
    anchors=[]
    if mapped is not None:
        for block in blocks:
            if block.mapping.get(mapped)!=old_index or block.length<3:
                continue
            for n,o in block.mapping.items():
                if o==old_index or matches.get(n)!=o:
                    continue
                if _exact(before,old_spans[o],after,new_spans[n]) and abs(
                    sum(old_spans[o])/2-sum(new_spans[n])/2-alignment.shift_px)<=1:
                    anchors.append((o,n,before_ids[o],after_ids[n]))
    exact=mapped is not None and _exact(before,old_spans[old_index],after,new_spans[mapped])
    # No second exact target raster in either viewport: repeated boilerplate
    # cannot establish a unique correspondence even inside a matching block.
    unique=False
    if mapped is not None and exact:
        possible_new=[i for i,s in enumerate(new_spans) if _exact(before,old_spans[old_index],after,s)]
        possible_old=[i for i,s in enumerate(old_spans) if _exact(before,s,after,new_spans[mapped])]
        unique=possible_new==[mapped] and possible_old==[old_index]
    a=padded_row_span(old_all,old_index+1,before.shape[0])
    b=padded_row_span(new_all,new_index+1,after.shape[0])
    x,y=before[a[0]:a[1]],after[b[0]:b[1]]
    similarity=1-float(np.mean(np.abs(x.astype(np.int16)-y.astype(np.int16))))/255 if x.shape==y.shape else None
    loss=False
    if bridge is not None and bridge_frame==before_frame+1==after_frame-1 and before.shape==after.shape==bridge.shape:
        # One-frame local raster loss; unchanged whole context outside the
        # target, with restored target and no viewport shift/append.
        mask=np.ones(before.shape[0],bool)
        mask[a[0]:a[1]]=False
        loss=(alignment.shift_px==0 and exact and np.array_equal(before[mask],after[mask]) and
              np.array_equal(before[mask],bridge[mask]) and
              np.ptp(bridge[a[0]:a[1],:-12])==0 and
              not np.array_equal(before[a[0]:a[1]],bridge[a[0]:a[1]]))
    newly=bool(alignment.proven and new_index not in matches and (
        alignment.shift_px>0 and new_spans[new_index][0]>=after.shape[0]-alignment.shift_px-8 or
        alignment.shift_px==0 and old_spans and new_spans[new_index][0]>max(s[1] for s in old_spans)+5))
    assert digest(before)==raw_before and digest(after)==raw_after
    frames=(before_frame,after_frame) if bridge is None else (before_frame,bridge_frame,after_frame)
    return MotionProof(tuple(frames),raw_before,raw_after,alignment.proven,alignment.shift_px,
        alignment.score if np.isfinite(alignment.score) else None,mapped,new_index,unique,exact,
        tuple(sorted(set(anchors))),similarity,a[0],b[0],bool(loss),newly)


def overlap_at(old,new,decision_frame):
    return any(max(a,c)<=min(b,d,decision_frame) for a,b in old.visible_spans
               for c,d in new.visible_spans if a<=decision_frame and c<=decision_frame)


def decide(old,new,proof,decision_frame, *, boundary_frames=(), simultaneous_positions=None,
           onset_proof=None):
    frames=[old.source_frame,new.source_frame,old.first_frame,old.last_frame,new.first_frame,new.last_frame]
    frames.extend(f for s in old.visible_spans+new.visible_spans for f in s)
    if proof:
        frames.extend(proof.evidence_frames)
    if onset_proof:
        frames.extend(onset_proof.evidence_frames)
    frames.extend(boundary_frames)
    if any(f is None or f>decision_frame for f in frames):
        return LinkResult(old.track_id,new.track_id,AMBIGUOUS,'FUTURE_FRAME_EVIDENCE_REJECTED',decision_frame,())
    used=tuple(sorted(set(frames)))
    def result(label,reason):
        return LinkResult(old.track_id,new.track_id,label,reason,decision_frame,used)
    if proof and (proof.evidence_frames[0]!=old.last_frame or proof.evidence_frames[-1]!=new.first_frame or
                  proof.old_y!=old.y1 or proof.new_y!=new.y1 or proof.candidate_new_index!=new.row_index-1):
        return result(AMBIGUOUS,'MOTION_ENDPOINTS_DO_NOT_MATCH_TRACK_SNAPSHOTS')
    if any(not(old.last_frame<f<=new.first_frame) for f in boundary_frames):
        return result(AMBIGUOUS,'BOUNDARY_OUTSIDE_TRACK_SEPARATION_INTERVAL')
    if onset_proof and (onset_proof.evidence_frames[0]!=new.first_frame-1 or
        onset_proof.evidence_frames[-1]!=new.first_frame or onset_proof.new_y!=new.y1 or
        onset_proof.candidate_new_index!=new.row_index-1):
        return result(AMBIGUOUS,'ONSET_ENDPOINTS_DO_NOT_MATCH_NEW_TRACK')
    if (old.table_id,old.session_id)!=(new.table_id,new.session_id):
        return result(DIFFERENT,'DIFFERENT_TABLE_OR_SESSION')
    if simultaneous_positions is not None and simultaneous_positions[0]!=simultaneous_positions[1] and overlap_at(old,new,decision_frame):
        return result(DIFFERENT,'SIMULTANEOUS_DISTINCT_PHYSICAL_ROW_POSITIONS')
    if overlap_at(old,new,decision_frame):
        return result(AMBIGUOUS,'OVERLAP_WITHOUT_POSITION_WITNESS')
    strong=bool(proof and proof.aligned and proof.target_unique and proof.exact_target_content and len(proof.neighbor_anchors)>=2)
    gap=new.first_frame-old.last_frame
    adjacent=gap==1 or gap==2 and proof is not None and proof.local_loss_bridge
    if strong and adjacent and proof.target_new_index!=proof.candidate_new_index:
        return result(DIFFERENT,'OLD_ROW_CONTINUES_AT_ANOTHER_POSITION_WITH_TWO_NEIGHBOR_ANCHORS')
    if (onset_proof and onset_proof.newly_exposed and onset_proof.aligned and
        len(onset_proof.neighbor_anchors)>=2 and boundary_frames and gap>2):
        return result(DIFFERENT,'CONFIRMED_BOUNDARY_AND_PROVEN_NEW_BOTTOM_ONSET')
    if boundary_frames:
        return result(AMBIGUOUS,'BOUNDARY_BLOCKS_SAME_WITHOUT_PHYSICAL_ONSET_PROOF')
    text_same=bool(old.normalized_text and old.normalized_text==new.normalized_text)
    if strong and text_same and (gap==1 or gap==2 and proof.local_loss_bridge):
        return result(SAME,'UNIQUE_EXACT_ROW_CONTINUITY_WITH_TWO_NEIGHBORS_AND_CAUSAL_VIEWPORT_MOTION')
    return result(AMBIGUOUS,'INSUFFICIENT_COMBINED_PHYSICAL_CONTINUITY')


def text_similarity(old,new):
    return SequenceMatcher(None,old.normalized_text,new.normalized_text).ratio()
