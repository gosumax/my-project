"""Confidence spelling selection for an already associated fixed-seat occupant.

This module never infers associations by changing characters or edit distance.
Raw observations, automatic choice and confirmed manual truth are separate.
"""
import copy
import math

from .chat_nick_actor import literal_constraint, matches


def _confidence(row):
    value=row.get('confidence',row.get('raw_confidence',row.get('score')))
    return float(value) if isinstance(value,(int,float)) and math.isfinite(value) else None


def select_nickname(nick_observation, chat_observations=(), *, context=None, manual=None,
                    other_nick_observations=(), anomalous_nick_observations=(), associated_aliases=()):
    nick=copy.deepcopy(nick_observation)
    chats=copy.deepcopy(list(chat_observations))
    raw=nick.get('raw_text',nick.get('raw_nick_OCR'))
    candidates=[dict(text=raw,confidence=_confidence(nick),source='NICK',observation=nick)]
    candidates += [dict(text=r.get('raw_text',r.get('raw_nick_OCR')),confidence=_confidence(r),
                        source='NICK',observation=copy.deepcopy(r)) for r in other_nick_observations]
    candidates += [dict(text=r.get('raw_text',r.get('actor')),confidence=_confidence(r),
                        source='CHAT',observation=r) for r in chats]
    candidates=[r for r in candidates if isinstance(r['text'],str) and r['text'].strip()]
    label=literal_constraint(raw) if isinstance(raw,str) else None
    # Existing literal truncation rule expands a display prefix using full CHAT.
    associated=set(associated_aliases)
    expansion=[c for c in candidates if c['source']=='CHAT' and label and
               label['kind']=='PREFIX' and (matches(c['text'],label) or
               raw in associated and c['text'] in associated)]
    eligible=expansion or candidates
    texts={c['text'] for c in candidates}
    comparable=all(c['confidence'] is not None for c in eligible)
    best=max((c['confidence'] for c in eligible),default=None) if comparable else None
    winners=[c for c in eligible if c['confidence']==best] if comparable else []
    tied=len({c['text'] for c in winners})>1
    automatic=(eligible[0]['text'] if len({c['text'] for c in eligible})==1 else
               winners[0]['text'] if winners and not tied else None)
    reason=('MANUAL_REVIEW' if automatic is None else
            ('EXISTING_DOT_PREFIX' if all(matches(c['text'],label) for c in expansion)
             else 'CONFIRMED_TRUNCATED_CHAT') if expansion else
            'HIGHER_CONFIDENCE' if len(texts)>1 else 'EXACT_NICK_ROI')
    anomalies=copy.deepcopy(list(anomalous_nick_observations))
    expanded={c['text'] for c in expansion}
    resolved_prefix=len(expanded)==1 and all(c['text'] in expanded or
        c['source']=='NICK' and (constraint:=literal_constraint(c['text'])) is not None and
        constraint['kind']=='PREFIX' and (matches(next(iter(expanded)),constraint) or
        c['text'] in associated and next(iter(expanded)) in associated) for c in candidates)
    needs=(len(texts)>1 and not resolved_prefix) or automatic is None or bool(anomalies)
    result=dict(context or {},seat=nick['seat'],frame=nick.get('frame',nick.get('source_frame')),
        nick_crop_reference={k:nick.get(k) for k in ('source_file','source_line','roi','pixel_sha256','item_id')},
        nick_roi_raw_text=raw,nick_roi_confidence=_confidence(nick),
        chat_raw_text=chats[0].get('raw_text',chats[0].get('actor')) if chats else None,
        chat_confidence=_confidence(chats[0]) if chats else None,
        nick_observation=nick,chat_observations=chats,candidates=candidates,
        anomalous_nick_observations=anomalies,
        existing_dot_prefix_rule=dict(constraint=label,matching_chat=[c['text'] for c in expansion]),
        auto_selected_nick=automatic,auto_selection_reason=reason,needs_manual_review=needs,
        manual_ground_truth=None,manual_confirmed=False,canonical_nick=automatic,
        WHO_status='PROVEN' if automatic else 'MANUAL_REVIEW')
    if manual and manual.get('manual_confirmed') and manual.get('manual_ground_truth'):
        result.update(manual_ground_truth=manual['manual_ground_truth'],manual_confirmed=True,
                      canonical_nick=manual['manual_ground_truth'],WHO_status='PROVEN')
    return result
