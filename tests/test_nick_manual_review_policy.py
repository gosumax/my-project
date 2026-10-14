"""Bounded spelling review contracts for an already associated occupant."""
import copy
import pytest

from parser_core.nick_manual_review import select_nickname


def nick(text='retre1', confidence=.90):
    return dict(seat='seat5',frame=123,raw_text=text,confidence=confidence,
                source_file='saved_nicks.jsonl',source_line=5,roi=[1,2,81,17],pixel_sha256='abc')


def chat(text='retrel',confidence=.99):
    return dict(raw_text=text,confidence=confidence,original_raw_text=f'Dealer: {text} folds',
                source_rows=[dict(frame_id=124,raw_ocr_text=f'Dealer: {text} folds')])


@pytest.mark.parametrize('nc,cc,expected',[(.90,.99,'retrel'),(.99,.90,'retre1')])
def test_associated_occupant_auto_selects_higher_confidence_spelling(nc,cc,expected):
    result=select_nickname(nick(confidence=nc),[chat(confidence=cc)],context={'table_id':9,'session_id':'saved'})
    assert result['canonical_nick']==expected and result['auto_selected_nick']==expected
    assert result['auto_selection_reason']=='HIGHER_CONFIDENCE'
    assert result['needs_manual_review'] is True
    assert result['seat']=='seat5' and result['table_id']==9


def test_equal_confidence_conflicting_spellings_stay_manual_review():
    result=select_nickname(nick(confidence=.97),[chat(confidence=.97)])
    assert result['canonical_nick'] is None and result['auto_selected_nick'] is None
    assert result['WHO_status']=='MANUAL_REVIEW' and result['needs_manual_review'] is True


def test_missing_confidence_cannot_choose_conflicting_spelling():
    result=select_nickname(nick(confidence=None),[chat()])
    assert result['canonical_nick'] is None
    assert result['WHO_status']=='MANUAL_REVIEW'


def test_manual_confirmation_preserves_distinct_auto_choice_and_raw_fields():
    original_nick=nick(confidence=.99)
    original_chat=chat(confidence=.90)
    before=copy.deepcopy((original_nick,original_chat))
    result=select_nickname(original_nick,[original_chat],manual={'manual_ground_truth':'retrel','manual_confirmed':True})
    assert result['canonical_nick']=='retrel'
    assert result['auto_selected_nick']=='retre1'
    assert result['manual_ground_truth']=='retrel' and result['manual_confirmed'] is True
    assert result['nick_roi_raw_text']=='retre1' and result['nick_roi_confidence']==.99
    assert result['chat_raw_text']=='retrel' and result['chat_confidence']==.90
    assert result['chat_observations'][0]['original_raw_text']=='Dealer: retrel folds'
    assert (original_nick,original_chat)==before


def test_unconfirmed_manual_suggestion_does_not_override_automatic():
    result=select_nickname(nick(confidence=.99),[chat(confidence=.90)],manual={'manual_ground_truth':'retrel','manual_confirmed':False})
    assert result['canonical_nick']=='retre1'
    assert result['manual_ground_truth'] is None


def test_existing_dot_prefix_expands_using_full_chat_without_rewriting_raw():
    result=select_nickname(nick('romashka...',.99),[chat('romashka91808',.90)])
    assert result['canonical_nick']=='romashka91808'
    assert result['auto_selection_reason']=='EXISTING_DOT_PREFIX'
    assert result['nick_roi_raw_text']=='romashka...'
    assert result['existing_dot_prefix_rule']['matching_chat']==['romashka91808']
    assert result['needs_manual_review'] is False


@pytest.mark.parametrize('suffix',['..','...','…'])
def test_chat_has_priority_for_matching_truncation_even_with_lower_confidence(suffix):
    result=select_nickname(nick('romashka'+suffix,.999),[chat('romashka91808',.90)])
    assert result['canonical_nick']=='romashka91808' and not result['needs_manual_review']


def test_previously_confirmed_truncated_pair_uses_chat_without_character_rewrite():
    result=select_nickname(nick('zeroto1era..',.999),[chat('zerotolerance',.99)],
        associated_aliases=['zeroto1era..','zerotolerance'])
    assert result['canonical_nick']=='zerotolerance' and not result['needs_manual_review']
    assert result['nick_roi_raw_text']=='zeroto1era..'
    assert result['auto_selection_reason']=='CONFIRMED_TRUNCATED_CHAT'


def test_same_spelling_has_no_manual_review_conflict():
    result=select_nickname(nick('sameplayer',.93),[chat('sameplayer',.99)])
    assert result['canonical_nick']=='sameplayer'
    assert result['needs_manual_review'] is False


def test_confirmed_manual_truth_resolves_tie_without_fabricating_auto_choice():
    result=select_nickname(nick(confidence=.97),[chat(confidence=.97)],manual={'manual_ground_truth':'retrel','manual_confirmed':True})
    assert result['canonical_nick']=='retrel' and result['auto_selected_nick'] is None
    assert result['WHO_status']=='PROVEN'


def test_canonical_actor_revalidation_preserves_all_raw_source_rows():
    from parser_core.poker_pipeline import analyze_timeline
    from parser_core.poker_reconstruction_layer import reconstruct_production_hands
    from parser_core.full_hh_profile import reconstruction_settings
    from parser_core.reconstruction_validation import validate_reconstruction
    texts=['Hand #123456','Dealer: Starting a new hand','Dealer: A posts Small Blind 5',
           'Dealer: B posts Big Blind 10','Dealer: A folds','Dealer: B gets uncalled bet','Dealer: B wins 10']
    raw=[dict(stable_row_id=f's:R{i}',row_pixel_sha256=f'{i:064x}',table_id='T01',session_id='s',
              frame_id=1,row_index=i,epoch=1,origin='NEW',raw_ocr_text=t,video_time=0)
         for i,t in enumerate(texts)]
    observed=analyze_timeline(raw,ocr_status={r['stable_row_id']:'OK' for r in raw})[0]
    original=copy.deepcopy(observed)
    result=reconstruct_production_hands(observed,config=reconstruction_settings(
        dict(enabled=True,max_nodes=20000,allow_logic_math_output=False)))[0][0]
    canonical=copy.deepcopy(result)
    for event in canonical['kept_events']:
        if event.get('player')=='A':
            event['player']='manualA'
    for step in canonical['diagnostic_trace']:
        if step['candidate_event'].get('player')=='A':
            step['candidate_event']['player']='manualA'
        if 'A' in step['street_contributions']:
            step['street_contributions']['manualA']=step['street_contributions'].pop('A')
    assert validate_reconstruction(observed[0]['observed_hand'],canonical)['reason']=='OBSERVED_ACTION_OR_ACTOR_NOT_REPRODUCIBLE'
    validated=validate_reconstruction(observed[0]['observed_hand'],canonical,nickname_canonicalization={'A':'manualA'})
    assert validated['accepted']
    messages=validated['candidate_hand']['messages']
    assert [r for m in messages for r in m['source_rows']]==[r for m in original[0]['observed_hand']['messages'] for r in m['source_rows']]
    assert any(m.get('player')=='manualA' and m.get('original_player')=='A' for m in messages)
    assert observed==original


def test_declared_pair_can_use_later_native_nick_but_not_another_hand():
    from parser_core.simple_seat_who import SeatIdentityLineage
    from parser_core.simple_saved_policy import LineageActorProvider
    def observation(frame,text):
        return dict(table_id=9,session_id='session',seat='seat5',frame=frame,raw_text=text,confidence=.95)
    lineage=SeatIdentityLineage([observation(10,'retre1'),observation(90,'5OLD'),observation(91,'5OLD'),
                                observation(150,'retre1'),observation(151,'retre1')])
    def hand(number):
        return dict(table_id=9,session_id='session',hand_id=number,messages=[dict(player='retrel',
            source_rows=[dict(confidence=.99,frame_id=100,raw_ocr_text='Dealer: retrel folds')])])
    first,second=hand('first'),hand('second')
    association=dict(table_id=9,session_id='session',seat='seat5',hand_id='first',
                     aliases=['retrel','retre1'],authority='USER_EXPLICIT_SAME_OCCUPANT')
    provider=LineageActorProvider(lineage,{(9,'session','first'):(100,200),(9,'session','second'):(201,300)},
                                 [first,second],[association])
    proved=provider.for_actor(first,'retrel')
    assert proved['status']=='PROVEN' and proved['seat']=='seat5'
    assert proved['anchor']['frame']==150
    assert provider.for_actor(second,'retrel')['status']=='UNKNOWN'
