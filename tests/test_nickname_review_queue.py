import copy

import pytest

from parser_core.nickname_review import prepare_cases,unreviewed_cases
from parser_core.player_registry import PlayerRegistry,digest


def case():
    result=dict(context=dict(table_id=1,session_id='s',hand_id='h',seat='seat1'),
        identity_interval=[10,None],source_pins={},reason='LOW_CONFIDENCE',observations=[
            dict(id='nick',kind='NICK',text='actualPlayer',confidence=.998,selected=True,source={}),
            dict(id='noise',kind='NICK',text='5OLD',confidence=.519,selected=False,source={}),
            dict(id='chat',kind='CHAT',text='actualPlayer',confidence=.99,selected=True,source={})])
    result['case_id']=digest(result)
    return result


def test_unrelated_ignored_noise_does_not_send_high_confidence_reference_to_review():
    original=case();before=copy.deepcopy(original)
    cases,audit=prepare_cases([original])
    assert not cases and original==before
    assert audit[0]['excluded_spelling_suggestions'][0]['text']=='5OLD'
    assert audit[0]['identity_evidence_changed'] is False


def test_low_confidence_regular_nickname_stays_in_review():
    original=case();original['observations'][0]['confidence']=.87
    cases,_=prepare_cases([original])
    assert len(cases)==1
    assert {o['text'] for o in cases[0]['observations']}=={'actualPlayer'}
    assert cases[0]['excluded_spelling_suggestions'][0]['text']=='5OLD'


@pytest.mark.parametrize('suffix',['..','...','…'])
def test_matching_truncated_nickname_is_automatic_not_a_review_case(suffix):
    original=case();original['observations']=[
        dict(original['observations'][0],text='actual'+suffix,confidence=.87),
        original['observations'][2]]
    before=copy.deepcopy(original)
    cases,audit=prepare_cases([original])
    assert cases==[] and original==before
    assert audit[0]['automatic_truncation_matches']==[dict(text='actual'+suffix,chat_nickname='actualPlayer')]


def test_near_ignored_spelling_stays_even_when_anchor_is_high_confidence():
    original=case();original['observations'][1]['text']='actualP1ayer'
    cases,_=prepare_cases([original])
    assert len(cases)==1 and cases[0]['reason']=='SIMILAR_SPELLING_CONFLICT'
    assert len(cases[0]['observations'])==3


def test_exact_duplicate_spelling_cases_collapse_without_combining_physical_intervals():
    first=case();first['observations'][0]['confidence']=.87
    second=copy.deepcopy(first);second['identity_interval']=[80,None];second['case_id']=digest(second)
    cases,audit=prepare_cases([first,second])
    assert len(cases)==1 and cases[0]['identity_interval']==[10,None]
    assert cases[0]['duplicate_contexts'][0]['identity_interval']==[80,None]
    assert audit[1]['queue_status']=='DUPLICATE_SPELLING_CASE'
    other=copy.deepcopy(second);other['context']['seat']='seat2';other['case_id']=digest(other)
    assert len(prepare_cases([first,other])[0])==2


def test_known_spellings_leave_queue_but_new_similar_spelling_requires_review(tmp_path):
    original=case();original['observations'][0]['confidence']=.87
    cases,_=prepare_cases([original]);current=cases[0]
    path=tmp_path/'players.sqlite';db=PlayerRegistry(path)
    assert unreviewed_cases(cases,db)==cases
    db.confirm(current,'actualPlayer',[o['id'] for o in current['observations']])
    db=PlayerRegistry(path)
    assert not unreviewed_cases(cases,db)
    typo=copy.deepcopy(current);typo['observations'][0]['text']='actualP1ayer'
    assert unreviewed_cases([typo],db)==[typo]


def test_ui_alias_scope_correction_is_append_only_and_cannot_add_aliases(tmp_path):
    original=case();db=PlayerRegistry(tmp_path/'players.sqlite')
    prior=db.confirm(original,'actualPlayer',['nick','noise','chat'])
    db=PlayerRegistry(db.path);old=copy.deepcopy(db.reviews[prior['review_id']])
    db.confirm(original,old['canonical'],['nick','chat'],source_review_id=prior['review_id'],restrict_ui_aliases=True)
    db=PlayerRegistry(db.path)
    assert db.reviews[prior['review_id']]==old
    assert db.resolve('5OLD','NICK') is None
    assert db.resolve('actualPlayer','NICK')['canonical']=='actualPlayer'
    assert next(iter(db.active_reviews.values()))['ui_alias_scope_restricted']
    with pytest.raises(ValueError,match='original human decision'):
        db.confirm(original,'differentPlayer',['nick'],source_review_id=next(iter(db.active_reviews)),restrict_ui_aliases=True)


def test_scope_correction_requires_existing_human_review(tmp_path):
    with pytest.raises(ValueError,match='original human review'):
        PlayerRegistry(tmp_path/'players.sqlite').confirm(case(),'actualPlayer',['nick'],restrict_ui_aliases=True)


def test_known_aliases_do_not_hide_new_hand_scoped_identity_review(tmp_path):
    original=case();original['observations'][0]['confidence']=.87
    current=prepare_cases([original])[0][0]
    current.update(review_scope='HAND_SEAT',review_scope_key='this-hand',hand_bounds=[0,30])
    db=PlayerRegistry(tmp_path/'players.sqlite')
    db.confirm(current,'actualPlayer',[o['id'] for o in current['observations']]);db=PlayerRegistry(db.path)
    assert not unreviewed_cases([current],db)
    later=copy.deepcopy(current);later['context']['hand_id']='next-hand';later['review_scope_key']='next-hand'
    assert unreviewed_cases([later],db)==[later]


def test_cached_uncorroborated_case_cannot_live_on_after_its_bad_alias_was_removed(tmp_path):
    item=case();item['observations']=[dict(item['observations'][1],selected=True)]
    assert not unreviewed_cases([item],PlayerRegistry(tmp_path/'players.sqlite'))


def test_uncorroborated_native_manual_band_does_not_bypass_junk_exclusion(tmp_path):
    from parser_core.nickname_review import MANUAL_NICK_REVIEW
    item=case();item['observations']=[dict(item['observations'][0],confidence=.85)]
    item['review_eligibility']=MANUAL_NICK_REVIEW
    db=PlayerRegistry(tmp_path/'players.sqlite')
    assert unreviewed_cases([item],db)==[]
    item['observations'][0]['confidence']=.8499
    assert unreviewed_cases([item],db)==[]
    item['observations'][0]['confidence']=.90
    assert unreviewed_cases([item],db)==[]


def test_repeated_manual_band_spelling_shown_once_keeps_individual_hand_cases():
    from parser_core.nickname_review import MANUAL_NICK_REVIEW,group_spelling_cases
    items=[]
    for n in range(7):
        item=case();item['observations']=[dict(item['observations'][0],confidence=.87),item['observations'][2]]
        item.update(review_eligibility=MANUAL_NICK_REVIEW,review_scope='HAND_SEAT',review_scope_key=str(n))
        item['context']['hand_id']=str(n);item['case_id']=digest({k:v for k,v in item.items() if k!='case_id'});items.append(item)
    before=copy.deepcopy(items)
    grouped=group_spelling_cases(items)
    assert len(grouped)==1 and grouped[0]['review_group']['case_ids']==[c['case_id'] for c in items]
    assert grouped[0]['review_group']['contexts']==[c['context'] for c in items]
    assert items==before and grouped[0]['source_pins']==items[0]['source_pins']
    other=copy.deepcopy(items[0]);other['context']['seat']='seat2'
    assert len(group_spelling_cases([*items,other]))==2
    typo=copy.deepcopy(items[0]);typo['observations'][0]['text']='actua1Player'
    assert len(group_spelling_cases([*items,typo]))==2


def grouped_confirmation_cases(tmp_path):
    from parser_core.nickname_review import MANUAL_NICK_REVIEW
    import hashlib
    path=tmp_path/'physical_rows.jsonl';path.write_text('{}',encoding='utf-8')
    pins={str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
    items=[]
    for n in range(7):
        item=case();item['observations']=[dict(item['observations'][0],confidence=.87),item['observations'][2]]
        item.update(review_eligibility=MANUAL_NICK_REVIEW,review_scope='HAND_SEAT',review_scope_key=str(n),
                    hand_bounds=[n*100,n*100+99],source_pins=pins)
        item['context']['hand_id']=str(n);item['case_id']=digest({k:v for k,v in item.items() if k!='case_id'});items.append(item)
    return items


def test_group_confirmation_keeps_seven_separate_source_bound_receipts(tmp_path):
    from scripts.review_nicknames import confirm_review_cases
    from parser_core.nickname_review import group_spelling_cases
    items=grouped_confirmation_cases(tmp_path);before=copy.deepcopy(items)
    db_path=tmp_path/'players.sqlite'
    result=confirm_review_cases(items,items[0]['case_id'],'actualPlayer',['nick','chat'],tmp_path,db_path)
    registry=PlayerRegistry(db_path)
    assert result['confirmed_cases']==7 and len(registry.active_reviews)==7 and items==before
    records={r['case_id']:r for r in registry.active_reviews.values()}
    for item in items:
        saved=records[item['case_id']]
        for field in ('context','hand_bounds','source_pins','review_scope_key'):
            assert saved[field]==item[field]
    assert group_spelling_cases(unreviewed_cases(items,registry))==[]


@pytest.mark.parametrize('change',['child','source'])
def test_bad_group_child_or_source_prevents_every_confirmation(tmp_path,change):
    from scripts.review_nicknames import confirm_review_cases
    items=grouped_confirmation_cases(tmp_path)
    if change=='child':
        items[-1]['context']['hand_id']='tampered'
    else:
        (tmp_path/'physical_rows.jsonl').write_text('{"changed":true}',encoding='utf-8')
    path=tmp_path/'players.sqlite'
    with pytest.raises(ValueError):
        confirm_review_cases(items,items[0]['case_id'],'actualPlayer',['nick','chat'],tmp_path,path)
    assert not path.exists()


def test_group_displays_union_of_weak_variants_and_confirms_only_each_child_sources(tmp_path):
    from parser_core.nickname_review import group_spelling_cases
    from scripts.review_nicknames import confirm_review_cases
    items=grouped_confirmation_cases(tmp_path)
    for i,item in enumerate(items[1:]):
        typo='actualP1ayer' if i%2 else 'actualPlayer_'
        item['observations'].append(dict(id=digest(['NICK',typo]),kind='NICK',text=typo,
                                        confidence=.82,selected=True,source={}))
        item['case_id']=digest({k:v for k,v in item.items() if k!='case_id'})
    display=group_spelling_cases(items)
    assert len(display)==1
    assert {o['text'] for o in display[0]['observations']}=={'actualPlayer','actualP1ayer','actualPlayer_'}
    result=confirm_review_cases(items,display[0]['case_id'],'actualPlayer',
        [o['id'] for o in display[0]['observations']],tmp_path,tmp_path/'players.sqlite')
    records={r['case_id']:r for r in PlayerRegistry(tmp_path/'players.sqlite').active_reviews.values()}
    assert result['confirmed_cases']==7
    for item in items:
        assert {o['text'] for o in records[item['case_id']]['observations']}=={o['text'] for o in item['observations']}


def test_group_does_not_include_stronger_or_dissimilar_other_nickname(tmp_path):
    from parser_core.nickname_review import group_spelling_cases
    items=grouped_confirmation_cases(tmp_path)
    for raw,confidence in [('actualP1ayer',.90),('differentPlayer',.87)]:
        other=copy.deepcopy(items[0]);other['observations'].append(dict(id='other',kind='NICK',text=raw,
            confidence=confidence,selected=True,source={}))
        assert len(group_spelling_cases([*items,other]))==2
