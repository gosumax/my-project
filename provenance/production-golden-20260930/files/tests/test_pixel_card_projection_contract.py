import pytest
from parser_core.visual_cards import project_chat_card_events
from tests.test_chat_card_projection import fixture

def pixel_fixture():
    rows,reading=fixture()
    response={**reading['raw_vl_response'],'recognizer':'FROZEN_NATIVE_PIXEL_CARD_READER',
              'rank_config_sha256':'c'*64,'suit_config_sha256':'d'*64}
    reading.update(recognition_mode='PIXEL_ONLY',raw_vl_response=None,raw_pixel_response=response,
                   hybrid={'development_only':True,'rank_mode':'PIXEL_ONLY','rank_ncc_config_sha256':'c'*64})
    return rows,reading

def test_pixel_projection_retains_native_response_and_frozen_policy_provenance():
    rows,r=pixel_fixture()
    assert project_chat_card_events(rows,{'1:8:2':r})==1
    assert rows[0]['raw_vl_response'] is None
    assert rows[0]['raw_pixel_response']==r['raw_pixel_response']
    assert rows[0]['rank_mode']=='PIXEL_ONLY' and rows[0]['development_only'] is True
    assert rows[0]['rank_ncc_config_sha256']=='c'*64

@pytest.mark.parametrize('field,value',[
    ('source_sha256','f'*64),('recognizer','FOREIGN'),
    ('rank_config_sha256','not-a-hash'),('suit_config_sha256',None),('status','ERROR')])
def test_foreign_or_incomplete_pixel_response_cannot_project(field,value):
    rows,r=pixel_fixture();r['raw_pixel_response'][field]=value
    assert project_chat_card_events(rows,{'1:8:2':r})==0
    assert 'cards' not in rows[0]
