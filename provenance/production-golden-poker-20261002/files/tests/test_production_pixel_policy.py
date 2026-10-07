from pathlib import Path
import json
import pytest
from parser_core.card_hybrid import CardHybrid
ROOT=Path(__file__).resolve().parents[1]

def test_authorized_production_policy_keeps_frozen_matcher_and_disables_vl():
    reader=CardHybrid(ROOT/'configs/card_hybrid_local.json',ROOT/'configs/suit_backup_local.json',
                      rank_ncc_config=ROOT/'configs/rank_ncc_production.json')
    assert reader.rank_mode=='PIXEL_ONLY' and reader.development_only is False
    assert reader.rank_ncc is not None

@pytest.mark.parametrize('field', ['production_authorized','production_enabled'])
def test_production_policy_requires_explicit_authorization_and_enablement(tmp_path,field):
    policy=json.loads((ROOT/'configs/rank_ncc_production.json').read_text('utf8'))
    policy[field]=False;path=tmp_path/'policy.json';path.write_text(json.dumps(policy),'utf8')
    with pytest.raises(ValueError):
        CardHybrid(ROOT/'configs/card_hybrid_local.json',ROOT/'configs/suit_backup_local.json',rank_ncc_config=path)
