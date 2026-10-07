"""NCC classification contracts and saved same-row evidence invariants."""
import json
from pathlib import Path
from start_stack_multiclass_20261003 import rank, CLASSES

def test_relative_winner_can_be_below_absolute_threshold():
    winner,second,margin=rank({'cards':.91,'Turn':.63,'Flop':.50,'River':.58})
    assert winner=='cards' and second=='Turn'
    assert abs(margin-.28)<1e-12
    assert .91<.95

def test_tie_has_zero_margin_and_no_strict_separation():
    winner,second,margin=rank({'cards':.91,'Turn':.91,'Flop':.50,'River':.58})
    assert {winner,second}=={'cards','Turn'}
    assert margin==0.
    assert not margin>=.05

def test_negative_correlations_are_ranked_numerically():
    winner,second,margin=rank({'cards':-.25,'Turn':-.5,'Flop':-.75,'River':-.8})
    assert winner=='cards' and second=='Turn' and margin==.25

def test_margin_levels_do_not_change_absolute_score():
    scores={'cards':.925,'Turn':.70,'Flop':.68,'River':.6}
    before=dict(scores)
    winner,_,margin=rank(scores)
    acceptance=[winner=='cards' and margin>=m for m in [0.,.05,.1,.2,.3]]
    assert acceptance==[True,True,True,True,False]
    assert scores==before

def test_saved_frame_margin_comes_from_one_physical_row():
    p=Path(__file__).resolve().parents[1]/'runs/start_stack_frame_ledger_wide_smoke14_20261003/multi_template_classification'
    rows=[json.loads(x) for x in (p/'row_scores.jsonl').read_text(encoding='utf-8').splitlines()]
    lookup={(r['table'],r['hand_id'],r['frame'],r['row_index']):r for r in rows}
    assert rows
    for r in rows:
        assert set(r['scores'])==set(CLASSES)
        assert {xy[1] for xy in r['shifts'].values()}=={r['row_chat_span'][0]}
        assert all(88<=xy[0]<=94 for xy in r['shifts'].values())
    frames=[json.loads(x) for x in (p/'frame_scores.jsonl').read_text(encoding='utf-8').splitlines()]
    for f in frames:
        if f['selected_row'] is None:continue
        r=lookup[(f['table'],f['hand_id'],f['frame'],f['selected_row'])]
        assert f['scores']==r['scores']
        assert (f['winner'],f['second'],f['margin'])==(r['winner'],r['second'],r['margin'])
