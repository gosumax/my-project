import copy
import json
from pathlib import Path

import pytest

from parser_core.joint_start_stack import own_posts


@pytest.fixture
def source():
    return json.loads((Path(__file__).parent/'fixtures/own_forced_wrapped_prefix.json').read_text('utf-8'))


def test_native_wrapped_forced_posts_reproduce_original_rows_and_amounts(source):
    before=copy.deepcopy(source)
    assert own_posts(source['source_context'],source['actor'])==source['own_forced_posts']
    assert source==before


@pytest.mark.parametrize('mutation',['missing_tail','duplicate_post','foreign_header','financial_line','after_deal','low_small_tail'])
def test_forced_prefix_grammar_cannot_ignore_a_gap_or_extra_action(source,monkeypatch,mutation):
    import parser_core.joint_start_stack as module
    # Reproduce native inputs first, then isolate the grammar's rejection of
    # malformed source sequences. Full provenance rejection has separate tests.
    rows=copy.deepcopy(module.source_rows(source['source_context']))
    small=next(i for i,(r,*_) in enumerate(rows) if r['raw_ocr_text'].endswith(('posts Small','posts Small Blind')))
    post=next(i for i,(r,*_) in enumerate(rows) if 'posts Ante' in r['raw_ocr_text'])
    if mutation=='missing_tail':rows.pop(small+1)
    elif mutation=='duplicate_post':rows.insert(post+1,copy.deepcopy(rows[post]))
    elif mutation=='foreign_header':rows[0][0]['raw_ocr_text']='Hand #12345'
    elif mutation=='financial_line':rows[post][0]['raw_ocr_text']='Dealer: someone calls 100'
    elif mutation=='after_deal':rows.append(copy.deepcopy(rows[post]))
    elif mutation=='low_small_tail':rows[small+1][0]['confidence']=.929
    monkeypatch.setattr(module,'source_rows',lambda context:rows)
    with pytest.raises(ValueError):
        own_posts(source['source_context'],source['actor'])
