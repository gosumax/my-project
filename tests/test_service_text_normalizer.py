"""Service normalization is a derived view, never a source/player correction."""
import copy
import pytest
import json
from pathlib import Path
from parser_core.service_text_normalizer import ALL_IN_VARIANTS, normalize_service, same_physical_neighbor
from parser_core.normalize_chat import normalize, normalize_with_service
from parser_core.poker_hand_assembly import parse_text, assemble_messages
from parser_core.poker_reconstruction.timeline import parse_rows
from parser_core.poker_reconstruction.assembly import candidate_assemblies

TAILS = json.loads((Path(__file__).parent / 'fixtures/saved_service_tail_contracts.json').read_text('utf8'))


@pytest.mark.parametrize('prefix',['Dealer: ','Дилер: ','[12:34] Dealer: ',''])
@pytest.mark.parametrize('suffix',['●','.●','!●'])
def test_time_bank_terminal_marker_is_lossless_in_both_parsers(prefix,suffix):
    raw=prefix+'antivoland uses Time Bank'+suffix
    derived=normalize_service(raw)
    assert derived.raw_ocr==raw and derived.normalized_text==raw[:-1]
    change=derived.changes[0]
    assert change['rule_id']=='SERVICE_TIME_BANK_TERMINAL_UI_MARKER_V1'
    assert change['source_span']==[len(raw)-1,len(raw)]
    assert change['observed_variant']=='●' and change['canonical_token']==''
    source=notice_rows([raw]);before=copy.deepcopy(source)
    message=assemble_messages(source,require_adjacency=True)[0]
    event=parse_rows(source)[0]
    assert message['action']=='TABLE_INFO' and message['player'] is None and message['amount'] is None
    assert event.kind=='INFO' and event.player is None and event.amount is None
    assert message['raw_text']==raw and message['source_rows']==before
    assert event.source['raw_text']==raw and event.source['rows']==before and source==before


@pytest.mark.parametrize('suffix',['','.','!'])
def test_ordinary_time_bank_is_unchanged_and_informational(suffix):
    raw='Dealer: user_1 uses Time Bank'+suffix
    derived=normalize_service(raw)
    assert derived.raw_ocr==derived.normalized_text==raw and not derived.changes
    source=notice_rows([raw])
    assert assemble_messages(source,require_adjacency=True)[0]['action']=='TABLE_INFO'
    assert parse_rows(source)[0].kind=='INFO'


@pytest.mark.parametrize('clause,action,kind,amount',[
    ('calls 6 000','CALL','CALL',6000),('bets 6 000','BET','BET',6000),
    ('raises to 18 000','RAISE','RAISE',18000),('posts Ante 700','ANTE','ANTE',700),
    ('posts Small Blind 3 000','SMALL_BLIND','SB',3000),
    ('posts Big Blind 6 000','BIG_BLIND','BB',6000),
    ('wins 16 200','WIN','WIN',16200),
])
def test_terminal_marker_on_chip_actions_keeps_literal_semantics(clause,action,kind,amount):
    raw='Dealer: antivoland '+clause+'●'
    derived=normalize_service(raw)
    assert derived.normalized_text==raw and not derived.changes
    source=notice_rows([raw])
    message=assemble_messages(source,require_adjacency=True)[0];event=parse_rows(source)[0]
    assert (message['action'],message['player'],int(message['amount']))==(action,'antivoland',amount)
    assert (event.kind,event.player,event.amount)==(kind,'antivoland',amount)


@pytest.mark.parametrize('raw',[
    'Dealer: a calls 6000 uses Time Bank●','Dealer: a uses Time Bank; calls 6000●',
    'Hand #123 uses Time Bank●','123●','Time Bank●','Dealer: a uses Time Bank●●',
    'Dealer: a uses Time● Bank','Dealer: a uses Time Bank● now',
    'Dealer: a has joined the table●','Dealer: a sits out●',
])
def test_unknown_or_material_clause_cannot_gain_time_bank_marker_normalization(raw):
    result=normalize_service(raw)
    assert result.raw_ocr==result.normalized_text==raw and not result.changes


@pytest.mark.parametrize('field,value',[('frame_id',101),('session_id','other'),('epoch',2),('row_index',4)])
def test_time_bank_fragment_needs_verified_physical_group(field,value):
    source=notice_rows(['Dealer: antivoland uses','Time Bank●'])
    source[1][field]=value
    raw=' '.join(r['raw_ocr_text'] for r in source)
    parsed=parse_text(raw,source_rows=source)
    assert parsed['action']=='UNKNOWN_EVENT' and not parsed.get('service_normalizations')


def indexed_saved_tail(kind):
    source = copy.deepcopy(TAILS[kind]['rows'])
    for i, row in enumerate(source, 1):
        row['source_index'] = i
    return source


def notice_rows(texts):
    return [dict(table_id='T02', session_id='s', epoch=1, frame_id=100, row_index=i,
        hand_id='h', source_file='saved', source_index=i + 1, stable_row_id=f's:{i}',
        raw_ocr_text=text, row_pixel_sha256=f'{i:064x}') for i, text in enumerate(texts)]


@pytest.mark.parametrize('texts', [
    ['Dealer: winner is awarded 100 p. for','busted player(s):','literal_nick0'],
    ['Dealer: winner is awarded 100 p. for','busted player(s):','A, B_'],
    ['Dealer: Please wait while players are','reseating●']])
def test_bounded_service_notice_same_source_cross_path(texts):
    source = notice_rows(texts); before = copy.deepcopy(source)
    messages = assemble_messages(source, require_pixel_cards=True, require_adjacency=True)
    events = parse_rows(source)
    paths, exhausted, _ = candidate_assemblies(source)
    assert len(messages) == len(events) == 1 and not exhausted
    assert messages[0]['action'] == 'TABLE_INFO' and messages[0]['player'] is None and messages[0]['amount'] is None
    assert events[0].kind == 'INFO' and all(len(path)==1 and path[0].kind=='INFO' for path in paths)
    assert messages[0]['source_rows'] == source and source == before


@pytest.mark.parametrize('field,value', [('table_id','T03'),('session_id','other'),('epoch',2),
    ('frame_id',101),('row_index',4),('hand_id','foreign'),('source_file','other')])
def test_bounded_service_notice_context_conflict_not_grouped(field,value):
    source = notice_rows(['Dealer: winner is awarded 100 p. for','busted player(s):','literal_nick0'])
    source[1][field] = value
    assert not any(len(m['source_rows'])==3 for m in assemble_messages(source,require_adjacency=True))
    assert not any(len(e.source['rows'])==3 for path in candidate_assemblies(source)[0] for e in path)


@pytest.mark.parametrize('tail', ['Dealer: A bets 100','Hand #123','A calls 100','Ka 7074K00060'])
def test_bounded_service_notice_cannot_swallow_material_clause(tail):
    source = notice_rows(['Dealer: winner is awarded 100 p. for','busted player(s):',tail])
    assert not any(len(m['source_rows'])==3 for m in assemble_messages(source,require_adjacency=True))


def test_bounded_show_saved_exact_card_receipt_joins_rank_and_opaque_tail():
    source = indexed_saved_tail('SHOW'); before = copy.deepcopy(source)
    messages = assemble_messages(source, require_pixel_cards=True, require_adjacency=True)
    assert len(messages) == 1 and messages[0]['action'] == 'SHOW'
    assert messages[0]['cards'] == ['Kc','Qh'] and messages[0]['field_sources']['cards'] == 'OBSERVED_PIXEL'
    assert messages[0]['source_rows'] == before and source == before
    assert all(len(path)==1 and path[0].kind=='SHOW' for path in candidate_assemblies(source)[0])


def test_bounded_existing_two_row_card_receipt_keeps_legacy_hand_container_binding():
    source = indexed_saved_tail('SHOW')
    source = [source[0], source[2]]
    source[0]['raw_text'] = source[0]['raw_ocr_text'] = 'Dealer: leopold7024 shows High card, K'
    source[1]['row_index'] = source[0]['row_index'] + 1
    for row in source:
        row['hand_id'] = None
    messages = assemble_messages(source, require_pixel_cards=True, require_adjacency=True)
    assert len(messages) == 1 and messages[0]['cards'] == ['Kc','Qh']
    assert messages[0]['source_rows'] == source


@pytest.mark.parametrize('field,value', [('frame_id',9212),('row_index',20),('hand_id','foreign'),
    ('row_pixel_sha256','0'*64),('session_id','foreign')])
def test_bounded_show_card_receipt_conflict_keeps_opaque_tail(field,value):
    source = indexed_saved_tail('SHOW'); source[2][field] = value
    messages = assemble_messages(source,require_pixel_cards=True,require_adjacency=True)
    assert not any(len(m['source_rows'])==3 for m in messages)
    assert messages[-1]['action'] == 'UNKNOWN_EVENT' and messages[-1]['raw_text'] == 'Ka'


def test_bounded_win_unresolved_saved_crop_is_not_a_card_binding():
    source = indexed_saved_tail('WIN_UNBOUND'); before = copy.deepcopy(source)
    messages = assemble_messages(source,require_pixel_cards=True,require_adjacency=True)
    assert messages[-1]['action'] == 'UNKNOWN_EVENT' and messages[-1]['raw_text'] == '7074K00060'
    assert not any(m.get('win_best_five_cards') for m in messages)
    assert source == before


@pytest.mark.parametrize('texts', [
    ['Dealer: Player A finished the','tournament'],
    ['Dealer: Player A has left the','table'],
    ['Dealer: Player A has left','the table'],
    ['Dealer: Player A has joined','the table'],
    ['Dealer: Player mooglle1 has left the','table0']])
def test_bounded_proved_complete_service_wrap_has_one_lossless_graph_path(texts):
    source=notice_rows(texts);before=copy.deepcopy(source)
    paths,exhausted,graph=candidate_assemblies(source)
    assert not exhausted and len(paths)==1 and len(paths[0])==1
    assert paths[0][0].kind=='INFO' and paths[0][0].source['rows']==source
    assert len(graph['0'])==1 and source==before


@pytest.mark.parametrize('field,value',[('frame_id',101),('epoch',2),('hand_id','foreign'),('session_id','other')])
def test_bounded_unproved_service_wrap_does_not_gain_single_graph_edge(field,value):
    source=notice_rows(['Dealer: Player A has left the','table'])
    source[1][field]=value
    paths,_,graph=candidate_assemblies(source)
    assert graph['0'][0]['event']['kind']=='UNKNOWN'

def rows(tail='All-n'):
    return [dict(table_id='T02',session_id='s',epoch=1,frame_id=1451,row_index=17+i,
                 hand_id='2878365630',source_file='saved',source_index=i+1,
                 stable_row_id=f's:R{i}',row_pixel_sha256=f'{i:064x}',
                 raw_ocr_text=t) for i,t in enumerate(['Dealer: temazm bets 135 000 and is',tail])]

@pytest.mark.parametrize('variant',ALL_IN_VARIANTS)
@pytest.mark.parametrize('action',['bets','calls','raises to','posts Ante','posts Small Blind','posts Big Blind'])
def test_exact_suffix_preserves_nick_amount_source_spans(variant,action):
    raw=f'Dealer: All-nSomething {action} 135 000 and is {variant}'
    result=normalize_service(raw)
    assert result.raw_ocr==raw and result.normalized_text==raw[:-len(variant)]+'All-in'
    lo,hi=result.changes[0]['source_span']
    assert raw[:lo]==result.normalized_text[:lo] and raw[lo:hi]==variant
    candidate,receipt=normalize_with_service([raw])
    assert candidate.actor=='All-nSomething' and candidate.amount_decimal=='135000'
    assert receipt.raw_ocr==raw

@pytest.mark.parametrize('raw',['All-nSomething','All-n','Al-in','Allin','Al-in0','All-in0','table0','rebuy...0','135 0000','Dealer: All-nSomething sits out','Dealer: a bets UNKNOWN and is All-n','Dealer: a bets 135 000 and is All-x','Hand #1350000','Dealer: a shows Ah Ad','Dealer: a bets 135 000 and is All-nSomething'])
def test_unsupported_or_protected_literal_unchanged(raw):
    r=normalize_service(raw)
    assert r.raw_ocr==r.normalized_text==raw and not r.changes

@pytest.mark.parametrize('variant',ALL_IN_VARIANTS)
def test_both_parser_paths_preserve_raw_group(variant):
    source=rows(variant);before=copy.deepcopy(source)
    messages=assemble_messages(source,require_adjacency=True)
    assert len(messages)==1 and messages[0]['action']=='BET' and messages[0]['amount']=='135000' and messages[0]['all_in']
    assert messages[0]['source_rows']==source
    events=parse_rows(source)
    assert len(events)==1 and events[0].kind=='BET' and events[0].amount==135000 and events[0].all_in
    assert events[0].source['rows']==source
    assert events[0].source['raw_text']==' '.join(r['raw_ocr_text'] for r in source)
    paths,exhausted,graph=candidate_assemblies(source)
    assert any(len(path)==1 and path[0].kind=='BET' and path[0].amount==135000 and path[0].all_in for path in paths)
    assert source==before


def test_prepared_receipt_preserves_whitespace_border_and_raw_fragments():
    fragments = ['  Dealer: temazm bets 135 000 and is |  ', '\tAll-n||  ']
    original = list(fragments)
    candidate, receipt = normalize_with_service(fragments)
    assert candidate.actor == 'temazm' and candidate.amount_decimal == '135000'
    assert fragments == original
    assert receipt.raw_ocr == ' '.join(original)
    assert receipt.normalized_text == 'Dealer: temazm bets 135 000 and is All-in'
    assert len(receipt.changes) == 1
    change = receipt.changes[0]
    assert change['raw_ocr'] == ' '.join(original)
    assert change['source_raw_fragments'] == original
    assert change['source_span_basis'] == 'PREPARED_SERVICE_INPUT'
    prepared = 'Dealer: temazm bets 135 000 and is All-n'
    assert change['prepared_service_input'] == prepared
    lo, hi = change['source_span']
    assert prepared[lo:hi] == 'All-n'
    assert prepared[:lo] == receipt.normalized_text[:lo]


def test_generic_cross_frame_group_retains_legacy_literal_service_suffix():
    source = rows('and is Allin')
    source[0]['raw_ocr_text'] = 'Dealer: temazm bets 135 000'
    source[1]['frame_id'] += 1
    before = copy.deepcopy(source)
    raw = ' '.join(row['raw_ocr_text'] for row in source)
    legacy, legacy_receipt = normalize_with_service([raw], apply_service=False)
    assert legacy.amount_decimal == '135000' and not legacy_receipt.changes
    assert legacy_receipt.normalized_text == raw
    # Generic diagnostic grouping already accepts this literal tail, but that
    # old grouping cannot grant a service-token correction across frames.
    messages = assemble_messages(source, require_adjacency=False)
    assert len(messages) == 1 and messages[0]['source_rows'] == source
    assert messages[0]['amount'] == '135000'
    assert messages[0]['normalized_text'].endswith('and is Allin')
    assert not messages[0].get('service_normalizations')
    parsed = parse_text(raw, source_rows=source)
    assert parsed['amount'] == '135000' and not parsed.get('service_normalizations')
    assert source == before

@pytest.mark.parametrize('field,value',[('table_id','T03'),('session_id','other'),('epoch',2),('frame_id',1452),('row_index',19),('hand_id','other'),('source_file','different')])
def test_service_neighbor_context_conflict_blocked(field,value):
    source=rows();source[1][field]=value
    assert not same_physical_neighbor(*source)
    assert not any(e.all_in for e in parse_rows(source))
    paths,_,_=candidate_assemblies(source)
    assert not any(e.all_in for path in paths for e in path)

@pytest.mark.parametrize('field',['table_id','session_id','epoch','frame_id','row_index'])
def test_missing_required_source_context_blocked(field):
    source=rows();source[1].pop(field)
    assert not same_physical_neighbor(*source)

def test_live_missing_hand_id_keeps_existing_epoch_authority():
    source=rows()
    for row in source:row.pop('hand_id')
    assert same_physical_neighbor(*source)

@pytest.mark.parametrize('raw,expected',[('Dealer: Waiting for bob to rebuy...0','Dealer: Waiting for bob to rebuy...'),('Dealer: bob has joined the table0','Dealer: bob has joined the table'),('Dealer: bob has left the table0','Dealer: bob has left the table')])
def test_exact_service_clause_aliases(raw,expected):
    result=normalize_service(raw)
    assert result.normalized_text==expected and result.raw_ocr==raw
    lo,hi=result.changes[0]['source_span']
    assert raw[:lo]==expected[:lo]

@pytest.mark.parametrize('conflict',[None,'frame_id','session_id','epoch'])
def test_three_row_normalization_requires_every_physical_link(conflict):
    source=rows('135 000')
    source[0]['raw_ocr_text']='Dealer: temazm bets'
    source[1]['raw_ocr_text']='135 000'
    source.append(dict(source[1],row_index=19,source_index=3,stable_row_id='s:R2',raw_ocr_text='and is All-n'))
    if conflict:
        source[0][conflict]={'frame_id':1450,'session_id':'other','epoch':0}[conflict]
    before=copy.deepcopy(source)
    messages=assemble_messages(source,require_adjacency=True)
    repaired=lambda m:m.get('action')=='BET' and m.get('amount')=='135000' and m.get('all_in')
    events=parse_rows(source)
    paths,_,_=candidate_assemblies(source)
    if conflict:
        assert not any(repaired(m) for m in messages)
        assert not any(e.kind=='BET' and e.amount==135000 and e.all_in for e in events)
        assert not any(e.kind=='BET' and e.amount==135000 and e.all_in for path in paths for e in path)
    else:
        assert any(repaired(m) for m in messages)
        assert any(e.kind=='BET' and e.amount==135000 and e.all_in for e in events)
        assert any(e.kind=='BET' and e.amount==135000 and e.all_in for path in paths for e in path)
    assert source==before
