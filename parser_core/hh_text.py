"""Presentation-only HH renderer; source diagnostics stay in their JSON audit.

No metadata, money, card or seat repair is performed here. Only a certified
complete hand can be rendered. Optional metadata must be supplied by its caller
from this hand's own evidence, never from the user's example hand.
"""
import re
from collections import defaultdict

SERVICE = {'HAND_HEADER', 'HAND_ID', 'NEW_HAND', 'TABLE_INFO', 'BLIND_LEVEL', 'HAND_END'}
STREETS = {'FLOP': 3, 'TURN': 4, 'RIVER': 5}
SIMPLE_SITOUT_CONTRACT = 'CURRENT_SIMPLE_SITOUT_STATE_V1'


def simple_sitout_state_valid(state, *, table, session, hand, player, seat):
    """Check an existing SIMPLE classification's scope; never infer a status."""
    from .simple_stack_policy import POLICY
    from .simple_saved_policy import SIT_OUT_GATE_AUTHORITY, valid_sitout_status
    if not isinstance(state, dict):
        return False
    if (type(state.get('hand_start')) is not int or type(state.get('hand_end')) is not int or
            not isinstance(state.get('sitout_sources'), list)):
        return False
    table_id = state.get('table_id')
    return (type(table_id) is int and f'T{table_id:02d}' == table and
            state.get('session_id') == session and state.get('hand_id') == hand and
            state.get('player') == player and state.get('seat') == seat and
            state.get('policy') == POLICY and state.get('WHO_status') == 'PROVEN' and
            state.get('player_status') == 'SIT_OUT' and
            state.get('sitout_authority') == SIT_OUT_GATE_AUTHORITY and
            state.get('reason') == 'NORMAL_SIT_OUT_WITHOUT_NUMERIC' and
            all(state.get(k) is None for k in ('DISPLAYED_STACK', 'START_STACK', 'first_numeric_source')) and
            not state.get('sitout_numeric_conflict') and
            not state.get('sitout_state_conflicts') and
            any(valid_sitout_status(source,table_id=table_id,session_id=session,seat=seat,
                    hand_id=hand,hand_start=state['hand_start'],hand_end=state['hand_end'])
                for source in state.get('sitout_sources', []) if isinstance(source, dict)))


def sitout_exemption(hand,player):
    if (hand.get('hh_contract') == SIMPLE_SITOUT_CONTRACT and
            player.get('ACTOR_status') == 'PROVEN' and player.get('player_status') == 'SIT_OUT' and
            player.get('stack_requirement') == 'NOT_REQUIRED_SIT_OUT' and
            player.get('START_STACK_status') == 'UNKNOWN' and player.get('START_STACK') is None and
            player.get('DISPLAYED_STACK') is None and
            simple_sitout_state_valid(player.get('simple_sitout_state'), table=hand.get('table'),
                session=hand.get('session'), hand=hand.get('hand_number'),
                player=player.get('player'), seat=player.get('seat'))):
        return True
    from .sitout_hand_evidence import HH_CONTRACT,RULE
    receipt=player.get('sitout_binding_receipt') or {}
    own=[a for a in hand.get('actions',[]) if a.get('player')==player.get('player')]
    return (hand.get('hh_contract') in {HH_CONTRACT, SIMPLE_SITOUT_CONTRACT} and player.get('player_status')=='SIT_OUT' and
            player.get('stack_requirement')=='NOT_REQUIRED_SIT_OUT' and
            player.get('START_STACK_status')=='UNKNOWN' and player.get('START_STACK') is None and
            receipt.get('rule') in {RULE,RULE+'__DIRECT_CHAT_NICK_V1'} and receipt.get('hh_contract')==HH_CONTRACT and
            receipt.get('actor')==player.get('player') and receipt.get('seat')==player.get('seat') and
            receipt.get('hand_id')==hand.get('hand_number') and receipt.get('session_id')==hand.get('session') and
            type(receipt.get('table_id')) is int and f"T{receipt['table_id']:02d}"==hand.get('table') and
            receipt.get('numeric_stack_authority') is False and receipt.get('synthetic_fold_authority') is False and
            bool(own) and own[-1].get('action')=='FOLD' and sum(a.get('action')=='FOLD' for a in own)==1 and
            all(a.get('action') in {'ANTE','SMALL_BLIND','BIG_BLIND','FOLD'} and not a.get('all_in') for a in own))


def integer(value):
    if isinstance(value, bool) or not re.fullmatch(r'\d+', str(value)):
        raise ValueError('UNPROVEN_INTEGER')
    return int(value)


def render_complete(hand, metadata=None, *, no_sb_source=None):
    if hand.get('status') != 'COMPLETE' or hand.get('certified') is not True or hand.get('blockers'):
        raise ValueError('COMPLETE_HH_REQUIRED')
    metadata = dict(metadata or {})
    seats = {}
    for player in hand['players']:
        exempt=sitout_exemption(hand,player)
        if player.get('START_STACK_status') != 'PROVEN' and not exempt:
            raise ValueError('PROVEN_SEAT_AND_STACK_REQUIRED')
        match = re.fullmatch(r'seat([1-9]\d*)', str(player.get('seat')))
        if not match:
            raise ValueError('PROVEN_SEAT_REQUIRED')
        seat = int(match[1])
        if seat in seats or any(p['player'] == player['player'] for p in seats.values()):
            raise ValueError('SEAT_PLAYER_CONFLICT')
        if not exempt:integer(player['START_STACK'])
        seats[seat] = player
    actions = hand['actions']
    blinds = {m['action']: integer(m['amount']) for m in actions if m['action'] in {'SMALL_BLIND', 'BIG_BLIND'}}
    if set(blinds) == {'BIG_BLIND'}:
        from .legitimate_no_sb import prove_no_sb
        candidate=hand.get('derived_final_validation',{}).get('candidate_hand')
        absence=prove_no_sb(candidate,no_sb_source) if candidate is not None else None
        if (absence is None or hand.get('no_sb_authority')!=absence or
                (hand.get('table'),hand.get('session'),hand.get('hand_number')) !=
                    (f"T{absence['table_id']:02d}",absence['session_id'],absence['hand_id']) or
                actions!=candidate['messages'] or blinds['BIG_BLIND']!=absence['nominal_blinds']['BB']):
            raise ValueError('LEGITIMATE_NO_SB_AUTHORITY_UNPROVEN')
        # Nominal stakes are metadata. No SMALL_BLIND action is inserted.
        blinds['SMALL_BLIND']=absence['nominal_blinds']['SB']
    if set(blinds) != {'SMALL_BLIND', 'BIG_BLIND'}:
        raise ValueError('BLINDS_UNPROVEN')
    title = f"PokerDom Hand #{hand['hand_number']}: Hold'em No Limit ({blinds['SMALL_BLIND']}/{blinds['BIG_BLIND']})"
    if metadata.get('utc_datetime'):
        title += ' - ' + metadata['utc_datetime'] + ' UTC'
    table = f"Table '{metadata.get('table_name', hand['table'])}'"
    if metadata.get('max_seats'):
        table += f" {integer(metadata['max_seats'])}-max"
    if metadata.get('button_seat') is not None:
        button = integer(metadata['button_seat'])
        if button not in seats:
            raise ValueError('BUTTON_SEAT_NOT_IN_ROSTER')
        table += f' Seat #{button} is the button'
    lines = [title, table]
    lines.extend(f"Seat {seat}: {p['player']} "+('(sitting out)' if sitout_exemption(hand,p)
                 else f"({p['START_STACK']} in chips)") for seat, p in sorted(seats.items()))
    board = []
    street = 'PREFLOP'
    committed = defaultdict(int)
    winners = defaultdict(int)
    folds = {}
    shows = {}
    roles = defaultdict(list)
    showdown = False
    for action in actions:
        kind = action['action']
        if kind in SERVICE:
            continue
        actor = action.get('player')
        if actor is not None and actor not in {p['player'] for p in seats.values()}:
            raise ValueError('ACTION_ACTOR_NOT_IN_ROSTER')
        suffix = ' and is all-in' if action.get('all_in') else ''
        if kind == 'DEAL':
            lines.append('*** HOLE CARDS ***')
        elif kind == 'SHOWDOWN':
            if not showdown:
                lines.append('*** SHOW DOWN ***')
                showdown = True
        elif kind in STREETS:
            cards = list(action.get('cards') or [])
            if len(cards) != STREETS[kind] or cards[:len(board)] != board or len(set(cards)) != len(cards):
                raise ValueError('COMPLETE_BOARD_REQUIRED')
            if kind == 'FLOP':
                lines.append('*** FLOP *** [' + ' '.join(cards) + ']')
            else:
                lines.append(f'*** {kind} *** [' + ' '.join(cards[:-1]) + '] [' + cards[-1] + ']')
            board = cards
            street = kind
            committed.clear()
        elif kind in {'ANTE', 'SMALL_BLIND', 'BIG_BLIND'}:
            amount = integer(action['amount'])
            verb = {'ANTE': 'posts the ante', 'SMALL_BLIND': 'posts small blind', 'BIG_BLIND': 'posts big blind'}[kind]
            lines.append(f'{actor}: {verb} {amount}{suffix}')
            if kind != 'ANTE':
                committed[actor] += amount
                roles[actor].append('small blind' if kind == 'SMALL_BLIND' else 'big blind')
        elif kind in {'CALL', 'BET'}:
            amount = integer(action['amount'])
            # Renderer consumes the validated economic delta, not OCR meanings.
            if not action.get('amount_proof'):
                raise ValueError('VALIDATED_AMOUNT_REQUIRED')
            committed[actor] += amount
            lines.append(f"{actor}: {'calls' if kind == 'CALL' else 'bets'} {amount}{suffix}")
        elif kind == 'RAISE':
            if not action.get('amount_proof'):
                raise ValueError('VALIDATED_AMOUNT_REQUIRED')
            target = integer(action['amount'])
            increment = target - max(committed.values(), default=0)
            if increment <= 0:
                raise ValueError('POSITIVE_RAISE_INCREMENT_REQUIRED')
            committed[actor] = target
            lines.append(f'{actor}: raises {increment} to {target}{suffix}')
        elif kind == 'UNCALLED_RETURN':
            amount = integer(action['amount'])
            committed[actor] -= amount
            if committed[actor] < 0:
                raise ValueError('RETURN_EXCEEDS_STREET_CONTRIBUTION')
            lines.append(f'Uncalled bet ({amount}) returned to {actor}')
        elif kind == 'FOLD':
            folds[actor] = street
            lines.append(f'{actor}: folds')
        elif kind == 'CHECK':
            lines.append(f'{actor}: checks')
        elif kind == 'NO_SHOW':
            lines.append(f"{actor}: doesn't show hand")
        elif kind == 'SHOW':
            cards = list(action.get('cards') or [])
            if len(cards) != 2:
                raise ValueError('SHOW_CARDS_UNPROVEN')
            if not showdown:
                lines.append('*** SHOW DOWN ***')
                showdown = True
            shows[actor] = cards
            lines.append(f'{actor}: shows [' + ' '.join(cards) + ']')
        elif kind == 'WIN':
            amount = integer(action['amount'])
            winners[actor] += amount
            raw = action.get('raw_text', '')
            pot = re.search(r'\b(side pot(?: #\d+)?|main pot)\b', raw, re.I)
            label = pot[1].lower() if pot else 'pot'
            lines.append(f'{actor} collected {amount} from {label}')
        else:
            raise ValueError('UNSUPPORTED_POKER_ACTION: ' + kind)
    total = integer(hand['final_pot'])
    rake = total - sum(winners.values())
    if rake != 0:
        raise ValueError('PAYOUT_TOTAL_CONFLICT')
    # Exact validated contributions less returns equal observed payouts. Zero
    # rake follows from that equality; no fee is supplied by the example file.
    lines.extend(['*** SUMMARY ***', f'Total pot {total} | Rake {rake}', 'Board [' + ' '.join(board) + ']'])
    for seat, player in sorted(seats.items()):
        actor = player['player']
        tags = list(roles[actor])
        if sitout_exemption(hand,player):tags.append('sitting out')
        if metadata.get('button_seat') == seat:
            tags.insert(0, 'button')
        label = actor + ''.join(f' ({tag})' for tag in tags)
        if actor in folds:
            outcome = 'folded before Flop' if folds[actor] == 'PREFLOP' else 'folded on the ' + folds[actor].title()
        elif actor in shows:
            outcome = 'showed [' + ' '.join(shows[actor]) + '] and ' + (f'won ({winners[actor]})' if winners[actor] else 'lost')
        elif winners[actor]:
            outcome = f'collected ({winners[actor]})'
        else:
            outcome = 'did not show'
        lines.append(f'Seat {seat}: {label} {outcome}')
    return '\n'.join(lines) + '\n\n'
