"""Read a bound forced all-in proof without changing any source text."""
import re

ROLES = {'ANTE':'ANTE','SB':'SMALL_BLIND','BB':'BIG_BLIND',
         'SMALL_BLIND':'SMALL_BLIND','BIG_BLIND':'BIG_BLIND'}
PHYSICAL_ALL_IN = re.compile(
    r'^Dealer:\s*(.+?)\s+posts\s+(Ante|Small Blind|Big Blind)\s+'
    r'(\d+(?:[ \u00a0]\d{3})*)\s+and is All-in$', re.I)


def table_number(value):
    text = str(value)
    return int(text[1:] if text.startswith('T') else text) if re.fullmatch(r'T?\d+',text) else None


def bound_forced_all_in(rows, action, actor, amount):
    """Return (claim present, verified receipt); an invalid claim stays unknown."""
    claimed = any(r.get('canonical_provenance',{}).get('post',{}).get('all_in') is True for r in rows)
    if not claimed:
        return False,None
    if not rows or action not in ROLES or amount is None:
        return True,None
    first = rows[0]
    event_id = first.get('canonical_event_id')
    provenance = first.get('canonical_provenance',{})
    post = provenance.get('post',{})
    binding = provenance.get('binding_receipt',{})
    if (not event_id or post.get('all_in') is not True or
            post.get('role') not in ROLES or ROLES[post['role']] != ROLES[action] or
            post.get('actor') != actor or str(post.get('amount')) != str(amount) or
            any(r.get('canonical_conflict') or r.get('canonical_duplicate_of') or
                r.get('binding_status') != 'PROVEN_ACTOR_SEAT_HAND' or
                r.get('canonical_event_id') != event_id or r.get('canonical_provenance') != provenance or
                (table_number(r.get('table_id')),r.get('session_id'),r.get('hand_id'),r.get('seat')) !=
                (post.get('table'),post.get('session'),post.get('hand_id'),post.get('seat')) for r in rows) or
            (binding.get('table'),binding.get('session'),binding.get('hand_id'),binding.get('actor'),
             binding.get('seat'),binding.get('epoch')) !=
            (post.get('table'),post.get('session'),post.get('hand_id'),actor,post.get('seat'),post.get('epoch'))):
        return True,None
    physical = post.get('source_rows',[])
    refs = post.get('source_refs',[])
    if (not physical or [r.get('source_ref') for r in physical] != refs or
            refs not in binding.get('forced_source_refs',[]) or
            any((r.get('table_id'),r.get('session_id'),r.get('hand_id'),r.get('source_frame')) !=
                (post['table'],post['session'],post['hand_id'],post.get('source_frame')) for r in physical)):
        return True,None
    literal = ' '.join(r.get('text','').strip() for r in physical)
    match = PHYSICAL_ALL_IN.fullmatch(literal)
    if (not match or match.group(1) != actor or
            {'ante':'ANTE','small blind':'SMALL_BLIND','big blind':'BIG_BLIND'}[match.group(2).lower()] != ROLES[action] or
            match.group(3).replace(' ','').replace('\u00a0','') != str(amount)):
        return True,None
    return True,dict(canonical_event_id=event_id,source_refs=list(refs),
                     actor=actor,seat=post['seat'],occupancy_epoch=post['epoch'],
                     authority='OBSERVED_SOURCE_BOUND_ALL_IN')
