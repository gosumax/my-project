"""One explicit production profile and a checked module connection matrix."""
FULL_HH_PRODUCTION = 'FULL_HH_PRODUCTION'
REQUIRED = (
    'CHAT', 'CHAT_TRACKER', 'CHAT_OCR', 'PHYSICAL_CHAT_ADMISSION', 'HAND_HEADER',
    'HAND_NUMBER', 'HAND_STATE_TRIGGER', 'FORCED_POSTS', 'ANTE', 'SB', 'BB',
    'DEALING_CARDS', 'NICKNAME_BINDING', 'SEAT_BINDING', 'CARDS', 'BOARD', 'SHOWDOWN',
    'STACK', 'STACK_VISUAL_TRACKER', 'DIGIT_GATE', 'STACK_OCR', 'STACK_SQLITE',
    'STACK_EVIDENCE', 'START_STACK', 'HAND_SEGMENTATION', 'STREETS',
    'ACTION_RECONSTRUCTION', 'POKER_LOGIC', 'MATH_VALIDATION', 'POT_STACK_CONSISTENCY',
    'FINAL_VALIDATOR', 'HH_EXPORT', 'PERFORMANCE_DIAGNOSTICS',
)


def require_complete(matrix):
    missing = [name for name in REQUIRED if matrix.get(name) is not True]
    if missing:
        raise ValueError('FULL_HH_PRODUCTION_INCOMPLETE: ' + ', '.join(missing))
    return matrix


def configure(args):
    if args.profile != FULL_HH_PRODUCTION:
        return False
    if args.skip_vl or args.stack_evidence is not None:
        raise ValueError('FULL_HH_PRODUCTION_INCOMPLETE: cards required; fresh live STACK required')
    args.enable_dealing_cards_hand_state_trigger = True
    args.enable_start_stack_capture = True
    args.dealing_start_stack_acceptance = True
    args.stack_timeline = True
    args.stack_start_stack = True
    args.performance_diagnostics = True
    return True


def reconstruction_settings(config):
    """FULL explores CALL meanings and validates a separate derived candidate."""
    return dict(config,allow_logic_math_output=True,strict_source_uniqueness=True,first_visible_stack_contract=True)


def project_current_card_evidence(timeline, readings):
    """Shared live/saved FULL projection after the source-bound canonical join.

    Readings come from this run's PIXEL_ONLY producer (or its fresh saved-pixel
    replay). Each proof keeps its existing native source and consumer guards.
    """
    from .visual_cards import associate_visual_evidence, project_chat_card_events
    from .win_tail_evidence import project_win_tails
    from .card_set_evidence import project_unique_card_sets

    associate_visual_evidence(timeline, readings)
    native = project_chat_card_events(timeline, readings)
    winning = project_win_tails(timeline, readings)
    derived = project_unique_card_sets(timeline, readings)
    return dict(native_card_events=native, win_best_five_proofs=winning,
                unique_card_set_proofs=derived)


def connection_matrix(*, stage, ocr, header_lane, sidecar, stack_live, seat_binding,
                      stack_timeline, cards, reconstruction_enabled, diagnostic, hh_export):
    """ON denotes a connected consumer, never proven evidence or HH quality."""
    values = dict.fromkeys(REQUIRED, False)
    values.update(CHAT=stage is not None, CHAT_TRACKER=stage is not None, CHAT_OCR=ocr is not None,
        PHYSICAL_CHAT_ADMISSION=bool(stage and stage.physical_admission),
        HAND_HEADER=header_lane is not None, HAND_NUMBER=header_lane is not None,
        HAND_STATE_TRIGGER=sidecar is not None, FORCED_POSTS=seat_binding is not None,
        ANTE=seat_binding is not None, SB=seat_binding is not None, BB=seat_binding is not None,
        DEALING_CARDS=sidecar is not None, NICKNAME_BINDING=seat_binding is not None,
        SEAT_BINDING=seat_binding is not None, CARDS=cards is not None, BOARD=cards is not None,
        SHOWDOWN=cards is not None, STACK=stack_timeline is not None,
        STACK_VISUAL_TRACKER=stack_timeline is not None, DIGIT_GATE=stack_timeline is not None,
        STACK_OCR=stack_timeline is not None, STACK_SQLITE=stack_timeline is not None,
        STACK_EVIDENCE=stack_timeline is not None,
        START_STACK=stack_live is not None and stack_timeline is not None,
        HAND_SEGMENTATION=True, STREETS=True, ACTION_RECONSTRUCTION=reconstruction_enabled,
        POKER_LOGIC=True, MATH_VALIDATION=True, POT_STACK_CONSISTENCY=True,
        FINAL_VALIDATOR=True, HH_EXPORT=callable(hh_export), PERFORMANCE_DIAGNOSTICS=diagnostic is not None)
    return require_complete(values)
