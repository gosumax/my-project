from parser_core.seat_occupancy import SeatOccupancyEvidence


KEY = (2, 'source-session', '1234567890')


def test_nickname90_does_not_lower_empty95_or_change_pixel_continuity():
    evidence=SeatOccupancyEvidence(.95,nickname_confidence=.90)
    for frame in range(3):
        for seat,text,score in [('seat1','Alice',.90),('seat2','Bob',.90),('seat3','OPEN SEAT',.90)]:
            sha=(text.encode().hex()+'0'*64)[:64]
            evidence.observe(KEY,seat,frame,sha)
            if frame==0:
                evidence.completed(KEY,seat,frame,sha,dict(status='OK',raw_text=text,score=score),pts=frame,time_base=(1,30))
    assert evidence.receipt(KEY,2,['seat1','seat2'])['occupied_seats']==['seat1','seat2']
    assert evidence.receipt(KEY,2,['seat1','seat2','seat3']) is None


def test_truncated_empty_text_remains_unknown_in_full_contract():
    for fragment in ('PENSEAT','PEN SEAT'):
        evidence=SeatOccupancyEvidence(reject_empty_fragments=True)
        for frame in range(3):feed(evidence,frame,dict(seat1='Player1',seat2=fragment))
        assert evidence.receipt(KEY,2,['seat1','seat2']) is None
        assert set(evidence.current_labels(KEY,2,['seat1','seat2']))=={'seat1'}


def feed(evidence, frame, labels, *, callback=True):
    for seat, label in labels.items():
        digest = (label.encode().hex()+'0'*64)[:64]
        evidence.observe(KEY, seat, frame, digest)
        if callback:
            evidence.completed(KEY, seat, frame, digest,
                dict(status='OK', raw_text=label, score=.99), pts=frame*512, time_base=(1,15360))


def test_six_seats_require_explicit_empty_evidence_and_three_sources():
    evidence = SeatOccupancyEvidence()
    labels = {f'seat{i}':f'Player{i}' if i<=6 else 'OPEN SEAT' for i in range(1,9)}
    for frame in (10,11):
        feed(evidence,frame,labels)
        assert evidence.receipt(KEY,frame,labels) is None
    feed(evidence,12,labels)
    receipt = evidence.receipt(KEY,12,labels)
    assert receipt['occupied_seats'] == list(labels)[:6]
    assert receipt['empty_seats'] == ['seat7','seat8']
    assert receipt['seats']['seat1']['observations'][0]['source_pts'] == 5120
    assert 'NO_HISTORICAL_IDENTITY_OR_STACK_AUTHORITY' in receipt['authority']
    # Identical source pixels continue the proven label without repeated OCR.
    feed(evidence,13,labels,callback=False)
    assert evidence.receipt(KEY,13,labels)
    assert evidence.receipt(KEY,12,labels) is None


def test_unknown_seat_overlay_truncation_and_duplicate_name_withhold_mask():
    for unknown in ('', 'ANTE', 'SIT OUT', 'Sit-out', 'RAISE 600', 'Player...', 'Player…', '?', '12345', 'Player1'):
        evidence=SeatOccupancyEvidence()
        labels=dict(seat1='Player1',seat2='Player2',seat3=unknown)
        for frame in range(3):feed(evidence,frame,labels)
        assert evidence.receipt(KEY,2,labels) is None


def test_source_change_and_gap_invalidate_even_if_old_ocr_arrives_late():
    for changed_frame, changed_text in ((3,'Different'),(4,'Player1')):
        evidence=SeatOccupancyEvidence()
        labels=dict(seat1='Player1',seat2='Player2')
        for frame in range(3):feed(evidence,frame,labels)
        assert evidence.receipt(KEY,2,labels)
        labels['seat1']=changed_text
        feed(evidence,changed_frame,labels,callback=False)
        evidence.completed(KEY,'seat1',2,('Player1'.encode().hex()+'0'*64)[:64],
            dict(status='OK',raw_text='Player1',score=.99),pts=1024,time_base=(1,15360))
        assert evidence.receipt(KEY,changed_frame,labels) is None


def test_one_new_ocr_recovers_after_pixel_change_with_current_pixel_continuity():
    evidence=SeatOccupancyEvidence()
    labels=dict(seat1='Player1',seat2='Player2')
    feed(evidence,0,labels)
    for frame in (1,2):feed(evidence,frame,labels,callback=False)
    assert evidence.receipt(KEY,2,labels)
    labels['seat1']='Different'
    feed(evidence,3,labels)
    assert evidence.receipt(KEY,3,labels) is None
    for frame in (4,5):feed(evidence,frame,labels,callback=False)
    receipt=evidence.receipt(KEY,5,labels)
    assert receipt['seats']['seat1']['actor']=='Different'
    assert [r['frame'] for r in receipt['seats']['seat1']['observations']]==[3]


def test_wrong_hand_low_confidence_and_unobserved_seat_never_supply_proof():
    evidence=SeatOccupancyEvidence()
    labels=dict(seat1='Player1',seat2='Player2')
    for frame in range(3):feed(evidence,frame,labels)
    assert evidence.receipt((2,'source-session','different'),2,labels) is None
    assert evidence.receipt(KEY,2,[*labels,'seat3']) is None
    evidence.completed(KEY,'seat1',2,('Player1'.encode().hex()+'0'*64)[:64],
        dict(status='OK',raw_text='Player1',score=.90),pts=1024,time_base=(1,15360))
    assert evidence.receipt(KEY,2,labels) is None


def test_same_ocr_text_on_different_pixels_does_not_prove_continuity():
    evidence=SeatOccupancyEvidence()
    for frame in range(3):
        for seat in ('seat1','seat2'):
            digest=f'{frame:064x}'
            evidence.observe(KEY,seat,frame,digest)
            evidence.completed(KEY,seat,frame,digest,
                dict(status='OK',raw_text=seat,score=.99),pts=frame*512,time_base=(1,15360))
    assert evidence.receipt(KEY,2,['seat1','seat2']) is None


def test_conflicting_high_confidence_labels_for_identical_pixels_withhold_mask():
    evidence=SeatOccupancyEvidence()
    labels=dict(seat1='Player1',seat2='Player2')
    for frame in range(3):feed(evidence,frame,labels)
    assert evidence.receipt(KEY,2,labels)
    digest=('Player1'.encode().hex()+'0'*64)[:64]
    evidence.completed(KEY,'seat1',2,digest,dict(status='OK',raw_text='PlayerX',score=.99),
                       pts=1024,time_base=(1,15360))
    assert evidence.receipt(KEY,2,labels) is None
    feed(evidence,3,labels)
    assert evidence.receipt(KEY,3,labels) is None
