"""Adversarial pixel contracts for the isolated experiment."""
import numpy as np
from experiments.vertical_chat_shift import correlation, reference, coordinate_map
from experiments.vertical_chat_fast_ncc import correlation_fft


def raster():
    rng=np.random.default_rng(45)
    image=np.full((180,132),20,np.uint8)
    spans=[(y,y+8) for y in range(10,161,15)]
    for y1,y2 in spans:
        image[y1:y2,4:110]=rng.integers(80,240,(y2-y1,106),dtype=np.uint8)
    return image,spans


def test_three_new_rows_keep_old_coordinates():
    old,os=raster()
    new=np.full_like(old,20); new[:-45]=old[45:]
    ns=[(a-45,b-45) for a,b in os if a>=45]
    for y in [130,145,160]:
        new[y:y+8,4:110]=np.random.default_rng(y).integers(80,240,(8,106),dtype=np.uint8)
        ns.append((y,y+8))
    ref=reference(old,new,os,ns)
    assert ref['shift']==45 and len(ref['new_indices'])==3
    alg=correlation(old,new,True)
    assert alg['shift']==45 and alg['accepted']
    assert coordinate_map(os,ns,45)==ref['mapping']


def test_append_without_scroll_disproves_division_rule():
    old,spans=raster(); os=spans[:6]
    old[100:]=20; new=old.copy(); new[100:108,4:110]=150
    ns=os+[(100,108)]
    ref=reference(old,new,os,ns)
    assert ref['shift']==0 and len(ref['new_indices'])==1
    assert correlation(old,new,True)['shift']==0


def test_identical_repeated_rows_are_ambiguous():
    old,spans=raster()
    for a,b in spans: old[a:b]=old[10:18]
    assert reference(old,old.copy(),spans,spans) is None


def test_inside_row_mutation_not_new_ground_truth():
    old,spans=raster(); new=old.copy(); new[55:63,20:40]=255
    ref=reference(old,new,spans,spans)
    assert ref['shift']==0 and not ref['count_known']
    assert ref['unknown_indices']==[3]


def test_full_replacement_has_no_identity_proof():
    old,spans=raster(); new=255-old
    assert reference(old,new,spans,spans) is None


def test_bottom_row_pixel_change_is_not_proven_new():
    old,spans=raster(); new=old.copy(); new[160:168,20:40]=255
    ref=reference(old,new,spans,spans)
    assert ref['shift']==0 and not ref['count_known']
    assert ref['new_indices']==[] and ref['unknown_indices']==[10]


def test_fft_keeps_shift_and_confidence_contract():
    old,_=raster()
    for shift in [0,1,15,45,100]:
        new=np.full_like(old,20)
        new[:len(old)-shift]=old[shift:]
        for mask in [False,True]:
            scalar=correlation(old,new,mask); fast=correlation_fft(old,new,mask)
            assert scalar['shift']==fast['shift']
            assert scalar['accepted']==fast['accepted']
            # Scalar float32 accumulation may produce NCC slightly above 1.
            # Decisions and shifts are exact; scores allow 2e-6 roundoff.
            assert abs(scalar['score']-fast['score'])<2e-6


def test_constant_pixels_cannot_establish_shift():
    a=np.full((180,132),20,np.uint8)
    assert not correlation_fft(a,a,True)['accepted']
