"""Exact decoder parity for the PNG-compatible RGB grayscale conversion."""
import cv2
import numpy as np
import pytest

@pytest.mark.parametrize('strided',[False,True])
def test_png_compatible_gray_matches_original_codec_for_rgb_cube(strided):
    from parser_core.ram_first_stage import png_compatible_gray
    green,blue=np.meshgrid(np.arange(256,dtype=np.uint8),np.arange(256,dtype=np.uint8),indexing='ij')
    for red in (0,1,63,127,128,191,254,255):
        rgb=np.stack([np.full_like(green,red),green,blue],axis=2)
        if strided:
            padded=np.zeros((256,512,3),dtype=np.uint8);padded[:,::2]=rgb;rgb=padded[:,::2]
        before=rgb.copy()
        ok,packed=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,0])
        assert ok
        expected=cv2.imdecode(packed,cv2.IMREAD_GRAYSCALE)
        actual=png_compatible_gray(rgb)
        assert actual.dtype==np.uint8 and actual.shape==rgb.shape[:2]
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(rgb,before)

@pytest.mark.parametrize('pixels', [np.zeros((2,2),dtype=np.uint8),np.zeros((2,2,4),dtype=np.uint8),np.zeros((2,2,3),dtype=np.float32)])
def test_png_compatible_gray_rejects_non_native_rgb(pixels):
    from parser_core.ram_first_stage import png_compatible_gray
    with pytest.raises(ValueError,match='uint8 RGB'): png_compatible_gray(pixels)

def test_png_compatible_gray_rgb_corner_values():
    from parser_core.ram_first_stage import png_compatible_gray
    rgb=np.array([[[0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255],[1,0,0],[0,1,0],[0,0,1]]],dtype=np.uint8)
    assert png_compatible_gray(rgb).tolist()==[[0,255,76,149,29,0,0,0]]
