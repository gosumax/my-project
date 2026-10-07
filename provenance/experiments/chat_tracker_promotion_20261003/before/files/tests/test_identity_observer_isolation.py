"""The diagnostic inherited-row observer must not alter capture output."""
from fractions import Fraction
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import parser_core.ram_first_stage as module

def test_mutating_identity_observer_cannot_change_rows_pixels_or_identity(monkeypatch):
    images=[]
    for i in range(3):
        rgb=np.zeros((160,226,3),dtype=np.uint8)
        for j,y in enumerate((8,38,68)):
            rgb[y:y+8,12:112]=100+j*40
            rgb[y+2:y+6,25+j*7:39+j*7]=230
        rgb[150,225]=i
        images.append(rgb)
    class Frame:
        width=226;height=160;time_base=Fraction(1,30)
        def __init__(self,i):self.pts=i
        def to_ndarray(self,format):assert format=='rgb24';return images[self.pts].copy()
    class Source:
        streams=SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1,30))])
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def decode(self,stream):return iter(Frame(i) for i in range(3))
    layout=SimpleNamespace(source_width=226,source_height=160,slots=[SimpleNamespace(slot_id=1,table_box=(0,0,226,160),chat_box=(0,0,226,160))])
    monkeypatch.setattr(module,'load_fixed_layout',lambda *args:layout)
    monkeypatch.setattr(module.av,'open',lambda *args:Source())
    baseline=module.FirstStage(Path('synthetic.mp4'),{'1':'s'},3,root=Path('.'))
    plain=list(baseline.rows());calls=[]
    def observer(metadata,pixels):
        calls.append((metadata.copy(),pixels.copy()));pixels[:]=0;metadata.clear()
    watched=module.FirstStage(Path('synthetic.mp4'),{'1':'s'},3,root=Path('.'),identity_observer=observer)
    altered=list(watched.rows());assert calls
    assert baseline.seen_frames==watched.seen_frames==3
    assert len(plain)==len(altered)==2
    for a,b in zip(plain,altered):
        assert (a.frame_id,a.table_id,a.source_span_index,a.stable_row_id,a.pixel_sha256,a.origin,a.epoch)==(b.frame_id,b.table_id,b.source_span_index,b.stable_row_id,b.pixel_sha256,b.origin,b.epoch)
        assert np.array_equal(a.pixels,b.pixels)
