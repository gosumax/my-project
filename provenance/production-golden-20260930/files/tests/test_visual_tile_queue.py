from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from parser_core.visual_tile_queue import VisualTileQueue


@dataclass
class Row:
    table_id: int = 1
    frame_id: int = 5
    source_span_index: int = 2
    session_id: str = "session"
    video_time: float = .2
    source_pts: int = 3072
    time_base_num: int = 1
    time_base_den: int = 15360
    pixel_sha256: str = "raw-pixel-hash"
    pixels: np.ndarray = None


def test_vl_failure_preserves_crop_and_raw_response(tmp_path):
    class Client:
        def read(self, path, sha, trim):
            assert trim == 12
            assert path.name == "000000005_002.png"
            assert np.asarray(Image.open(path)).shape == (16, 226)
            return {"status": "ERROR", "raw_text": "partial glyph", "error": "model failed"}

        def close(self):
            pass

    pixels = np.zeros((16, 226), dtype=np.uint8)
    pixels[:, :50] = 255
    pool = VisualTileQueue(lambda _: Client(), tmp_path, workers=1, queue_size=1)
    assert pool.submit(Row(pixels=pixels))
    pool.finish()
    result = pool.results["1:5:2"]
    assert result["status"] == "UNRESOLVED"
    assert result["vl_raw_text"] == "partial glyph"
    assert result["source_pts"] == 3072
    assert result["crop_sha256"]
    assert pool.max_queue <= 1


def test_regular_row_does_not_call_vl(tmp_path):
    pool = VisualTileQueue(lambda _: None, tmp_path, workers=1, queue_size=1)
    assert not pool.submit(Row(pixels=np.zeros((16, 226), dtype=np.uint8)))
    pool.finish()
    assert not pool.results


def test_strict_vl_text_is_saved_without_accepting_unverified_cards(tmp_path):
    class Client:
        def read(self, path, sha, trim):
            return {"status": "OK", "raw_text": "Ah 7c 2s"}

        def close(self):
            pass

    pixels = np.zeros((16, 226), dtype=np.uint8)
    pixels[:, :50] = 255
    pool = VisualTileQueue(lambda _: Client(), tmp_path, workers=1, queue_size=1)
    assert pool.submit(Row(pixels=pixels))
    pool.finish()
    result = pool.results["1:5:2"]
    assert result["vl_text_cards_unverified"] == ["Ah", "7c", "2s"]
    assert result["status"] == "UNRESOLVED"


@pytest.mark.parametrize("mode", ["OFF", "SHADOW", "BACKUP"])
def test_hybrid_mode_controls_publication(tmp_path, mode):
    class Client:
        def read(self, path, sha, trim):
            return {"status": "OK", "raw_text": "Qd 7c"}
        def close(self):
            pass
    class Hybrid:
        suit_backup = SimpleNamespace(mode=mode)
        def read(self, *args, **kwargs):
            return {"cards": ["Qd", "7c"], "tiles": [], "reason": None,
                    "card_reader": "test_hybrid"}
    pixels = np.zeros((16, 226), dtype=np.uint8)
    pixels[:, :50] = 255
    pool = VisualTileQueue(lambda _: Client(), tmp_path, workers=1, queue_size=1)
    pool.hybrid = Hybrid()
    assert pool.submit(Row(pixels=pixels))
    pool.finish()
    result = pool.results["1:5:2"]
    assert result["hybrid"]["cards"] == ["Qd", "7c"]
    if mode == "BACKUP":
        assert result["status"] == "RAW_UNREVIEWED"
        assert result["cards"] == ["Qd", "7c"]
    else:
        assert result["status"] == "UNRESOLVED"
        assert "cards" not in result
