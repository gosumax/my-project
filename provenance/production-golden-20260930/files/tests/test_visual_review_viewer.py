import json
import re

import pytest
from PIL import Image

from parser_core.annotation import sha256, write_json
from scripts.build_visual_review_viewer import build


def test_viewer_shows_all_frozen_frames_without_ocr(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    evidence = []
    for frame in (0, 1, 2):
        for kind in ("chat", "table"):
            path = corpus / f"{frame}_{kind}.png"
            Image.new("L", (20, 10), frame * 40).save(path)
            evidence.append({"evidence_id": path.stem, "frame_id": frame,
                "roi_type": kind, "source_pts": frame * 512, "time_base": [1, 15360],
                "local": path.name, "sha256": sha256(path), "size": [20, 10]})
    write_json(corpus / "manifest.json", {"evidence": evidence, "units": [{
        "unit_id": "unit_0000", "table_session_id": "session",
        "split": "test_candidate", "evidence_ids": [e["evidence_id"] for e in evidence]}]})
    (corpus / "manifest.sha256").write_text(sha256(corpus / "manifest.json"))
    output = tmp_path / "review" / "viewer.html"
    result = build(corpus, output)
    assert result["frames"] == 3
    assert result["verified_png"] == 6
    html = output.read_text(encoding="utf-8")
    assert "0_chat" in html and "2_table" in html
    assert "bbox" in html and "raw_ocr_hint" not in html
    assert "Править текст" in html
    assert "Править сценарий" in html
    assert '<select id="completeness"><option>UNKNOWN</option>' in html
    assert '<input id="scenarios" type="text" placeholder=' in html
    assert "review_status:'UNREVIEWED'" in html
    payload = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',
                                   html, re.S).group(1))
    frames = payload["units"][0]["frames"]
    assert [f["frame_id"] for f in frames] == [0, 1, 2]
    assert all((output.parent / f[kind]["image"]).is_file()
               for f in frames for kind in ("chat", "table"))
    with pytest.raises(ValueError, match="new viewer path"):
        build(corpus, output)
    Image.new("L", (20, 10), 255).save(corpus / "0_chat.png")
    with pytest.raises(ValueError, match="hash changed"):
        build(corpus, tmp_path / "other.html")
