import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from parser_core.card_hybrid import CardHybrid, _VL_RANK_TOKEN
from parser_core.suit_backup import SuitBackup, decide_suit


ROOT = Path(__file__).resolve().parents[1]


def template(suit="d", confident=True, mode="BACKUP"):
    return {"template_suit": suit, "template_confident": confident,
            "suit_backup_mode": mode}


def test_primary_template_agreement():
    assert decide_suit("d", True, template()) == {"suit": "d", "suit_source": "agreement"}


def test_unknown_or_low_confidence_primary_uses_confident_template():
    assert decide_suit(None, False, template())["suit"] == "d"
    decision = decide_suit("c", False, template())
    assert decision["suit"] == "d"
    assert decision["suit_status"] == "PRIMARY_UNCALIBRATED_DISAGREEMENT"


def test_confident_primary_conflict_is_unresolved():
    decision = decide_suit("c", True, template())
    assert decision["suit"] == "UNRESOLVED"
    assert decision["suit_status"] == "SUIT_CONFLICT"


def test_low_confidence_template_cannot_override_and_shadow_is_primary_only():
    assert decide_suit(None, False, template(confident=False))["suit"] == "UNRESOLVED"
    assert decide_suit("c", True, template(confident=False))["suit"] == "c"
    assert decide_suit("c", False, template(mode="SHADOW"))["suit"] == "c"


def test_ncc_matches_frozen_validation_and_hash_pinned_bank():
    reader = SuitBackup(ROOT / "configs/suit_backup_local.json")
    assert reader.validation_pass == 312
    with (ROOT / "runs/suit_template_experiment_20260930/results.csv").open(encoding="utf-8-sig") as stream:
        records = [r for r in csv.DictReader(stream) if r["split"] == "validation" and
                   r["method"] == "grayscale_ncc" and r["size"] == "16"]
    assert {Path(r["source_video"]).name for r in records} == {"smoke_14.mp4", "smoke_15.mp4"}
    for row in records:
        with Image.open(row["tile_png"]) as opened:
            rgb = np.asarray(opened.convert("RGB"))
        result = reader.read(rgb)
        assert result["template_suit"] == row["predicted_suit"]
        assert abs(result["template_best_score"] - float(row["best_score"])) < 1e-6
        assert abs(result["template_margin"] - float(row["margin"])) < 1e-6


def test_whole_row_rank_plus_suit_keeps_card_and_tile_provenance(tmp_path):
    record = json.loads((ROOT / "runs/card_train_smoke16_test_20260930/current_production_rank_probe/raw_responses.jsonl")
                        .read_text(encoding="utf-8").splitlines()[0])
    reader = CardHybrid(ROOT / "configs/card_hybrid_local.json",
                        ROOT / "configs/suit_backup_local.json")
    result = reader.read(Path(record["chat_row_png"]), record["chat_row_sha256"],
                         record["vl_response"]["raw_text"], tmp_path)
    assert result["cards"] == record["ground_truth_adjudicated"]
    assert len(result["tiles"]) == 5
    assert all(t["tile_sha256"] and t["tile_pixel_sha256"] for t in result["tiles"])


def test_vl_rank_tokens_include_ten_as_t_or_10():
    assert _VL_RANK_TOKEN.findall("Ts 10♦ 5h") == ["T", "10", "5"]


def test_low_ncc_confidence_does_not_accept_uncalibrated_raw_suit(tmp_path):
    record = json.loads((ROOT / "runs/card_train_smoke16_test_20260930/current_production_rank_probe/raw_responses.jsonl")
                        .read_text(encoding="utf-8").splitlines()[0])
    reader = CardHybrid(ROOT / "configs/card_hybrid_local.json",
                        ROOT / "configs/suit_backup_local.json")
    reader.suit_backup.read = lambda _: template(confident=False)
    reader.component_suit.read = lambda _: template(confident=False)
    result = reader.read(Path(record["chat_row_png"]), record["chat_row_sha256"],
                         record["vl_response"]["raw_text"], tmp_path)
    assert result["reason"] == "SUIT_UNRESOLVED"
    assert all(t["suit_status"] == "TEMPLATE_LOW_CONFIDENCE" for t in result["tiles"])
