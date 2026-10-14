"""CHAT card reading: whole-row VL ranks and hash-pinned CHAT templates for suits."""
from __future__ import annotations

import hashlib
import base64
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from parser_core.board_tiles import (card_spans, secondary_card_spans, _glyph_boxes,
    _glyphs, _scores, TEMPLATE_PATH, prove_split_card_partition, SPLIT_PARTITION_RULE)
from parser_core.suit_backup import SuitBackup, decide_suit
from parser_core.component_suit import ComponentSuitBackup
from parser_core.rank_ncc import RankNCC, rank_geometry_fallback


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_VL_RANK_TOKEN = re.compile(r"10|[2-9TJQKA]")
_STRUCTURED_RANK_TOKEN = re.compile(r"(?<![A-Z0-9])(?:10|[2-9TJQKA])(?=(?:[SHDC](?![A-Z0-9]))|[^A-Z0-9]|$)")
_PRODUCTION_RANK_POLICY = (.71, .015)
_DEVELOPMENT_RANK_POLICY = (.80, .01)


def parse_rank_tokens(text: str) -> list[str]:
    """Read literal rank tokens, never turn OCR garbage into a card rank."""
    return ["T" if value == "10" else value for value in _STRUCTURED_RANK_TOKEN.findall(text.upper())]


class CardHybrid:
    def __init__(self, config_path: Path, suit_config_path: Path, *, rank_ncc_config: Path | None = None):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        manifest = Path(config["train_manifest"])
        if config["method"] != "raw_suit2" or _sha(manifest.read_bytes()) != config["train_manifest_sha256"]:
            raise ValueError("Card template selection or manifest changed")
        selection = Path(config["selection_file"])
        if _sha(selection.read_bytes()) != config["selection_sha256"]:
            raise ValueError("Card selection evidence changed")
        chosen = json.loads(selection.read_text(encoding="utf-8"))
        if chosen["selected_method"] != config["method"] or chosen["smoke16_used_for_selection"] is not False:
            raise ValueError("Card template method was not validation-selected")
        bank, labels = [], []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            source = Path(item["tile_png"]).read_bytes()
            if _sha(source) != item["tile_sha256"]:
                raise ValueError("Card template PNG changed")
            with Image.open(io.BytesIO(source)) as opened:
                rgb = opened.convert("RGB")
                if _sha(rgb.tobytes()) != item["tile_pixel_sha256"] or rgb.height != 16 or rgb.width not in (20, 21):
                    raise ValueError("Card template pixels or geometry changed")
                bank.append(np.asarray(rgb.convert("L"))[:, :20].copy())
            labels.append(item["ground_truth"])
        if len(bank) != 431 or len(set(labels)) != 52:
            raise ValueError("Card template bank class count changed")
        self.bank = np.stack(bank).astype(np.int16)
        self.labels = labels
        # Derive exact rank masks only from the already hash-verified TRAIN
        # pixels. No runtime labels, source identities or query hashes enter
        # this channel. Nonzero nearest-template distance never repairs a rank.
        self.rank_masks = []
        self.rank_labels = []
        self.rank_train_indices = []
        for index, tile in enumerate(bank):
            glyphs = _glyphs(tile, 0, 19)
            if glyphs is not None:
                self.rank_masks.append(glyphs[0])
                self.rank_labels.append(labels[index][0])
                self.rank_train_indices.append(index)
        self.rank_masks = np.asarray(self.rank_masks, dtype=np.uint8)
        legacy_bytes = TEMPLATE_PATH.read_bytes()
        self.legacy_rank_model = json.loads(legacy_bytes)["ranks"]
        self.legacy_rank_sha256 = _sha(legacy_bytes)
        self.config_sha256 = _sha(config_path.read_bytes())
        self.train_manifest_sha256 = config["train_manifest_sha256"]
        self.selection_sha256 = config["selection_sha256"]
        self.suit_backup = SuitBackup(suit_config_path)
        self.component_suit = (ComponentSuitBackup(suit_config_path)
                               if self.suit_backup.mode == "BACKUP" else None)
        self.rank_ncc = None
        self.rank_mode = "OFF"
        self.rank_ncc_config_sha256 = None
        self.development_only = False
        if rank_ncc_config is not None:
            rank_bytes = Path(rank_ncc_config).read_bytes()
            rank_config = json.loads(rank_bytes)
            production_rank = (rank_config.get("mode") == "PIXEL_ONLY" and
                               rank_config.get("production_authorized") is True and
                               rank_config.get("production_enabled") is True)
            rank_policy = (_PRODUCTION_RANK_POLICY if production_rank else
                           _DEVELOPMENT_RANK_POLICY)
            if (rank_config.get("mode") not in ("FALLBACK", "PIXEL_ONLY") or
                    not (rank_config.get("development_only") is True or
                         (rank_config.get("mode") == "PIXEL_ONLY" and
                          rank_config.get("production_authorized") is True and
                          rank_config.get("production_enabled") is True)) or
                    rank_config.get("method") != "component_ncc" or
                    rank_config.get("best_score_threshold") != rank_policy[0] or
                    rank_config.get("margin_threshold") != rank_policy[1] or
                    self.suit_backup.mode != "BACKUP"):
                raise ValueError("Rank policy is not the registered production or frozen development policy")
            implementation = Path(__file__).with_name("rank_ncc.py")
            if _sha(implementation.read_bytes()) != rank_config["implementation_sha256"]:
                raise ValueError("Frozen RankNCC implementation changed")
            if (rank_config["train_manifest_sha256"] != self.train_manifest_sha256):
                raise ValueError("Rank and suit template provenance disagree")
            self.rank_ncc = RankNCC(Path(rank_config["train_manifest"]), method="component_ncc",
                                   threshold=rank_policy[0], margin=rank_policy[1],
                                   expected_manifest_sha256=rank_config["train_manifest_sha256"])
            self.rank_mode = rank_config["mode"]
            self.development_only = rank_config.get("development_only", True)
            self.rank_ncc_config_sha256 = _sha(rank_bytes)

    def read(self, row_png: Path, row_sha256: str, vl_text: str, tiles_dir: Path,
             primary_cards: list[str] | None = None) -> dict:
        return self.read_bytes(row_png.read_bytes(), row_sha256, vl_text,
                               tiles_dir=tiles_dir, primary_cards=primary_cards)

    def read_bytes(self, raw: bytes, row_sha256: str, vl_text: str, *,
                   tiles_dir: Path | None = None,
                   primary_cards: list[str] | None = None) -> dict:
        if _sha(raw) != row_sha256:
            raise ValueError("CHAT row PNG SHA mismatch")
        with Image.open(io.BytesIO(raw)) as opened:
            gray = np.asarray(opened.convert("L"))
        result = {"card_reader": "vl_rank_template_suit_v1", "rank_source": "VL_WHOLE_CHAT_ROW",
                  "vl_rank_source": "VL_WHOLE_CHAT_ROW",
                  "suit_source": "VERIFIED_TRAIN_CHAT_TEMPLATE_RAW_SUIT2",
                  "template_manifest_sha256": self.train_manifest_sha256,
                  "template_selection_sha256": self.selection_sha256,
                  "template_config_sha256": self.config_sha256,
                  "right_trim_px": 12, "tiles": []}
        if self.rank_ncc is not None:
            result.update(rank_mode=self.rank_mode,
                          rank_ncc_config_sha256=self.rank_ncc_config_sha256,
                           development_only=self.development_only)
            if self.rank_mode == "PIXEL_ONLY":
                result.update(card_reader="pixel_rank_ncc_template_suit_v1", vl_rank_source="NOT_USED")
        if gray.shape[0] != 16 or gray.shape[1] <= 112:
            return {**result, "reason": "UNSUPPORTED_CHAT_ROW_GEOMETRY"}
        model = gray[:, :-12]
        spans, recovered_spans = card_spans(model, recover_split=self.suit_backup.mode == "BACKUP")
        primary_spans = spans
        geometry_valid = (2 <= len(spans) <= 5 and all(
            2 <= spans[i+1][0]-spans[i][1]-1 <= 5 for i in range(len(spans)-1)))
        if self.suit_backup.mode == "BACKUP":
            secondary, secondary_evidence = secondary_card_spans(model)
            if geometry_valid and len(secondary) > len(primary_spans):
                # A valid-looking subset is not a complete physical card row.
                # The independently supported y2 geometry may expose a card
                # whose y3 background was split by ink. Keep both native box
                # sets; neither a rank score nor VL can erase this conflict.
                proof = prove_split_card_partition(model)
                if (proof is None or proof['chosen_spans_inclusive'] != [list(s) for s in secondary]
                        or proof['primary_spans_inclusive'] != [list(s) for s in primary_spans]):
                    return {**result, "reason": "CARD_SPAN_COVERAGE_UNRESOLVED",
                            "detected_spans_inclusive": [list(span) for span in primary_spans],
                            "secondary_candidate_spans_inclusive": secondary,
                            "secondary_geometry_evidence": secondary_evidence,
                            "recovered_split_spans": recovered_spans,
                            "row_geometry_source": "PRIMARY_SECONDARY_COVERAGE_CONFLICT"}
                spans = secondary
                recovered_spans = recovered_spans+proof['bridged_spans']
                result.update(row_geometry_source=SPLIT_PARTITION_RULE,
                    secondary_partition_proof=proof,
                    primary_incomplete_spans_inclusive=[list(s) for s in primary_spans])
            if not geometry_valid and secondary:
                spans, recovered_spans = secondary, recovered_spans+secondary_evidence
                result["primary_rejected_spans_inclusive"] = primary_spans
                result["row_geometry_source"] = "SECONDARY_NATIVE_BACKGROUND_Y2"
        # This public receipt is consumed before and after JSON persistence.
        # Emit its declared array shape here; persistence must not repair it.
        result["detected_spans_inclusive"] = [list(span) for span in spans]
        result["recovered_split_spans"] = recovered_spans
        if not 2 <= len(spans) <= 5 or any(not 2 <= spans[i+1][0]-spans[i][1]-1 <= 5 for i in range(len(spans)-1)):
            return {**result, "reason": "CARD_SPAN_COUNT_OR_GAP_UNRESOLVED"}
        ranks = (parse_rank_tokens(vl_text) if self.suit_backup.mode == "BACKUP" else
                 ["T" if value == "10" else value for value in _VL_RANK_TOKEN.findall(vl_text.upper())])
        result["vl_rank_tokens"] = ranks
        rank_aligned = len(ranks) == len(spans)
        result["rank_alignment_status"] = "ALIGNED" if rank_aligned else "UNRESOLVED"
        primary_rank_conflict = (primary_cards is not None and rank_aligned and
                                 len(primary_cards) == len(ranks) and
                                 any(card[0] != rank for card, rank in zip(primary_cards, ranks)))
        result["primary_vl_rank_conflict"] = primary_rank_conflict
        cards = []
        if tiles_dir is not None:
            tiles_dir.mkdir(parents=True, exist_ok=True)
        for index, (start, end) in enumerate(spans):
            rank = ranks[index] if rank_aligned else None
            vl_rank = rank
            if self.rank_mode == "PIXEL_ONLY":
                rank = None
            rank_source = "VL_POSITIONAL" if rank_aligned and self.rank_mode != "PIXEL_ONLY" else "UNRESOLVED"
            rank_status = None
            rank_matches = []
            exact_rank = None
            rank_scores = []
            glyphs = _glyphs(model, start, end)
            if glyphs is not None and len(self.rank_masks):
                hits = np.flatnonzero(np.all(self.rank_masks == glyphs[0], axis=(1, 2)))
                rank_matches = [self.rank_train_indices[int(i)] for i in hits]
                exact_labels = {self.rank_labels[int(i)] for i in hits}
                exact_rank = next(iter(exact_labels)) if len(exact_labels) == 1 else None
                rank_scores = _scores(glyphs[0], self.legacy_rank_model)
            if self.suit_backup.mode == "BACKUP":
                if exact_rank is not None:
                    rank = exact_rank
                    rank_source = "EXACT_VERIFIED_TRAIN_RANK_GLYPH"
                    rank_status = "VL_RANK_CORRECTED_FROM_PIXELS" if vl_rank and vl_rank != rank else None
                elif rank and len(rank_scores) >= 2:
                    legacy_rank, best = rank_scores[0]
                    legacy_rank = "T" if legacy_rank == "10" else legacy_rank
                    distance = next((score for label, score in rank_scores if
                                     ("T" if label == "10" else label) == rank), None)
                    # Reuse the existing strong-contradiction gate as a veto,
                    # not as a nearest-template rank replacement.
                    if (legacy_rank != rank and best <= 8 and rank_scores[1][1]-best >= 4 and
                            (distance is None or distance-best >= 8)):
                        rank = None
                        rank_source = "UNRESOLVED"
                        rank_status = "RANK_VISUAL_CONFLICT"
            rgb = Image.fromarray(model[:, start:end+1]).convert("RGB")
            pixel_rank = self.rank_ncc.read(np.asarray(rgb)) if self.rank_ncc is not None else None
            if pixel_rank is not None:
                pixel_rank = rank_geometry_fallback(np.asarray(rgb), pixel_rank)
            if pixel_rank is not None and rank is None and pixel_rank.get("rank_accepted"):
                rank = pixel_rank["rank_result"]
                rank_source = "FROZEN_RANK_NCC_PIXEL_FALLBACK"
                rank_status = "PIXEL_RANK_FALLBACK"
            buffer = io.BytesIO()
            rgb.save(buffer, format="PNG")
            payload = buffer.getvalue()
            path = tiles_dir / f"{index:02d}.png" if tiles_dir is not None else None
            if path is not None:
                if path.exists() and path.read_bytes() != payload:
                    raise ValueError("Existing CARD tile conflicts with source pixels")
                path.write_bytes(payload)
            evidence = {"card_index": index, "box_xyxy_exclusive": [start, 0, end+1, 16],
                        "tile_png": str(path.resolve()) if path is not None else None,
                        "tile_png_base64": base64.b64encode(payload).decode("ascii") if path is None else None,
                        "tile_sha256": _sha(payload), "tile_pixel_sha256": _sha(rgb.tobytes()),
                        "vl_rank": vl_rank or "UNRESOLVED", "final_rank": rank or "UNRESOLVED",
                        "rank_source": rank_source, "rank_status": rank_status,
                        "rank_exact_candidate": exact_rank or "UNRESOLVED",
                        "rank_exact_train_indices": rank_matches,
                        "legacy_rank_scores": rank_scores,
                        "legacy_rank_template_sha256": self.legacy_rank_sha256,
                        "size_wh": [rgb.width, rgb.height],
                        "detected_bright_box_xyxy_exclusive": [start, 0, end+1, 16],
                        "tile_geometry_source": "BRIGHT_SPAN"}
            if pixel_rank is not None:
                evidence["rank_ncc"] = pixel_rank
            base_suit = label = best = second = None
            if rgb.height == 16 and rgb.width in (20, 21):
                query = np.asarray(rgb.convert("L"))[:, :20].astype(np.int16)
                diff = np.abs(self.bank[:, 2:14, 1:19] - query[2:14, 1:19])
                diff[:, :, 9:] *= 2
                distances = diff.mean(axis=(1, 2))
                order = np.argsort(distances, kind="stable")
                best_idx = int(order[0])
                label = self.labels[best_idx]
                other_idx = next(int(j) for j in order[1:] if self.labels[int(j)] != label)
                best, second = float(distances[best_idx]), float(distances[other_idx])
                base_suit = label[1] if best < second else None
                backup = self.suit_backup.read(np.asarray(rgb))
                # A fixed tile-origin ROI can be one pixel out of phase with
                # the native suit component. Retry only an unconfident BACKUP
                # query at the independently detected component origin.
                if self.suit_backup.mode == "BACKUP" and not backup.get("template_confident"):
                    boxes = _glyph_boxes(model, start, end)
                    if boxes is not None:
                        sx, sy, sw, sh = boxes[1]
                        qx = sx-10
                        if (qx != start and 0 <= qx and qx+20 <= model.shape[1] and
                                sw <= 9 and 2 <= sy and sy+sh <= 13 and sx+9 <= end+1):
                            query_rgb = Image.fromarray(model[:, qx:qx+20]).convert("RGB")
                            localized = self.suit_backup.read(np.asarray(query_rgb))
                            if localized.get("template_confident"):
                                original_backup = backup
                                backup = localized
                                query_buffer = io.BytesIO()
                                query_rgb.save(query_buffer, format="PNG")
                                query_payload = query_buffer.getvalue()
                                evidence.update(ncc_original_query_result=original_backup,
                                    ncc_query_box_xyxy_exclusive=[qx, 0, qx+20, 16],
                                    ncc_suit_roi_box_xyxy_exclusive=[sx, 2, sx+9, 13],
                                    ncc_query_png_base64=base64.b64encode(query_payload).decode("ascii"),
                                    ncc_query_sha256=_sha(query_payload),
                                    ncc_query_pixel_sha256=_sha(query_rgb.tobytes()),
                                    ncc_geometry_source="SUIT_COMPONENT_NATIVE_ROW_ROI",
                                    suit_glyph_box_xywh=[sx, sy, sw, sh])
            else:
                backup = self.suit_backup.read(np.asarray(rgb))
                boxes = _glyph_boxes(model, start, end) if rgb.width in (18, 19) else None
                if boxes is not None:
                    sx, sy, sw, sh = boxes[1]
                    # Align only the suit ROI, using the existing pixel glyph
                    # locator. The detected tile itself is never widened.
                    qx = sx-10
                    if 0 <= qx and qx+20 <= model.shape[1] and sw <= 9 and 2 <= sy and sy+sh <= 13 and sx+9 <= end+1:
                        query_rgb = Image.fromarray(model[:, qx:qx+20]).convert("RGB")
                        query_buffer = io.BytesIO()
                        query_rgb.save(query_buffer, format="PNG")
                        query_payload = query_buffer.getvalue()
                        backup = self.suit_backup.read(np.asarray(query_rgb))
                        evidence.update(ncc_query_box_xyxy_exclusive=[qx, 0, qx+20, 16],
                            ncc_suit_roi_box_xyxy_exclusive=[sx, 2, sx+9, 13],
                            ncc_query_png_base64=base64.b64encode(query_payload).decode("ascii"),
                            ncc_query_sha256=_sha(query_payload),
                            ncc_query_pixel_sha256=_sha(query_rgb.tobytes()),
                            ncc_geometry_source="SUIT_COMPONENT_NATIVE_ROW_ROI",
                            suit_glyph_box_xywh=[sx, sy, sw, sh])
            if self.component_suit is not None and not backup.get("template_confident", False):
                component = self.component_suit.read(np.asarray(rgb))
                evidence["component_suit_result"] = component
                if component.get("template_confident"):
                    evidence.setdefault("ncc_original_query_result", backup)
                    evidence["component_suit_original_result"] = backup
                    backup = component
                    evidence["ncc_geometry_source"] = "NATIVE_SUIT_COMPONENT_NORMALIZED"
            legacy_suit = (primary_cards[index][-1] if self.rank_mode != "PIXEL_ONLY" and primary_cards is not None and
                           len(primary_cards) == len(spans) else None)
            legacy_confident = legacy_suit is not None
            # The validation-selected native channel owns the primary in BACKUP.
            # A legacy binary glyph result is independent diagnostic evidence;
            # it must not silently replace the selected primary channel.
            primary_suit = ((base_suit if base_suit is not None else legacy_suit)
                            if self.suit_backup.mode == "BACKUP" else
                            (legacy_suit if legacy_suit is not None else base_suit))
            primary_confident = legacy_confident and primary_suit == legacy_suit
            decision = decide_suit(primary_suit, primary_confident, backup)
            if (self.suit_backup.mode == "BACKUP" and base_suit is not None and
                    backup.get("template_confident") and base_suit != backup.get("template_suit")):
                decision = {"suit": "UNRESOLVED", "suit_source": "conflict", "suit_status": "SUIT_CONFLICT"}
            suit = decision["suit"] if decision["suit"] != "UNRESOLVED" else None
            if (self.suit_backup.mode == "BACKUP" and not primary_confident and
                    not backup.get("template_confident", False)):
                suit = None
                decision = {"suit": "UNRESOLVED", "suit_source": "UNRESOLVED",
                            "suit_status": backup.get("template_reason") or "TEMPLATE_LOW_CONFIDENCE"}
            tile = {**evidence,
                    "nearest_train_card": label, "raw_suit2_suit": base_suit or "UNRESOLVED",
                    "primary_suit": primary_suit or "UNRESOLVED",
                    "legacy_board_suit": legacy_suit or "UNRESOLVED",
                    "primary_channel": "SELECTED_NATIVE_RAW_SUIT2" if self.suit_backup.mode == "BACKUP" and base_suit is not None else "LEGACY_OR_UNAVAILABLE",
                    "primary_confidence": "BOARD_TILE_ACCEPTED" if primary_confident else None,
                    "best_distance": best, "runner_up_other_card_distance": second,
                    **backup, **decision,
                    "card": rank+suit if rank and suit else "UNRESOLVED"}
            result["tiles"].append(tile)
            cards.append(tile["card"])
        if self.suit_backup.mode == "BACKUP":
            sources = {tile["rank_source"] for tile in result["tiles"] if tile["rank_source"] != "UNRESOLVED"}
            result["rank_source"] = ("EXACT_VERIFIED_TRAIN_RANK_GLYPH" if sources == {"EXACT_VERIFIED_TRAIN_RANK_GLYPH"}
                                     else "VL_WHOLE_CHAT_ROW" if sources == {"VL_POSITIONAL"}
                                     else "MIXED_VL_AND_EXACT_PIXEL_TEMPLATES" if sources else "UNRESOLVED")
            if self.rank_mode == "PIXEL_ONLY":
                result["rank_source"] = "PIXEL_ONLY_EXACT_AND_RANK_NCC" if sources else "UNRESOLVED"
            elif "FROZEN_RANK_NCC_PIXEL_FALLBACK" in sources:
                result["rank_source"] = ("FROZEN_RANK_NCC_PIXEL_FALLBACK" if sources == {"FROZEN_RANK_NCC_PIXEL_FALLBACK"}
                                         else "MIXED_VL_AND_PIXEL_RANK_CHANNELS")
        final_ranks_complete = all(tile["final_rank"] != "UNRESOLVED" for tile in result["tiles"])
        if not rank_aligned and not final_ranks_complete:
            return {**result, "reason": "PIXEL_RANK_UNRESOLVED" if self.rank_mode == "PIXEL_ONLY" else "VL_RANK_COUNT_MISMATCH"}
        if not rank_aligned:
            result["rank_alignment_status"] = ("INDEPENDENT_PIXEL_RANK_CHANNELS" if any(
                tile["rank_source"] == "FROZEN_RANK_NCC_PIXEL_FALLBACK" for tile in result["tiles"])
                else "INDEPENDENT_EXACT_TILE_RANKS")
        if not final_ranks_complete:
            return {**result, "reason": "RANK_VISUAL_CONFLICT"}
        if primary_rank_conflict and self.suit_backup.mode != "BACKUP":
            return {**result, "reason": "PRIMARY_VL_RANK_CONFLICT"}
        if any(tile.get("suit_status") == "UNSUPPORTED_TILE_GEOMETRY" for tile in result["tiles"]):
            return {**result, "reason": "UNSUPPORTED_TILE_GEOMETRY"}
        if any(tile.get("suit_status") == "SUIT_CONFLICT" for tile in result["tiles"]):
            return {**result, "reason": "SUIT_CONFLICT"}
        if any(card == "UNRESOLVED" for card in cards):
            return {**result, "reason": "SUIT_UNRESOLVED"}
        if len(set(cards)) != len(cards):
            return {**result, "reason": "DUPLICATE_CARD_CONFLICT"}
        return {**result, "cards": cards, "reason": None}
