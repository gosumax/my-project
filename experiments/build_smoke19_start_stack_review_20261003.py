#!/usr/bin/env python3
"""Build an offline, provenance-linked Smoke19 visual review gallery.

This helper reads completed official-run evidence, exports existing RAM crops,
and decodes only marker-source CHAT row crops from the original video. It never
invokes the production pipeline or writes beneath acceptance_ON/.
"""
from __future__ import annotations

import base64
import collections
import hashlib
import html
import json
import re
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/smoke19_full_start_stack_validation_20261003/acceptance_ON"
OUT = ROOT / "runs/smoke19_full_start_stack_validation_20261003/review"
VIDEO = ROOT / "video/smoke_19.mp4"


def load_json(name: str):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


def read_jsonl(name: str):
    with (RUN / name).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def key3(row, table_key="table_id", session_key="session_id"):
    return (int(row[table_key]), row[session_key], str(row["hand_id"]))


def png_payload(record):
    return record.get("png_base64") or record.get("crop_png_base64")


def save_png(payload: str, rel: str, expected_hash: str | None = None):
    raw = base64.b64decode(payload)
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    actual = None
    if expected_hash:
        with Image.open(path) as im:
            gray = np.asarray(im.convert("L"), dtype=np.uint8)
        actual = hashlib.sha256(gray.tobytes()).hexdigest()
    return rel.replace("\\", "/"), actual == expected_hash if expected_hash else None, actual


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def gray_from_rgb(rgb):
    wide = rgb.astype(np.uint32)
    return ((wide[:, :, 0] * 9797 + wide[:, :, 1] * 19234 + wide[:, :, 2] * 3737) >> 15).astype(np.uint8)


def build_ncc_previews(triggers, ncc_rows, layout):
    wanted = {}
    for trig in triggers:
        target = (int(trig["table_id"]), trig["session_id"], int(trig["trigger_source_frame"]), trig["pixel_hash"])
        receipt = ncc_rows.get(target)
        trig["ncc_receipt"] = receipt
        trig["ncc_preview_status"] = "NCC_RECEIPT_NOT_FOUND"
        if receipt:
            wanted.setdefault(int(trig["trigger_source_frame"]), []).append((trig, receipt))
    if not wanted:
        return
    max_frame = max(wanted)
    slots = {slot.slot_id: slot for slot in layout.slots}
    container = av.open(str(VIDEO))
    stream = container.streams.video[0]
    index = -1
    for frame in container.decode(stream):
        index += 1
        if index not in wanted:
            if index >= max_frame:
                break
            continue
        rgb = frame.to_ndarray(format="rgb24")
        for trig, rec in wanted[index]:
            slot = slots[int(trig["table_id"])]
            x1, y1, x2, y2 = slot.chat_box
            gray = gray_from_rgb(rgb[y1:y2, x1:x2])
            a, b = map(int, rec["row_bounds"])
            crop = np.ascontiguousarray(gray[a:b])
            digest = hashlib.sha256(crop.tobytes()).hexdigest()
            filename = f"assets/deal_ncc/t{trig['table_id']:02d}_{trig['hand_id']}_f{index:05d}.png"
            path = OUT / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(crop, mode="L").save(path)
            pts = frame.pts
            pts_ok = pts is None or int(pts) == int(trig["trigger_source_pts"])
            hash_ok = digest == rec["pixel_hash"] == trig["pixel_hash"]
            trig.update(ncc_image=filename.replace("\\", "/"), ncc_hash_recomputed=digest,
                        ncc_hash_verified=hash_ok, ncc_pts_decoded=pts, ncc_pts_verified=pts_ok,
                        ncc_preview_status="EXTRACTED_HASH_MATCH" if hash_ok and pts_ok else "EXTRACTED_REVIEW_HASH_OR_PTS_MISMATCH")
        if index >= max_frame:
            break
    container.close()


def add_contact_sheet(cards, out_path, *, cols=3, cell_w=460, img_w=118, img_h=42):
    font = ImageFont.load_default()
    rows = (len(cards) + cols - 1) // cols
    cell_h = 78
    canvas = Image.new("RGB", (cols * cell_w, max(1, rows) * cell_h), "#f2f4f7")
    draw = ImageDraw.Draw(canvas)
    for i, card in enumerate(cards):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        draw.rectangle((x + 2, y + 2, x + cell_w - 4, y + cell_h - 4), outline="#c5ccd3", fill="white")
        draw.text((x + 8, y + 7), card["label"][:150], fill="#18212b", font=font)
        if card.get("images"):
            ix = x + 8
            for image_path in card["images"][:3]:
                try:
                    with Image.open(OUT / image_path) as im:
                        im = im.convert("L").resize((img_w, img_h), Image.Resampling.NEAREST)
                    canvas.paste(im.convert("RGB"), (ix, y + 27))
                    ix += img_w + 6
                except Exception:
                    pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    final = load_json("final_summary.json")
    live = load_json("start_stack_live_summary.json")
    dealing = load_json("dealing_start_stack_summary.json")
    forced = load_json("seat_forced_summary.json")
    math = load_json("start_stack_math_summary.json")

    live_hands = {key3(h): h for h in live["hands"]}
    math_results = math["results"]
    math_by_hand = collections.defaultdict(list)
    for row in math_results:
        math_by_hand[(int(row["table"]), row["session"], str(row["hand_id"]))].append(row)

    # Header candidate-to-confirmed mapping preserves table/session/frame/span.
    hand_state = list(read_jsonl("hand_state_receipts.jsonl"))
    header_by_key = {(int(x["table_id"]), x["session_id"], int(x["source_frame"]), int(x["source_span_index"])): x
                     for x in hand_state}
    header_crop_by_source = {}
    for row in read_jsonl("hand_header_ocr.jsonl"):
        if row.get("status") == "ADMITTED":
            header_crop_by_source[(int(row["table_id"]), row["session_id"], int(row["source_frame"]),
                                   int(row["source_span_index"]))] = row

    triggers = list(read_jsonl("dealing_cards_triggers.jsonl"))
    trigger_keys = {(int(t["table_id"]), t["session_id"], str(t["hand_id"])) for t in triggers}
    target_ncc = {(int(t["table_id"]), t["session_id"], int(t["trigger_source_frame"]), t["pixel_hash"])
                  for t in triggers}
    ncc_rows = {}
    for row in read_jsonl("dealing_ncc_receipts.jsonl"):
        key = (int(row["table_id"]), row["session_id"], int(row["source_frame"]), row["pixel_hash"])
        if key in target_ncc and row.get("status") == "cards":
            ncc_rows[key] = row

    # One header crop per confirmed trigger hand, with the original grayscale hash checked.
    header_images = []
    header_cards = []
    for trig in triggers:
        k = (int(trig["table_id"]), trig["session_id"], str(trig["hand_id"]))
        header_receipts = [x for x in hand_state if key3(x) == k]
        if not header_receipts:
            trig["header_preview_status"] = "HEADER_CONFIRMATION_NOT_FOUND"
            continue
        hs = header_receipts[0]
        source_key = (int(hs["table_id"]), hs["session_id"], int(hs["source_frame"]), int(hs["source_span_index"]))
        crop = header_crop_by_source.get(source_key)
        if not crop:
            trig["header_preview_status"] = "ADMITTED_HEADER_CROP_NOT_FOUND"
            continue
        rel = f"assets/headers/t{trig['table_id']:02d}_{trig['hand_id']}_f{crop['source_frame']:05d}.png"
        saved, verified, actual = save_png(crop["png_base64"], rel, crop.get("pixel_sha256"))
        trig.update(header_image=saved, header_hash_verified=verified, header_hash_recomputed=actual,
                    header_source_frame=crop["source_frame"], header_source_pts=crop["source_pts"],
                    header_span_index=crop["source_span_index"], header_pixel_sha256=crop["pixel_sha256"],
                    header_preview_status="EXTRACTED_HASH_MATCH" if verified else "EXTRACTED_HASH_MISMATCH")
        header_images.append(saved)
        header_cards.append({"label": f"T{trig['table_id']:02d} H{trig['hand_id']} header f{crop['source_frame']} pts{crop['source_pts']} sha {crop['pixel_sha256'][:12]}", "images": [saved]})

    # The saved NCC receipt carries a pixel hash but no crop. Recreate only those
    # exact row crops from the source video and verify each against the receipt hash.
    layout = None
    try:
        from parser_core.layout import load_fixed_layout
        layout = load_fixed_layout(ROOT / "configs/nine_table_layout.json", ROOT / "configs/layout_847x404.json",
                                   ROOT / "configs/chat_log_bottom_rois.json")
        build_ncc_previews(triggers, ncc_rows, layout)
    except Exception as exc:
        for trig in triggers:
            if trig.get("ncc_receipt"):
                trig["ncc_preview_status"] = f"NCC_SOURCE_CROP_EXTRACTION_FAILED:{type(exc).__name__}:{exc}"

    # Save numeric OCR crops for every accepted first triple; preserve the 3 source observations separately.
    numeric_cards = []
    numeric_rows = []
    crop_hash_checks = []
    for group_key, rows in sorted(math_by_hand.items()):
        for row in sorted(rows, key=lambda x: x["seat"]):
            evidence = row.get("evidence") or []
            imgs = []
            for ev in evidence:
                payload = ev.get("png_base64")
                if not payload:
                    continue
                rel = f"assets/numeric/t{group_key[0]:02d}_{group_key[2]}_{row['seat']}_f{int(ev['source_frame']):05d}.png"
                saved, verified, actual = save_png(payload, rel, ev.get("pixel_sha256"))
                imgs.append(saved)
                crop_hash_checks.append(verified)
            row_meta = {k: v for k, v in row.items() if k != "evidence"}
            row_meta["image_paths"] = imgs
            numeric_rows.append(row_meta)
            if imgs:
                numeric_cards.append({"label": f"T{group_key[0]:02d} H{group_key[2]} {row['seat']} {row.get('status')} stack={row.get('displayed_stack')} frames={row.get('frames')} OCR={row.get('ocr_values')}", "images": imgs})

    # Raw forced CHAT rows and exact nick ROI observations are already stored as RAM PNGs.
    forced_inputs = []
    forced_image_checks = []
    forced_cards = []
    for i, row in enumerate(read_jsonl("seat_forced_inputs.jsonl")):
        k = (int(row["table_id"]), row["session_id"], str(row["hand_id"]))
        if k not in trigger_keys:
            continue
        rel = f"assets/forced/t{row['table_id']:02d}_{row['hand_id']}_{row['kind'].lower()}_{row.get('seat','chat')}_f{int(row['source_frame']):05d}_{i:05d}.png"
        saved, verified, actual = save_png(row["png_base64"], rel, row.get("pixel_sha256"))
        forced_image_checks.append(verified)
        desc = dict(table=row["table_id"], session=row["session_id"], hand_id=row["hand_id"],
                    kind=row["kind"], seat=row.get("seat"), source_frame=row["source_frame"], source_pts=row["source_pts"],
                    time_base=row.get("time_base"), roi=row.get("roi"), row_bounds=row.get("row_bounds"),
                    actor_text=(row.get("ocr_response") or {}).get("raw_text"),
                    confidence=(row.get("ocr_response") or {}).get("score"), pixel_sha256=row.get("pixel_sha256"),
                    source_ref=row.get("source_ref"), image=saved, image_hash_verified=verified)
        forced_inputs.append(desc)
        label = f"T{row['table_id']:02d} H{row['hand_id']} {row['kind']} {row.get('seat','')} f{row['source_frame']} pts{row['source_pts']} OCR={(row.get('ocr_response') or {}).get('raw_text','')} sha {str(row.get('pixel_sha256',''))[:12]}"
        forced_cards.append({"label": label, "images": [saved]})

    deal_rows = list(read_jsonl("deal_complete_receipts.jsonl"))
    deal_status = collections.defaultdict(list)
    for row in deal_rows:
        deal_status[(int(row["table_id"]), row["session_id"], str(row["hand_id"]))].append(row)
    deals = {}
    for k, records in deal_status.items():
        deals[k] = next((r for r in reversed(records) if r.get("status") == "DEAL_COMPLETE"), records[-1])

    # Independent input audit: raw OCR sidecar is a separate text lane but not
    # an annotated hand-event ground truth, so its hits never produce TP/FP/FN claims.
    raw_text_rows = list(read_jsonl("raw_ocr_results.jsonl"))
    raw_pat = re.compile(r"dealer\s*:\s*.*\bdealing\s+cards\b", re.I)
    raw_dealing_hits = []
    raw_header_numbers = set()
    for row in raw_text_rows:
        text = (row.get("raw_ocr_response") or {}).get("raw_text", "")
        if raw_pat.search(text):
            raw_dealing_hits.append(dict(table=row["table_id"], frame=row["frame_id"], row_id=row.get("row_id"), text=text))
        for match in re.finditer(r"(?i)hand\s*#?\s*(\d{7,})", text):
            raw_header_numbers.add((row["table_id"], match.group(1)))

    # Counts are grounded in saved result schemas; PASS means math-candidate PASS,
    # not live START_STACK recovery, hand-history validation, or certification.
    pass_rows = [r for r in math_results if r.get("status") == "PASS"]
    pass_hands = [k for k, rows in math_by_hand.items() if any(r.get("status") == "PASS" for r in rows)]
    all_seat_pass_hands = [k for k, rows in math_by_hand.items() if rows and all(r.get("status") == "PASS" for r in rows)]
    unknown_rows = [r for r in math_results if r.get("status") != "PASS"]
    blocker_counts = collections.Counter(r.get("blocker") for r in math_results)
    historical_blockers = collections.Counter(r.get("blocker") for r in forced.get("historical_binding_blockers", []))
    event_counts = collections.Counter((r.get("kind"), r.get("state")) for r in forced.get("state_receipts", []))
    unresolved_counts = collections.Counter((r.get("kind"), r.get("blocker")) for r in forced.get("unresolved_state_events", []))

    live_counts = live.get("counts", {})
    hand_cards_html = []
    for trig in sorted(triggers, key=lambda x: (int(x["table_id"]), int(x["source_frame"]))):
        k = (int(trig["table_id"]), trig["session_id"], str(trig["hand_id"]))
        result_rows = sorted(math_by_hand.get(k, []), key=lambda x: x["seat"])
        deal = deals.get(k, {})
        if result_rows:
            pass_n = sum(r.get("status") == "PASS" for r in result_rows)
            unknown_n = len(result_rows) - pass_n
            label = f"{len(result_rows)} seat result(s): {pass_n} math PASS, {unknown_n} UNKNOWN"
        else:
            label = "No fact/math rows: marker-only candidate; hand-start/deal completion remains UNKNOWN"
        header_link = f'<a href="{html.escape(trig.get("header_image", ""))}">header crop</a>' if trig.get("header_image") else "header crop unavailable"
        ncc_link = f'<a href="{html.escape(trig.get("ncc_image", ""))}">NCC row crop</a>' if trig.get("ncc_image") else "NCC image unavailable"
        dealframe = deal.get("source_frame")
        hand_cards_html.append(f"""<section class=hand><h2>T{trig['table_id']:02d} · hand {html.escape(str(trig['hand_id']))}</h2>
<p><b>{html.escape(label)}</b> · trigger source frame {trig.get('trigger_source_frame')} PTS {trig.get('trigger_source_pts')} · NCC hash <code>{html.escape(str(trig.get('pixel_hash')))}</code></p>
<p>Header source frame {trig.get('header_source_frame')} PTS {trig.get('header_source_pts')} hash <code>{html.escape(str(trig.get('header_pixel_sha256')))}</code> ({html.escape(str(trig.get('header_preview_status')))}; verify={trig.get('header_hash_verified')}) · {header_link}</p>
<p>Internal DEAL_COMPLETE outcome: <b>{html.escape(str(deal.get('status','UNKNOWN')))}</b> frame {dealframe} blocker {html.escape(str(deal.get('blocker')))} · NCC image {ncc_link} · NCC crop verification {html.escape(str(trig.get('ncc_preview_status')))}</p>
<p>Marker is a candidate trigger. This run has no independent, externally annotated hand-trigger list; TP/FP/FN are NOT_MEASURED.</p>
{render_seats(result_rows, k)}
</section>""")

    pass_contact = OUT / "contact_numeric_first_triples_114.png"
    add_contact_sheet(numeric_cards, pass_contact, cols=3, cell_w=460, img_w=118, img_h=42)
    header_contact = OUT / "contact_header_candidates.png"
    add_contact_sheet(header_cards, header_contact, cols=4, cell_w=340, img_w=125, img_h=42)
    forced_contact = OUT / "contact_forced_nick_and_chat_inputs.png"
    # All saved forced/nickname crops are represented by links in the HTML; the contact
    # sheet is capped to keep it glanceable while preserving every image asset.
    add_contact_sheet(forced_cards[:240], forced_contact, cols=4, cell_w=340, img_w=105, img_h=38)

    summary = {
        "scope": "Offline visual review built from completed Smoke19 acceptance_ON evidence; no production files or run outputs modified.",
        "run": str(RUN.relative_to(ROOT)),
        "source_video": str(VIDEO.relative_to(ROOT)),
        "run_status": final.get("status"), "frames": final.get("frames"),
        "trigger_candidates": len(triggers), "unique_candidate_hands": len(trigger_keys),
        "header_confirmed_unique_hands": len({key3(x) for x in hand_state}),
        "internal_deal_complete_positive": sum(1 for r in deals.values() if r.get("status") == "DEAL_COMPLETE"),
        "internal_deal_complete_unknown": sum(1 for r in deals.values() if r.get("status") != "DEAL_COMPLETE"),
        "trigger_false_positive_false_negative": "NOT_MEASURED: no independent annotated source-event ground truth exists in the saved run.",
        "raw_ocr_dealing_cards_phrase_hits": len(raw_dealing_hits),
        "raw_ocr_hand_number_unique_pairs": len(raw_header_numbers),
        "dealing_ncc_positive_receipts": sum(1 for x in ncc_rows.values() if x.get("status") == "cards"),
        "math_result_rows": len(math_results), "math_pass_rows": len(pass_rows), "math_unknown_rows": len(unknown_rows),
        "distinct_hands_with_any_math_pass": len(pass_hands), "distinct_hands_with_all_listed_seats_math_pass": len(all_seat_pass_hands),
        "pass_sample_requirement": {"requested_distinct_hands": 10, "available_distinct_hands_with_any_math_pass": len(pass_hands), "shortfall": max(0, 10-len(pass_hands)), "sampled": [dict(table=k[0], hand_id=k[2]) for k in pass_hands]},
        "math_blocker_counts": dict(blocker_counts), "historical_binding_blocker_counts": dict(historical_blockers),
        "historical_binding_late_conflict_receipts": sum(bool(r.get("later_literal_conflicts")) for r in forced.get("historical_binding_receipts", [])),
        "occupancy_event_counts": {f"{k[0]}:{k[1]}":v for k,v in event_counts.items()},
        "unresolved_event_counts": {f"{k[0]}:{k[1]}":v for k,v in unresolved_counts.items()},
        "pending_replay_counts": live_counts,
        "start_stack_live_pass": live.get("start_stack_pass"), "math_pass_is_not_live_recovery": True,
        "numeric_triples_image_count": len(crop_hash_checks), "numeric_triple_png_hash_verified": sum(bool(x) for x in crop_hash_checks),
        "forced_chat_nick_input_crops": len(forced_inputs), "forced_crop_png_hash_verified": sum(bool(x) for x in forced_image_checks),
        "header_images": len(header_images), "header_png_hash_verified": sum(bool(t.get("header_hash_verified")) for t in triggers),
        "ncc_candidate_image_hash_verified": sum(bool(t.get("ncc_hash_verified")) for t in triggers),
        "ncc_candidate_images": sum(bool(t.get("ncc_image")) for t in triggers),
        "manual_visual_review_performed": False,
        "special_cases": [
            "T01 / 2878365444 / seat7 / numeric frames 26-28: FIRST_NUMERIC_SOURCE_COVERAGE_UNKNOWN",
            "T09 / 2878365800, T05 / 2878365932, T01 / 2878365962: FORCED_GRAMMAR_UNKNOWN",
            "T04 / 2878365485 / seat7 / frames 25-27 and T09 / 2878367252 / seat8 / frames 8132-34: FIRST_NUMERIC_TRIPLE_NOT_STABLE (verify exact matching in per-seat table)",
        ],
        "source_limits": [
            "`PREFLOP_DEALING_CARDS` counts are implementation trigger candidates, not ground truth.",
            "Hand-header OCR IDs cross-check hand identity, not whether every visual deal trigger was detected.",
            "DEAL_COMPLETE is the sidecar's internal positive receipt; it does not establish trigger recall.",
            "No human visual inspection is implied by gallery generation.",
            "NCC candidate crops are re-decoded from the original video and checked against saved pixel hashes; acceptance_ON has no runtime PNG files.",
        ],
    }
    write_json(OUT / "review_summary.json", summary)
    write_json(OUT / "trigger_and_crop_provenance.json", triggers)
    write_json(OUT / "numeric_result_rows.json", numeric_rows)
    write_json(OUT / "forced_nick_crop_index.json", forced_inputs)

    summary_table = """<table><tbody>""" + "".join(
        f"<tr><th>{html.escape(k.replace('_',' '))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in summary.items() if k not in {"special_cases", "source_limits", "pass_sample_requirement", "pending_replay_counts", "math_blocker_counts", "historical_binding_blocker_counts", "occupancy_event_counts", "unresolved_event_counts"}
    ) + "</tbody></table>"
    links = "".join(f'<li><a href="{name}">{name}</a></li>' for name in ["review_summary.json", "trigger_and_crop_provenance.json", "numeric_result_rows.json", "forced_nick_crop_index.json"])
    specials = "<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in summary["special_cases"]) + "</ul>"
    limits = "<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in summary["source_limits"]) + "</ul>"
    html_doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Smoke19 START_STACK evidence review</title>
<style>body{{font:14px/1.45 Segoe UI,Arial,sans-serif;margin:24px;color:#17212b;background:#f5f7f9}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;background:white}}td,th{{padding:5px 9px;border:1px solid #d7dde3;text-align:left;vertical-align:top}}code{{overflow-wrap:anywhere}}.warn{{background:#fff3cd;padding:12px;border-left:4px solid #d69b00}}.hand{{background:white;margin:18px 0;padding:16px;border:1px solid #ccd4dc;border-radius:6px}}.seatgrid{{border-collapse:collapse;width:100%}}.seatgrid img{{height:28px;image-rendering:pixelated;margin:2px}}.small{{font-size:12px;color:#394957}}.pass{{color:#156b39;font-weight:bold}}.unknown{{color:#875900;font-weight:bold}}</style></head><body>
<h1>Smoke19 START_STACK evidence review</h1><div class=warn><b>Gallery generated; manual visual review has not been performed.</b> The 71 marker rows are system candidates, not independently annotated ground truth. Trigger false positives and misses remain NOT_MEASURED. `start_stack_pass=0`; arithmetic PASS rows below do not mean live START_STACK recovery or hand-history certification.</div>
<h2>Run and coverage</h2>{summary_table}<h2>Special review cases</h2>{specials}<h2>Evidence limits</h2>{limits}
<h2>Contact sheets</h2><p><a href="contact_numeric_first_triples_114.png">Numeric first-triple contact sheet (all saved result seats)</a> · <a href="contact_header_candidates.png">Header candidates</a> · <a href="contact_forced_nick_and_chat_inputs.png">Forced/nickname crops (first 240 shown in contact sheet; all indexed below)</a></p>
<h2>Artifact indexes</h2><ul>{links}</ul><h2>All marker candidates and result groups</h2>{''.join(hand_cards_html)}
</body></html>"""
    (OUT / "index.html").write_text(html_doc, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("HTML", OUT / "index.html")
    print("CONTACT", pass_contact)


def render_seats(rows, hand_key):
    if not rows:
        return "<p>No math/fact seat rows were emitted for this candidate; this is UNKNOWN coverage.</p>"
    out = ["<table class=seatgrid><thead><tr><th>Seat</th><th>Math</th><th>Displayed stack</th><th>Forced source</th><th>Numeric crops / source audit</th></tr></thead><tbody>"]
    for row in rows:
        ev = row.get("evidence") or []
        images = []
        # Exact per-crop image links, frames, PTS, OCR, ROI and hash are in the JSON index.
        for item in ev:
            rel = f"assets/numeric/t{hand_key[0]:02d}_{hand_key[2]}_{row['seat']}_f{int(item['source_frame']):05d}.png"
            if (OUT / rel).exists():
                images.append(f'<a href="{html.escape(rel)}"><img src="{html.escape(rel)}" title="frame {item.get("source_frame")} PTS {item.get("source_pts")} OCR {html.escape(str(item.get("raw_text")))} SHA {html.escape(str(item.get("pixel_sha256")))}"></a>')
        st = row.get("status")
        out.append(f"<tr><td>{html.escape(str(row['seat']))}</td><td class={'pass' if st=='PASS' else 'unknown'}>{html.escape(str(st))}</td><td>{html.escape(str(row.get('displayed_stack')))}</td><td>{html.escape(str(row.get('forced_bets')))}</td><td>{''.join(images)}<br><span class=small>frames={html.escape(str(row.get('frames')))} · OCR={html.escape(str(row.get('ocr_values')))} · blocker={html.escape(str(row.get('blocker')))}</span></td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


if __name__ == "__main__":
    main()
