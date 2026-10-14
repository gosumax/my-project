"""Literal CHAT grammar and lossless source records, independent of production."""
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from .models import Event
from parser_core.poker_hand_assembly import split_small_blind, observed_wrapped_continuation
from parser_core.canonical_semantics import bound_forced_all_in
from parser_core.service_text_normalizer import normalize_service, physical_service_group, ServiceText

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
NUMBER = r"\d+(?:[ \u00a0]\d{3})*"
VERBS = [(r"posts (?:the )?ante", "ANTE"), (r"posts (?:the )?small blind", "SB"),
         (r"posts (?:the )?big blind", "BB"), (r"raises(?: all-in)? to", "RAISE"),
         (r"raises", "RAISE"), (r"calls to", "CALL"), (r"calls", "CALL"),
         (r"checks", "CHECK"), (r"folds", "FOLD"), (r"bets", "BET"),
         (r"wins", "WIN"), (r"gets uncalled bet", "RETURN"),
         (r"does not show|doesn't show", "INFO"), (r"shows", "SHOW")]


def verified_board_payload(previous, row):
    pixel = row.get("row_pixel_sha256")
    return (isinstance(previous.get("row_index"), int) and isinstance(row.get("row_index"), int)
            and previous.get("frame_id") is not None and previous.get("frame_id") == row.get("frame_id")
            and row["row_index"] == previous["row_index"] + 1
            and previous.get("identity_status") == "CHAT_ROW_SHA_ROUNDTRIP_VERIFIED"
            and isinstance(pixel, str) and re.fullmatch(r"[0-9a-f]{64}", pixel) is not None
            and previous.get("card_crop_pixel_sha256") == pixel
            and re.fullmatch(r"[\s\[\],0-9TJQKAshdc♠♥♦♣+]+", raw_text(row), re.I) is not None)


def read_timeline(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [dict(json.loads(line), source_index=i, source_file=str(path.resolve()))
                for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1) if line.strip()]
    if path.suffix.lower() != ".xlsx":
        raise ValueError("Expected .xlsx or .jsonl")
    with zipfile.ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(si.itertext()) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        book = ET.fromstring(z.read("xl/workbook.xml"))
        rels = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        sheets = book.find("m:sheets", NS)
        sheet = next((s for s in sheets if s.attrib["name"] == "ALL_EVENTS"), None)
        if sheet is None:
            raise ValueError("Workbook lacks ALL_EVENTS; schema must be explicitly inspected")
        rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = rels[rid].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        matrix = []
        for row in ET.fromstring(z.read(target)).findall("m:sheetData/m:row", NS):
            values = {}
            for cell in row:
                col = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                v = cell.find("m:v", NS)
                text = v.text if v is not None else None
                if cell.attrib.get("t") == "s":
                    text = strings[int(text)]
                elif cell.attrib.get("t") == "inlineStr":
                    text = "".join(cell.find("m:is", NS).itertext())
                elif text is not None and cell.attrib.get("t") not in {"str", "e"}:
                    try:
                        text = int(text)
                    except ValueError:
                        pass  # Preserve noninteger values; never round chip amounts.
                values[col] = text
            matrix.append((int(row.attrib["r"]), values))
        headers = matrix[0][1]
        result = []
        for index, values in matrix[1:]:
            record = {str(name): values.get(col) for col, name in headers.items() if name is not None}
            record.update(source_index=index, source_sheet="ALL_EVENTS", source_file=str(path.resolve()))
            result.append(record)
        return result


def raw_text(row):
    return str(row.get("raw_ocr_text") or row.get("raw_text") or row.get("text") or "")


def parse_text(text, source):
    if any(r.get('canonical_conflict') or r.get('canonical_duplicate_of') for r in source.get('rows', [])):
        return Event('UNKNOWN',source=dict(source,canonical_withheld=True))
    normalized = (normalize_service(text) if physical_service_group(source.get('rows', []))
                  else ServiceText(text, text, ()))
    if normalized.changes:
        source = dict(source, service_normalizations=list(normalized.changes),
                      normalized_text=normalized.normalized_text)
    clean = re.sub(r"^\s*\[\d+:\d+\]\s*", "", normalized.normalized_text)
    clean = re.sub(r"^(?:Dealer|Дилер)\s*[:;]\s*", "", clean, flags=re.I).strip()
    from parser_core.table_service_events import describe_service
    service = describe_service(clean)
    if service and service['kind'] in {'REBUY', 'BOUNTY', 'RESEATING', 'BREAK'}:
        return Event('INFO', source=dict(source, service_event=service))
    header = re.search(r"(?:Hand\s*#|^#)\s*(\d+)", clean, re.I)
    if header:
        return Event("START_HAND", source=dict(source, parsed_hand_id=header.group(1)))
    if re.search(r"Starting a new hand", clean, re.I) and re.search(r"#(\d+)", clean):
        return Event("START_HAND", source=dict(source, parsed_hand_id=re.search(r"#(\d+)", clean).group(1)))
    if re.fullmatch(r"Starting a new hand[.!●]?", clean, re.I):
        return Event("INFO", source=source)
    if re.fullmatch(r"Dealing cards[.!●]?", clean, re.I):
        return Event("DEAL", source=source)
    street = re.match(r"^(?:Dealing (?:the )?)?(Flop|Turn|River)\b", clean, re.I)
    if street:
        return Event("STREET", street=street.group(1).upper(), source=source)
    if re.fullmatch(r"Showdown[.!]?", clean, re.I):
        return Event("STREET", street="SHOWDOWN", source=source)
    for pattern, kind in VERBS:
        boundary=r'(?![^\W\d])' if kind=='RETURN' else r'(?!\w)'
        match = re.search(r"(?<![^\W_])(?:" + pattern + r")"+boundary, clean, re.I)
        if not match:
            continue
        player = clean[:match.start()].strip(" :")
        tail = clean[match.end():].strip()
        prefix_all_in = bool(re.match(r"all[ -]?in\b", tail, re.I)) or "all-in" in match.group().lower()
        if prefix_all_in:
            tail = re.sub(r"^all[ -]?in\s*", "", tail, flags=re.I)
        number = re.match(NUMBER, tail) if kind not in {"FOLD", "CHECK", "SHOW", "INFO"} else None
        amount = None
        if number:
            suffix = tail[number.end():].strip()
            if not suffix or re.fullmatch(r"(?:back|and is All[ -]?in|is All[ -]?in|All[ -]?in|and is|and)?[.!●]?", suffix, re.I) or (kind == "WIN" and re.match(r"from\b", suffix, re.I)):
                amount = int(number.group().replace(" ", "").replace("\u00a0", ""))
        semantics = "TO" if "to" in match.group().lower().split() else "DELTA"
        if kind == "RAISE" and semantics != "TO":
            semantics = "UNKNOWN"
        claimed,all_in_proof=bound_forced_all_in(source.get('rows',[]),kind,player or None,amount)
        if claimed and not all_in_proof:
            return Event('UNKNOWN',source=dict(source,canonical_withheld='CANONICAL_ALL_IN_PROOF_CONFLICT'))
        return Event(kind, player or None, amount, semantics,
                     bool(all_in_proof) or prefix_all_in or bool(re.search(r"all[ -]?in", tail, re.I)),
                     source=dict(source,all_in_provenance=all_in_proof) if all_in_proof else source)
    if re.fullmatch(r"[\s\[\],]*(?:(?:[2-9TJQKA]|10)[shdc♠♥♦♣][\s\[\],]*)+", clean, re.I):
        return Event("INFO", source=source)  # Card rows are retained; payouts are observed.
    if re.fullmatch(r"Limits going up: stakes.*|Waiting for .+ to(?: (?:reconnect|rebuy)\.{0,3})?|(?:reconnect|rebuy)\.{0,3}|.+[ _](?:is reconnected|uses Time Bank|sits out|sits in|finished the tournament|has joined the table|has left the table)[.!]?", clean, re.I):
        return Event("INFO", source=source)
    return Event("UNKNOWN", source=source)


def parse_rows(rows):
    events = []
    for row in rows:
        source = {"rows": [row], "source_index": row.get("source_index"),
                  "frame": row.get("frame_id", row.get("frame")),
                  "row_id": row.get("stable_row_id", row.get("row_id")), "raw_text": raw_text(row)}
        event = parse_text(source["raw_text"], source)
        # Only observed neighboring physical rows can form one message. No
        # matching across frames/session/epoch and no arbitrary OCR repair.
        if events and not re.match(r"^\s*(?:\[\d+:\d+\]|Dealer:|Дилер:)", raw_text(row), re.I):
            prior = events[-1]
            previous = prior.source["rows"][-1]
            continuation_text = raw_text(row)
            if physical_service_group(prior.source['rows'] + [row]):
                continuation_text = normalize_service(continuation_text,
                    previous_clause=prior.source["raw_text"]).normalized_text
            adjacent = (row.get("frame_id") is not None and row.get("frame_id") == previous.get("frame_id")
                        and row.get("epoch") == previous.get("epoch")
                        and isinstance(row.get("row_index"), int) and isinstance(previous.get("row_index"), int)
                        and row["row_index"] == previous["row_index"] + 1)
            continuation = ((prior.kind in {"SB", "BB", "ANTE", "CALL", "BET", "RAISE", "RETURN", "WIN"}
                             and (prior.amount is None or re.fullmatch(r"(?:and is All[ -]?in|is All[ -]?in|All[ -]?in|back)[.!●]?", continuation_text.strip(), re.I))
                             and re.fullmatch(NUMBER + r"(?:\s+(?:back|and is All[ -]?in|is All[ -]?in))?[.!●]?|(?:and is All[ -]?in|is All[ -]?in|All[ -]?in|back)[.!●]?", continuation_text.strip(), re.I))
                            or (re.search(r"Starting a new hand", prior.source["raw_text"], re.I) and re.fullmatch(r"\s*#\d+\s*", raw_text(row))))
            continuation = continuation or (prior.kind == "WIN" and re.fullmatch(r"(?:from )?(?:the )?(?:main pot|side pot(?: #\d+)?)(?: with .*)?", raw_text(row).strip(), re.I))
            continuation = continuation or (len(prior.source['rows']) == 1 and split_small_blind(previous, row))
            continuation = continuation or observed_wrapped_continuation(prior.source['rows'],row)
            if adjacent and continuation:
                merged = dict(prior.source, rows=prior.source["rows"] + [row],
                              raw_text=prior.source["raw_text"] + " " + raw_text(row))
                combined = parse_text(merged["raw_text"], merged)
                if re.search(r"Starting a new hand", merged["raw_text"], re.I) and re.search(r"#\d+", merged["raw_text"]):
                    combined = Event("START_HAND", source=dict(merged, parsed_hand_id=re.search(r"#(\d+)", merged["raw_text"]).group(1)))
                events[-1] = combined
                continue
            if adjacent and prior.kind == "STREET" and event.kind == "UNKNOWN" and verified_board_payload(previous, row):
                event = Event("INFO", source=dict(source, interpretation="OPAQUE_ADJACENT_BOARD_PAYLOAD; cards not evaluated"))
        events.append(event)
    return events
