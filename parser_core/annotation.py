"""Independent visual evidence contracts and conservative event comparison.

No OCR is used to create reference labels. Coordinates refer to frozen PNGs;
reference row/message IDs are assigned by the visual reviewer.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

VERSION = "visual_audit_v1"
FIELDS = ("actor", "action", "amount", "role", "cards")
STATES = {"KNOWN", "NONE", "UNKNOWN", "AMBIGUOUS"}
SCENARIOS = ("normal", "wrapped_amount", "identical_consecutive", "initial_partial",
             "clipped_row", "large_scroll", "all_in", "return", "side_pot",
             "show_cards", "unreadable", "table_move", "table_replace",
             "occlusion", "scale_change", "hidden_chat")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(json_bytes(value))


def local_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Evidence outside corpus: {relative}")
    return path


def valid_bbox(box: object, width: int, height: int) -> bool:
    return (isinstance(box, list) and len(box) == 4 and
            all(type(x) is int for x in box) and
            0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def audit(root: Path, labels: dict) -> dict:
    """Reject malformed or changed evidence before producing quality metrics."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256(manifest_path)
    errors, warnings = [], []
    if (root / "manifest.sha256").read_text().strip() != digest:
        errors.append("FROZEN_MANIFEST_CHANGED")
    if labels.get("manifest_sha256") != digest or labels.get("schema_version") != VERSION:
        errors.append("ANNOTATION_MANIFEST_OR_SCHEMA_MISMATCH")
    evidence = {x["evidence_id"]: x for x in manifest["evidence"]}
    units = {x["unit_id"]: x for x in manifest["units"]}
    if len(evidence) != len(manifest["evidence"]) or len(units) != len(manifest["units"]):
        errors.append("DUPLICATE_FROZEN_ID")
    for item in evidence.values():
        try:
            path = local_path(root, item["local"])
            if sha256(path) != item["sha256"]:
                errors.append(f"PNG_HASH_MISMATCH:{item['evidence_id']}")
            with Image.open(path) as image:
                if list(image.size) != item["size"]:
                    errors.append(f"PNG_SIZE_MISMATCH:{item['evidence_id']}")
        except (OSError, ValueError) as exc:
            errors.append(f"EVIDENCE_UNAVAILABLE:{item['evidence_id']}:{exc}")
    annotated_units = labels.get("units", [])
    if Counter(x.get("unit_id") for x in annotated_units) != Counter(units.keys()):
        errors.append("ANNOTATION_UNIT_SET_MISMATCH")
    message_ids, row_ids = set(), set()
    counts = Counter()
    for unit in annotated_units:
        uid = unit.get("unit_id")
        if uid not in units:
            continue
        reviewed = unit.get("review_status") == "VISUALLY_REVIEWED"
        if unit.get("review_status") not in {"UNREVIEWED", "VISUALLY_REVIEWED"}:
            errors.append(f"INVALID_REVIEW_STATUS:{uid}")
        if reviewed:
            counts["reviewed_units"] += 1
            if (not unit.get("reviewer") or not unit.get("review_time") or
                    unit.get("coverage_complete") is not True):
                errors.append(f"REVIEW_ATTESTATION_MISSING:{uid}")
        else:
            counts["unreviewed_units"] += 1
        rows = {}
        for row in unit.get("rows", []):
            rid = row.get("row_id")
            if not rid or rid in row_ids:
                errors.append(f"DUPLICATE_OR_EMPTY_ROW_ID:{rid}")
            row_ids.add(rid)
            rows[rid] = row
            if not row.get("anchors"):
                errors.append(f"ROW_EVIDENCE_MISSING:{rid}")
            visual_text = row.get("visual_text", {})
            if reviewed and (visual_text.get("state") not in {"KNOWN", "UNKNOWN", "AMBIGUOUS"} or
                             (visual_text.get("state") == "KNOWN" and
                              (not isinstance(visual_text.get("value"), str) or
                               not visual_text["value"].strip())) or
                             (visual_text.get("state") != "KNOWN" and visual_text.get("value") is not None)):
                errors.append(f"INVALID_VISUAL_TEXT:{rid}")
            for anchor in row.get("anchors", []):
                ev = evidence.get(anchor.get("evidence_id"))
                if (ev is None or ev["evidence_id"] not in units[uid]["evidence_ids"] or
                        ev["roi_type"] != "chat" or
                        not valid_bbox(anchor.get("bbox"), *ev["size"])):
                    errors.append(f"INVALID_ROW_ANCHOR:{rid}")
        used = Counter()
        orders = []
        for message in unit.get("messages", []):
            mid = message.get("message_id")
            if not mid or mid in message_ids:
                errors.append(f"DUPLICATE_OR_EMPTY_MESSAGE_ID:{mid}")
            message_ids.add(mid)
            orders.append(message.get("order"))
            if type(message.get("order")) is not int or message.get("order", -1) < 0:
                errors.append(f"INVALID_MESSAGE_ORDER:{mid}")
            if message.get("completeness") not in {"COMPLETE", "PARTIAL", "UNKNOWN"}:
                errors.append(f"INVALID_COMPLETENESS:{mid}")
            fragments = message.get("row_ids", [])
            if not fragments or len(set(fragments)) != len(fragments) or any(x not in rows for x in fragments):
                errors.append(f"INVALID_MESSAGE_FRAGMENTS:{mid}")
            used.update(fragments)
            if not message.get("scenarios") or any(s not in SCENARIOS for s in message.get("scenarios", [])):
                errors.append(f"INVALID_SCENARIO:{mid}")
            for field in FIELDS:
                value = message.get("fields", {}).get(field, {})
                state = value.get("state")
                if state not in STATES or "value" not in value:
                    errors.append(f"INVALID_FIELD:{mid}:{field}")
                elif state == "KNOWN" and value["value"] is None:
                    errors.append(f"KNOWN_NULL:{mid}:{field}")
                elif state != "KNOWN" and value["value"] is not None:
                    errors.append(f"NON_KNOWN_VALUE:{mid}:{field}")
                elif state == "KNOWN" and field == "cards" and not isinstance(value["value"], list):
                    errors.append(f"INVALID_CARDS:{mid}")
                elif state == "KNOWN" and field != "cards" and not isinstance(value["value"], str):
                    errors.append(f"INVALID_TEXT_FIELD:{mid}:{field}")
            if reviewed:
                counts["reviewed_messages"] += 1
                counts["complete_messages"] += message.get("completeness") == "COMPLETE"
        if len(set(orders)) != len(orders):
            errors.append(f"DUPLICATE_MESSAGE_ORDER:{uid}")
        if reviewed and (set(used) != set(rows) or any(n != 1 for n in used.values())):
            errors.append(f"ROW_MESSAGE_OWNERSHIP_INVALID:{uid}")
        if not reviewed and (rows or unit.get("messages")):
            warnings.append(f"UNREVIEWED_LABELS_EXCLUDED:{uid}")
    if counts["reviewed_units"] == 0:
        warnings.append("NO_VISUAL_REFERENCE_QUALITY_NOT_MEASURED")
    return {"version": VERSION, "manifest_sha256": digest, "errors": errors,
            "warnings": warnings, "counts": dict(counts), "valid": not errors,
            "independent_test": manifest["independent_test"],
            "independence_basis": manifest["independence_basis"]}


def covered_by(gold: list[int], prediction: list[int]) -> bool:
    # Reviewer boxes may be tight ink boxes; parser crops include vertical padding.
    intersection = max(0, min(gold[2], prediction[2]) - max(gold[0], prediction[0])) * max(
        0, min(gold[3], prediction[3]) - max(gold[1], prediction[1]))
    return intersection / ((gold[2] - gold[0]) * (gold[3] - gold[1])) >= 0.8


def compare(labels: dict, predictions: list[dict], accepted_statuses=("VERIFIED",)) -> dict:
    """Match only visual row anchors, never actor/action/text or ordinal positions.

    Many-to-one and one-to-many links remain errors, with no greedy correction.
    Candidate correctness and published/accepted correctness are distinct metrics.
    """
    by_unit = defaultdict(list)
    for prediction in predictions:
        by_unit[prediction["unit_id"]].append(prediction)
    totals, scenarios, details = Counter(), defaultdict(Counter), []
    unresolved_statuses = {"UNRESOLVED", "UNKNOWN", "PARTIAL", "REJECTED"}
    for unit in labels["units"]:
        if unit["review_status"] != "VISUALLY_REVIEWED":
            continue
        gold = unit["messages"]
        predicted = by_unit[unit["unit_id"]]
        rows = {x["row_id"]: x for x in unit["rows"]}
        anchors_by_evidence = defaultdict(list)
        for rid, row in rows.items():
            for anchor in row["anchors"]:
                anchors_by_evidence[anchor["evidence_id"]].append((rid, anchor))
        row_owner = {rid: index for index, message in enumerate(gold) for rid in message["row_ids"]}
        links = defaultdict(list)
        mapped = []
        for pi, prediction in enumerate(predicted):
            mapped_rows, ambiguous = [], False
            for fragment in prediction["fragments"]:
                found = {rid for b in fragment["anchors"]
                         for rid, a in anchors_by_evidence[b["evidence_id"]]
                         if covered_by(a["bbox"], b["bbox"])}
                ambiguous |= len(found) != 1
                mapped_rows.append(next(iter(found)) if len(found) == 1 else None)
                for rid in found:
                    if pi not in links[row_owner[rid]]:
                        links[row_owner[rid]].append(pi)
            mapped.append((mapped_rows, ambiguous))
        totals["predicted"] += len(predicted)
        totals["accepted_predictions"] += sum(p["status"] in accepted_statuses for p in predicted)
        totals["unresolved_predictions"] += sum(p["status"] in unresolved_statuses for p in predicted)
        linked_predictions = {pi for values in links.values() for pi in values}
        extras = [pi for pi in range(len(predicted)) if pi not in linked_predictions]
        totals["extra_predictions"] += len(extras)
        totals["false_accept"] += sum(predicted[pi]["status"] in accepted_statuses for pi in extras)
        exact_pairs = []
        for gi, message in enumerate(gold):
            counts = Counter(reference=1)
            indices = links[gi]
            boundary = False
            if not indices:
                counts["missing"] += 1
            elif len(indices) > 1:
                counts["multiple_predictions_for_message"] += 1
                counts["excess_predictions_for_message"] += len(indices) - 1
                # Splits and duplicates cannot be distinguished safely by text.
            else:
                pi = indices[0]
                mapped_rows, ambiguous = mapped[pi]
                boundary = not ambiguous and mapped_rows == message["row_ids"]
                if not boundary:
                    counts["boundary_error"] += 1
                else:
                    counts["exact_message_boundary"] += 1
                    exact_pairs.append((message["order"], predicted[pi]["order_key"], gi))
            evaluable = (message["completeness"] == "COMPLETE" and all(
                message["fields"][f]["state"] in {"KNOWN", "NONE"} for f in FIELDS))
            counts["exact_event_denominator"] += evaluable
            correct = False
            if len(indices) == 1:
                prediction = predicted[indices[0]]
                counts["unresolved"] += prediction["status"] in unresolved_statuses
                field_correct = []
                for field in FIELDS:
                    reference = message["fields"][field]
                    known = reference["state"] in {"KNOWN", "NONE"}
                    match = boundary and known and reference["value"] == prediction["fields"][field]
                    field_correct.append(match)
                correct = evaluable and all(field_correct)
                counts["exact_event"] += correct
                counts["accepted_exact_event"] += correct and prediction["status"] in accepted_statuses
            for field in FIELDS:
                reference = message["fields"][field]
                known = reference["state"] in {"KNOWN", "NONE"}
                counts[f"exact_{field}_denominator"] += known
                counts[f"exact_{field}"] += (len(indices) == 1 and boundary and known and
                    reference["value"] == predicted[indices[0]]["fields"][field])
            for pi in indices:
                if predicted[pi]["status"] in accepted_statuses and not (len(indices) == 1 and correct):
                    counts["false_accept"] += 1
            totals.update(counts)
            for scenario in message["scenarios"]:
                scenarios[scenario].update(counts)
            details.append({"unit_id": unit["unit_id"], "message_id": message["message_id"],
                            "predicted_ids": [predicted[i]["event_id"] for i in indices],
                            "counts": dict(counts)})
        # Count inversions of observation order per table window, not completion order.
        exact_pairs.sort()
        for i, (_, key, gi) in enumerate(exact_pairs):
            for _, other_key, gj in exact_pairs[i + 1:]:
                totals["order_pairs"] += 1
                wrong = key >= other_key
                totals["order_errors"] += wrong
                for scenario in set(gold[gi]["scenarios"] + gold[gj]["scenarios"]):
                    scenarios[scenario]["order_pairs"] += 1
                    scenarios[scenario]["order_errors"] += wrong
        # Scenario denominators count distinct predictions linked to that scenario.
        for scenario in {s for m in gold for s in m["scenarios"]}:
            linked = {pi for gi, m in enumerate(gold) if scenario in m["scenarios"] for pi in links[gi]}
            exact_linked = {d["predicted_ids"][0] for d in details if d["unit_id"] == unit["unit_id"]
                            and d["counts"].get("exact_event")}
            sc = scenarios[scenario]
            sc["predicted"] += len(linked)
            sc["accepted_predictions"] += sum(predicted[pi]["status"] in accepted_statuses for pi in linked)
            sc["unresolved_predictions"] += sum(predicted[pi]["status"] in unresolved_statuses for pi in linked)
            # Replace message-based counts below after all units, to avoid merged double counting.
            sc["unique_false_accept"] += sum(predicted[pi]["status"] in accepted_statuses and
                                              predicted[pi]["event_id"] not in exact_linked for pi in linked)
    # A merged accepted prediction may be linked to several gold messages: count once.
    false_ids = set()
    for unit in labels["units"]:
        if unit["review_status"] != "VISUALLY_REVIEWED":
            continue
        exact_ids = {d["predicted_ids"][0] for d in details if d["unit_id"] == unit["unit_id"]
                     and d["counts"].get("exact_event")}
        false_ids.update((unit["unit_id"], p["event_id"]) for p in by_unit[unit["unit_id"]]
                         if p["status"] in accepted_statuses and p["event_id"] not in exact_ids)
    totals["false_accept"] = len(false_ids)
    for sc in scenarios.values():
        sc["false_accept"] = sc["unique_false_accept"]
    def metrics(counts):
        pairs = {"exact_event": "exact_event_denominator", "accepted_exact_event": "exact_event_denominator",
                 "unresolved": "reference", "unresolved_predictions": "predicted",
                 "missing": "reference", "exact_message_boundary": "reference",
                 "false_accept": "accepted_predictions", "order_errors": "order_pairs"}
        pairs.update({f"exact_{f}": f"exact_{f}_denominator" for f in FIELDS})
        return {name: {"numerator": counts[name], "denominator": counts[denom],
                       "rate": counts[name] / counts[denom] if counts[denom] else None}
                for name, denom in pairs.items()}
    return {"version": VERSION, "counts": dict(totals), "metrics": metrics(totals),
            "scenarios": {s: {"status": "MEASURED" if s in scenarios else "NOT_TESTED",
                                "counts": dict(scenarios[s]), "metrics": metrics(scenarios[s])}
                          for s in SCENARIOS}, "details": details,
            "quality_status": "MEASURED" if totals["reference"] else "NOT_MEASURED",
            "accepted_statuses": list(accepted_statuses),
            "limitations": ["Geometry matching requires visually annotated row anchors.",
                            "Multiple predictions may be splits or duplicates; inspect details.",
                            "Transitions, entity boxes, complete hands and HH are not scored by this event evaluator."]}
