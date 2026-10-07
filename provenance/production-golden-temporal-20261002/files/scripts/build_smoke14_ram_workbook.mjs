import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [jsonlPath, outputPath, evidencePath] = process.argv.slice(2);
if (!jsonlPath || !outputPath) throw new Error("Usage: build_smoke14_ram_workbook.mjs INPUT.jsonl OUTPUT.xlsx");
const records = (await fs.readFile(jsonlPath, "utf8")).split(/\r?\n/u)
  .filter(Boolean).map((line) => JSON.parse(line));
const columns = ["table_id", "sequence_id", "frame_id", "video_time",
  "source_roi_change_id", "row_index", "raw_ocr_text", "normalized_text",
  "event_type", "confidence", "status", "notes_error", "session_id",
  "origin", "epoch", "row_pixel_sha256", "hand_id", "street", "cards",
  "raw_vl_response", "evidence_status", "continuity_status",
  "visual_crop_path", "visual_crop_sha256", "trigger_row_key", "ocr_event_type",
  "board_evidence_id", "board_sha", "visual_crop_type", "identity_status",
  "card_evidence_id", "card_row_key", "card_row_id", "card_crop_pixel_sha256",
  "evidence_context_status", "player"];
const columnName = (n) => { let s = ""; for (++n; n; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(65 + (n - 1) % 26) + s; return s; };
const eventLast = columnName(columns.length-1);
const workbook = Workbook.create();
const sheets = ["ALL_EVENTS", ...Array.from({ length: 9 }, (_, i) => `T${String(i + 1).padStart(2, "0")}`)];
for (const name of sheets) {
  const sheet = workbook.worksheets.add(name);
  const rows = name === "ALL_EVENTS" ? records : records.filter((item) => item.table_id === name);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.getRange(`A1:${eventLast}1`).values = [columns];
  sheet.getRange(`A1:${eventLast}1`).format = { fill: "#26384D", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
  sheet.getRange(`A:${eventLast}`).format.font = { name: "Arial", size: 10 };
  for (let start = 0; start < rows.length; start += 400) {
    const block = rows.slice(start, start + 400).map((item) => columns.map((key) =>
      key === "cards" ? (item.cards?.join(" ") ?? null) : (item[key] ?? null)));
    sheet.getRangeByIndexes(start + 1, 0, block.length, columns.length).values = block;
  }
  for (const [column, width] of Object.entries({ A: 75, B: 95, C: 85, D: 105, E: 265,
      F: 75, G: 380, H: 380, I: 150, J: 100, K: 120, L: 280, M: 260,
      N: 150, O: 75, P: 370, Q: 150, R: 100, S: 150, T: 300,
      U: 170, V: 190, W: 390, X: 370, Y: 150, Z: 150, AA: 300, AB: 370 })) {
    sheet.getRange(`${column}:${column}`).format.columnWidthPx = width;
  }
  if (rows.length) {
    sheet.getRange(`D2:D${rows.length + 1}`).format.numberFormat = "0.000";
    sheet.getRange(`J2:J${rows.length + 1}`).format.numberFormat = "0.000";
  }
}
if (evidencePath) {
  const evidence = (await fs.readFile(evidencePath, "utf8")).split(/\r?\n/u)
    .filter(Boolean).map((line) => JSON.parse(line));
  const columns = ["table_id", "table_session_id", "frame_id", "source_span_index",
    "row_key", "trigger_row_key", "hand_id", "type", "street", "player",
    "crop_path", "crop_sha256", "row_pixel_sha256", "vl_status", "vl_raw_text",
    "card_reader", "status", "reason", "card_index", "tile_png", "tile_sha256",
    "tile_pixel_sha256", "vl_rank", "primary_suit", "primary_confidence",
    "raw_suit2_suit", "template_suit", "template_best_score",
    "template_second_score", "template_margin", "template_confident",
    "suit_source", "suit_status", "suit", "card", "nearest_train_card",
    "best_distance", "runner_up_other_card_distance", "row_id", "epoch", "image_transport",
    "box_xyxy_exclusive", "detected_bright_box_xyxy_exclusive", "tile_geometry_source", "native_card_pitch",
    "template_manifest_sha256", "template_selection_sha256", "template_config_sha256",
    "suit_backup_config_sha256", "suit_backup_mode", "rank_alignment_status",
    "vl_rank_count", "card_tile_count", "hybrid_card_count", "published_card_count",
    "evidence_jsonl", "evidence_record_number", "identity_status", "response_source_sha256",
    "card_evidence_id", "context_reason", "ncc_geometry_source", "suit_glyph_box_xywh",
    "ncc_query_box_xyxy_exclusive", "ncc_suit_roi_box_xyxy_exclusive", "ncc_query_sha256",
    "ncc_query_pixel_sha256"];
  const audit = [];
  for (const [recordIndex, item] of evidence.entries()) {
    const tiles = item.hybrid?.tiles;
    const shared = { ...item, ...item.hybrid, evidence_jsonl: evidencePath,
      evidence_record_number: recordIndex + 1,
      vl_rank_count: item.hybrid?.vl_rank_tokens?.length ?? null,
      card_tile_count: tiles?.length ?? 0, hybrid_card_count: item.hybrid?.cards?.length ?? 0,
      published_card_count: item.projected_card_count ?? 0 };
    if (tiles?.length) {
      for (const tile of tiles) audit.push({ ...shared, ...tile, reason: item.hybrid?.reason ?? item.reason });
    } else if (item.crop_type === "CHAT_ROW_CARDS") {
      audit.push(shared);
    }
  }
  const sheet = workbook.worksheets.add("CARD_AUDIT");
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const columnName = (n) => { let s = ""; for (++n; n; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(65 + (n - 1) % 26) + s; return s; };
  const last = columnName(columns.length - 1);
  sheet.getRange(`A1:${last}1`).values = [columns];
  sheet.getRange(`A1:${last}1`).format = { fill: "#26384D", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" } };
  sheet.getRange(`A:${last}`).format.font = { name: "Arial", size: 10 };
  for (let start = 0; start < audit.length; start += 400) {
    const block = audit.slice(start, start + 400).map((item) => columns.map((key) =>
      Array.isArray(item[key]) ? JSON.stringify(item[key]) : (item[key] ?? null)));
    sheet.getRangeByIndexes(start + 1, 0, block.length, columns.length).values = block;
  }
  sheet.getRange(`A:${last}`).format.columnWidthPx = 130;
  for (const col of ["K", "T"]) sheet.getRange(`${col}:${col}`).format.columnWidthPx = 360;
  sheet.getRange("O:O").format.columnWidthPx = 280;
  sheets.push("CARD_AUDIT");
}
workbook.recalculate();
await workbook.inspect({ kind: "region", sheetId: "T01", range: "A1:L6", maxChars: 2000 });
if (workbook.worksheets.getItem("T01").getRange("G1").values[0][0] !== "raw_ocr_text") {
  throw new Error("T01 header inspection failed");
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ output: outputPath, sheets, rows: records.length }));
