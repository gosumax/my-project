import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [jsonlPath, outputPath] = process.argv.slice(2);
if (!jsonlPath || !outputPath) throw new Error("Usage: build_smoke14_ram_workbook.mjs INPUT.jsonl OUTPUT.xlsx");
const records = (await fs.readFile(jsonlPath, "utf8")).split(/\r?\n/u)
  .filter(Boolean).map((line) => JSON.parse(line));
const columns = ["table_id", "sequence_id", "frame_id", "video_time",
  "source_roi_change_id", "row_index", "raw_ocr_text", "normalized_text",
  "event_type", "confidence", "status", "notes_error", "session_id",
  "origin", "epoch", "row_pixel_sha256"];
const workbook = Workbook.create();
const sheets = ["ALL_EVENTS", ...Array.from({ length: 9 }, (_, i) => `T${String(i + 1).padStart(2, "0")}`)];
for (const name of sheets) {
  const sheet = workbook.worksheets.add(name);
  const rows = name === "ALL_EVENTS" ? records : records.filter((item) => item.table_id === name);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.getRange("A1:P1").values = [columns];
  sheet.getRange("A1:P1").format = { fill: "#26384D", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
  sheet.getRange("A:P").format.font = { name: "Arial", size: 10 };
  for (let start = 0; start < rows.length; start += 400) {
    const block = rows.slice(start, start + 400).map((item) => columns.map((key) => item[key] ?? null));
    sheet.getRangeByIndexes(start + 1, 0, block.length, columns.length).values = block;
  }
  for (const [column, width] of Object.entries({ A: 75, B: 95, C: 85, D: 105, E: 265,
      F: 75, G: 380, H: 380, I: 150, J: 100, K: 120, L: 280, M: 260,
      N: 150, O: 75, P: 370 })) {
    sheet.getRange(`${column}:${column}`).format.columnWidthPx = width;
  }
  if (rows.length) {
    sheet.getRange(`D2:D${rows.length + 1}`).format.numberFormat = "0.000";
    sheet.getRange(`J2:J${rows.length + 1}`).format.numberFormat = "0.000";
  }
}
workbook.recalculate();
await workbook.inspect({ kind: "region", sheetId: "T01", range: "A1:L6", maxChars: 2000 });
if (workbook.worksheets.getItem("T01").getRange("G1").values[0][0] !== "raw_ocr_text") {
  throw new Error("T01 header inspection failed");
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ output: outputPath, sheets, rows: records.length }));
