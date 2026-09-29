import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const dataDir = process.argv[2];
const outputPath = process.argv[3];
if (!dataDir || !outputPath) throw new Error("Usage: build_smoke14_workbook.mjs DATA_DIR OUTPUT.xlsx");

async function readJson(file) {
  return JSON.parse(await fs.readFile(file, "utf8"));
}

async function readJsonl(file) {
  const text = await fs.readFile(file, "utf8");
  return text.split(/\r?\n/u).filter(Boolean).map((line) => JSON.parse(line));
}

function findStage(report, name) {
  return report.stages?.find((item) => item.stage === name) ?? {};
}

function metric(performance, key, fallback = null) {
  return performance && Number.isFinite(Number(performance[key])) ? Number(performance[key]) : fallback;
}

const data = await readJson(path.join(dataDir, "workbook_data.json"));
const recognitions = await readJsonl(path.join(dataDir, "recognitions.jsonl"));
const newReport = data.new_report;
const oldReport = data.old_report;
const newOcr = data.new_ocr;
const newCapture = findStage(newReport, "capture");
const newTrack = findStage(newReport, "track_rows");
const newOcrStage = findStage(newReport, "ocr_rows");
const oldCapture = findStage(oldReport, "capture");
const oldOcrStage = findStage(oldReport, "ocr_rows");
const newPerf = newOcr.performance ?? {};
const oldPerf = oldOcrStage.details?.performance ?? {};
const newFrames = Number(newReport.counts?.frames ?? 0);
const oldFrames = Number(oldReport.counts?.frames ?? 0);
const newWall = Number(newReport.wall_seconds ?? 0);
const oldWall = Number(oldReport.wall_seconds ?? 0);
const newIps = Number(newOcr.images_per_inference_second ?? 0);
const oldVlSecondsPerJob = metric(oldPerf, "vl_jobs") ? metric(oldPerf, "vl_wall_seconds") / metric(oldPerf, "vl_jobs") : null;
const oldTessSecondsPerJob = metric(oldPerf, "tesseract_primary_jobs") ? metric(oldPerf, "tesseract_primary_wall_seconds_sum") / metric(oldPerf, "tesseract_primary_jobs") : null;

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
summary.showGridLines = false;
summary.tabColor = "#1F4E78";
summary.getRange("A2:H2").values = [["Smoke14: PP-OCRv6 Tiny production run", null, null, null, null, null, null, null]];
summary.getRange("A2:H2").format.font = { name: "Arial", size: 16, bold: true, color: "#1F2937" };
summary.getRange("A3:H3").format.borders = { bottom: { style: "thin", color: "#94A3B8" } };
summary.getRange("A5:B13").values = [
  ["Run", "Smoke14, full video"],
  ["Architecture", "PP-OCRv6_tiny_rec only; CPU; persistent worker; batch 8"],
  ["OCR policy", "Single pass; no retry; no Tesseract; no VL fallback"],
  ["Frames", newFrames],
  ["Physical rows", Number(newReport.counts?.physical_rows ?? 0)],
  ["Recognized rows", recognitions.length],
  ["Pipeline wall time, s", newWall],
  ["Pipeline frames/s", newFrames / newWall],
  ["Evidence status", "Raw OCR, not independent ground truth; HH remains NO_VERIFIED_EXPORT"],
];
summary.getRange("A5:A13").format.font = { name: "Arial", size: 10, bold: true, color: "#334155" };
summary.getRange("B5:B13").format.font = { name: "Arial", size: 10, color: "#1F2937" };
summary.getRange("B5:B13").format.wrapText = true;
summary.getRange("B11:B12").format.numberFormat = "0.000";

const speedRows = [
  ["Module", "Old Smoke13", "New Smoke14", "Unit", "Old/New", "Comment"],
  ["Full pipeline", oldWall, newWall, "wall s", null, "Different videos; compare normalized rates too"],
  ["Full pipeline", oldFrames / oldWall, newFrames / newWall, "frames/s", null, "End-to-end throughput"],
  ["Capture", oldCapture.wall_seconds ?? null, newCapture.wall_seconds ?? null, "wall s", null, "Streaming critical path candidate"],
  ["Capture", oldFrames / Number(oldCapture.wall_seconds ?? 1), newFrames / Number(newCapture.wall_seconds ?? 1), "frames/s", null, "All frames"],
  ["Row tracking", findStage(oldReport, "track_rows").wall_seconds ?? null, newTrack.wall_seconds ?? null, "wall s", null, "Includes wait for capture"],
  ["OCR stage", oldOcrStage.wall_seconds ?? null, newOcrStage.wall_seconds ?? null, "wall s", null, "Includes streaming wait/backlog"],
  ["Tesseract primary", oldTessSecondsPerJob, null, "s/crop", null, "Old architecture only"],
  ["VL", oldVlSecondsPerJob, null, "s/crop", null, "Old sequential bottleneck"],
  ["Tiny model", null, newIps ? 1 / newIps : null, "s/crop", null, "Inference only; batch 8"],
  ["Tiny model", null, newIps, "crop/s", null, "Inference only; batch 8"],
  ["OCR retries", metric(oldPerf, "tesseract_retry_jobs"), metric(newPerf, "retry_jobs"), "jobs", null, "New policy requires zero"],
  ["OCR fallback", metric(oldPerf, "vl_jobs"), metric(newPerf, "fallback_jobs"), "jobs", null, "New policy requires zero"],
];
summary.getRange(`A16:F${15 + speedRows.length}`).values = speedRows;
summary.getRange("A16:F16").format = { fill: "#1F4E78", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
summary.getRange(`A17:F${15 + speedRows.length}`).format.font = { name: "Arial", size: 10, color: "#1F2937" };
summary.getRange(`B17:C${15 + speedRows.length}`).format.numberFormat = "0.000";
for (let row = 17; row <= 15 + speedRows.length; row += 1) {
  summary.getRange(`E${row}`).formulas = [[`=IFERROR(B${row}/C${row},"")`]];
}
summary.getRange(`E17:E${15 + speedRows.length}`).format.numberFormat = "0.00x";

const bottleneckStart = 31;
const newStageWalls = [
  ["Capture", Number(newCapture.wall_seconds ?? 0)],
  ["Row tracking", Number(newTrack.wall_seconds ?? 0)],
  ["Tiny OCR", Number(newOcrStage.wall_seconds ?? 0)],
];
newStageWalls.sort((a, b) => b[1] - a[1]);
summary.getRange(`A${bottleneckStart}:D${bottleneckStart + 5}`).values = [
  ["Bottleneck analysis", "Measured value", "Conclusion", "Evidence"],
  ["Old architecture", oldOcrStage.wall_seconds ?? null, "OCR/VL dominated", `VL ${oldVlSecondsPerJob?.toFixed(3) ?? "n/a"} s/job; OCR stage ${Number(oldOcrStage.wall_seconds ?? 0).toFixed(1)} s`],
  ["New critical stage", newStageWalls[0][1], newStageWalls[0][0], "Largest concurrent stage wall time"],
  ["Capture PNG encoding", metric(newCapture.details?.performance, "png_encode_write_wall_seconds"), "Largest capture work bucket", "Sum across workers; not elapsed wall time"],
  ["Tiny inference", metric(newPerf, "tiny_inference_wall_seconds"), "Keeps up if OCR ends near capture", `${metric(newPerf, "tiny_jobs", 0)} crops in ${metric(newPerf, "tiny_batches", 0)} batches`],
  ["Quality boundary", recognitions.length, "Speed measured, accuracy not certified", "Workbook provides every raw crop for review"],
];
summary.getRange(`A${bottleneckStart}:D${bottleneckStart}`).format = { fill: "#8A4B08", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" } };
summary.getRange(`A${bottleneckStart + 1}:D${bottleneckStart + 5}`).format.font = { name: "Arial", size: 10, color: "#1F2937" };
summary.getRange(`B${bottleneckStart + 1}:B${bottleneckStart + 5}`).format.numberFormat = "0.000";
summary.getRange(`C${bottleneckStart + 1}:D${bottleneckStart + 5}`).format.wrapText = true;
for (const [column, width] of Object.entries({ A: 230, B: 170, C: 270, D: 390, E: 100, F: 390, G: 100, H: 100 })) {
  summary.getRange(`${column}:${column}`).format.columnWidthPx = width;
}
summary.getRange("2:2").format.rowHeightPx = 28;
summary.getRange("5:13").format.rowHeightPx = 30;
summary.getRange(`16:${15 + speedRows.length}`).format.rowHeightPx = 25;
summary.getRange(`${bottleneckStart}:${bottleneckStart + 5}`).format.rowHeightPx = 34;

const comparison = workbook.worksheets.add("Comparison");
comparison.showGridLines = false;
comparison.freezePanes.freezeRows(1);
const comparisonValues = [["Module", "Metric", "Old Smoke13", "New Smoke14", "Old/New", "Note"],
  ...data.comparison.map((row) => [row.module, row.metric, row.old_smoke13, row.new_smoke14, row.old_over_new, row.note])];
comparison.getRange(`A1:F${comparisonValues.length}`).values = comparisonValues;
comparison.getRange("A1:F1").format = { fill: "#1F4E78", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
comparison.getRange(`A2:F${comparisonValues.length}`).format.font = { name: "Arial", size: 10, color: "#1F2937" };
comparison.getRange(`C2:E${comparisonValues.length}`).format.numberFormat = "0.000";
comparison.getRange(`F2:F${comparisonValues.length}`).format.wrapText = true;
for (const [column, width] of Object.entries({ A: 190, B: 250, C: 140, D: 140, E: 100, F: 430 })) comparison.getRange(`${column}:${column}`).format.columnWidthPx = width;
comparison.tables.add(`A1:F${comparisonValues.length}`, true, "ArchitectureComparison").style = "TableStyleMedium2";

const metricsSheet = workbook.worksheets.add("Module Metrics");
metricsSheet.showGridLines = false;
metricsSheet.freezePanes.freezeRows(1);
const metricValues = [["Run", "Module", "Metric", "Value"], ...data.module_metrics.map((row) => [row.run, row.module, row.metric, row.value])];
metricsSheet.getRange(`A1:D${metricValues.length}`).values = metricValues;
metricsSheet.getRange("A1:D1").format = { fill: "#3A6D44", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
metricsSheet.getRange(`A2:D${metricValues.length}`).format.font = { name: "Arial", size: 9, color: "#1F2937" };
metricsSheet.getRange(`D2:D${metricValues.length}`).format.numberFormat = "0.000";
for (const [column, width] of Object.entries({ A: 150, B: 210, C: 430, D: 150 })) metricsSheet.getRange(`${column}:${column}`).format.columnWidthPx = width;
metricsSheet.tables.add(`A1:D${metricValues.length}`, true, "AllModuleMetrics").style = "TableStyleMedium4";

const chunkSize = 750;
const headers = ["#", "Crop", "Frame", "Table", "Tiny RAW text", "Score", "Status", "Reason", "Batch", "Retry policy", "Crop path", "physical_row_id", "observation_id", "recognition_id", "SHA256"];
for (let start = 0; start < recognitions.length; start += chunkSize) {
  const chunk = recognitions.slice(start, start + chunkSize);
  const sheetIndex = Math.floor(start / chunkSize) + 1;
  const sheet = workbook.worksheets.add(`Crops ${sheetIndex}`);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);
  const values = [headers, ...chunk.map((row) => [row.index, "", row.frame_id, row.table_session_id,
    row.raw_text, row.score, row.status, row.reason, row.batch_size, row.retry_policy,
    row.crop_path, row.physical_row_id, row.observation_id, row.recognition_id, row.crop_sha256])];
  sheet.getRange(`A1:O${values.length}`).values = values;
  sheet.getRange("A1:O1").format = { fill: "#1F4E78", font: { name: "Arial", size: 9, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
  sheet.getRange(`A2:O${values.length}`).format.font = { name: "Arial", size: 9, color: "#1F2937" };
  sheet.getRange(`A2:O${values.length}`).format.verticalAlignment = "center";
  sheet.getRange(`E2:H${values.length}`).format.wrapText = true;
  sheet.getRange(`F2:F${values.length}`).format.numberFormat = "0.000";
  sheet.getRange(`A2:O${values.length}`).format.rowHeightPx = 40;
  sheet.getRange("A1:O1").format.rowHeightPx = 34;
  for (const [column, width] of Object.entries({ A: 55, B: 460, C: 75, D: 210, E: 420, F: 70, G: 120, H: 240, I: 65, J: 150, K: 410, L: 220, M: 220, N: 220, O: 420 })) sheet.getRange(`${column}:${column}`).format.columnWidthPx = width;
  sheet.getRange(`G2:G${values.length}`).conditionalFormats.add("containsText", { text: "RAW_UNREVIEWED", format: { fill: "#DCFCE7", font: { color: "#166534" } } });
  sheet.getRange(`G2:G${values.length}`).conditionalFormats.add("containsText", { text: "UNREADABLE", format: { fill: "#FEF3C7", font: { color: "#92400E" } } });
  const table = sheet.tables.add(`A1:O${values.length}`, true, `Smoke14Crops${sheetIndex}`);
  table.style = "TableStyleMedium2";
  for (let offset = 0; offset < chunk.length; offset += 1) {
    const imageBytes = await fs.readFile(chunk[offset].crop_path);
    sheet.images.add({ dataUrl: `data:image/png;base64,${imageBytes.toString("base64")}`,
      anchor: { from: { row: offset + 1, col: 1 }, extent: { widthPx: 450, heightPx: 32 } } });
  }
}

workbook.recalculate();
const summaryInspect = await workbook.inspect({ kind: "table", sheetId: "Summary", range: "A1:F36", include: "values,formulas", tableMaxRows: 40, tableMaxCols: 6, maxChars: 14000 });
console.log(summaryInspect.ndjson);
const errorInspect = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 100 }, summary: "formula error scan" });
console.log(errorInspect.ndjson);
const summaryPreview = await workbook.render({ sheetName: "Summary", range: "A1:F36", scale: 1, format: "png" });
await fs.writeFile(path.join(dataDir, "summary_preview.png"), new Uint8Array(await summaryPreview.arrayBuffer()));
if (recognitions.length) {
  const cropPreview = await workbook.render({ sheetName: "Crops 1", range: "A1:J12", scale: 1, format: "png" });
  await fs.writeFile(path.join(dataDir, "crops_preview.png"), new Uint8Array(await cropPreview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, sheets: 3 + Math.ceil(recognitions.length / chunkSize), images: recognitions.length, rows: recognitions.length }));
