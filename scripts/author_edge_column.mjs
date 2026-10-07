import fs from 'node:fs/promises';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const [specPath, output] = process.argv.slice(2);
const specs = JSON.parse(await fs.readFile(specPath, 'utf8'));
const workbook = Workbook.create();
for (const [name, spec] of Object.entries(specs)) {
  const sheet = workbook.worksheets.add(name);
  sheet.getRange('A1').values = [['edge_status']];
  for (let start = 0; start < spec.values.length; start += 400) {
    const block = spec.values.slice(start, start+400).map(x => [x]);
    sheet.getRangeByIndexes(start+1, 0, block.length, 1).values = block;
  }
  sheet.getRange('A:A').format.columnWidth = 23;
}
workbook.recalculate();
const result = await SpreadsheetFile.exportXlsx(workbook);
await result.save(output);
console.log(JSON.stringify({authoredSheets: Object.keys(specs).length}));
