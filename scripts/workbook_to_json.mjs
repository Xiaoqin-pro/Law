import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [, , workbookPath, sheetName] = process.argv;
if (!workbookPath || !sheetName) {
  throw new Error("Usage: workbook_to_json.mjs <workbook.xlsx> <sheet_name>");
}

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const sheet = workbook.worksheets.getItem(sheetName);
const usedRange = sheet.getUsedRange();
const values = usedRange?.values ?? [];
const headers = values.length ? values[0].map((value) => String(value ?? "")) : [];
const rows = values.slice(1).map((valuesRow) => {
  const row = {};
  headers.forEach((header, index) => {
    row[header] = valuesRow[index] ?? "";
  });
  return row;
});

process.stdout.write(JSON.stringify({ sheet: sheetName, headers, rows }));
