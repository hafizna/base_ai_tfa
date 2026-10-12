import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const report = JSON.parse(readFileSync(new URL('../tmp/browser-validation/browser-results.json', import.meta.url), 'utf8'));
assert.equal(report.status, 'passed', 'Only summarize a completed successful run');
function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}
const rows = [];
for (const rawBytes of [...new Set(report.samples.map(sample => sample.rawBytes))]) {
  for (const mode of ['cold', 'warm']) {
    const row = { rawBytes, mode };
    for (const variant of ['before', 'after']) {
      const samples = report.samples.filter(sample => sample.rawBytes === rawBytes && sample.mode === mode && sample.variant === variant);
      row[variant] = { n: samples.length,
        readyMsMedian: median(samples.map(sample => sample.readyMs)),
        readyMsRange: [Math.min(...samples.map(sample => sample.readyMs)), Math.max(...samples.map(sample => sample.readyMs))],
        jsHeapBytesMedian: median(samples.map(sample => sample.jsHeapBytes)),
        jsTransferBytesMedian: median(samples.map(sample => sample.jsTransferBytes)),
        apiTransferBytesMedian: median(samples.map(sample => sample.apiTransferBytes)),
      };
    }
    rows.push(row);
  }
}
console.log(JSON.stringify({ browser: report.browser, method: report.method, apiReplay: report.apiReplay, rows, checks: report.checks }, null, 2));
