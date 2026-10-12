// Freeze real read-only API responses to isolate frontend costs in A/B runs.
import { request } from 'node:http';
import { writeFileSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const output = fileURLToPath(new URL('../tmp/browser-validation/api-responses.json', import.meta.url));
const sessions = JSON.parse(readFileSync(new URL('../tmp/browser-validation/sessions.json', import.meta.url), 'utf8'));
const responses = [];
for (const item of sessions) {
  const id = item.analysis_id;
  const paths = [`/api/analysis/${id}`, `/api/analysis/${id}/canonical`,
    ...(item.suggested_relay_type === 'OCR' ? [] : [
      `/api/analyze/21/extract-features?analysis_id=${id}`,
      `/api/analyze/21/electrical-params?analysis_id=${id}`,
      `/api/analyze/21/fault-classification?analysis_id=${id}`,
      `/api/analyze/21/locus-events?analysis_id=${id}`,
    ])];
  for (const path of paths) {
    const start = performance.now();
    const response = await new Promise((done, reject) => {
      const req = request({ hostname: '127.0.0.1', port: 8901, path,
        headers: { 'Accept-Encoding': 'gzip' } }, res => {
        const chunks = [];
        res.on('data', chunk => chunks.push(chunk));
        res.on('end', () => done({ method: 'GET', path, status: res.statusCode,
          headers: res.headers, body: Buffer.concat(chunks).toString('base64') }));
        res.on('error', reject);
      });
      req.setTimeout(300000, () => req.destroy(new Error('API snapshot timeout')));
      req.on('error', reject); req.end();
    });
    if (response.status !== 200) throw new Error(`API snapshot failed: ${path} (${response.status})`);
    responses.push(response);
    writeFileSync(output, JSON.stringify(responses));
    process.stdout.write(JSON.stringify({ rawBytes: item.raw_bytes, path: path.replace(id, '<session>'), elapsedMs: Math.round(performance.now() - start) }) + '\n');
  }
}
