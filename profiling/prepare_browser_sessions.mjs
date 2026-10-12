// Upload selected cases once to the isolated local API; reuse IDs in both builds.
import { readFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const casesPath = fileURLToPath(new URL('../tmp/browser-validation/cases.json', import.meta.url));
const outputPath = fileURLToPath(new URL('../tmp/browser-validation/sessions.json', import.meta.url));
const api = 'http://127.0.0.1:8901';
const health = await fetch(`${api}/api/health`).then(response => response.json());
if (health.status !== 'ok' || health.analysis_storage !== 'filesystem') {
  throw new Error('Expected healthy isolated filesystem API on port 8901');
}
const sessions = [];
for (const item of JSON.parse(readFileSync(casesPath, 'utf8'))) {
  const form = new FormData();
  form.append('cfg_file', new Blob([readFileSync(item.cfg)]), basename(item.cfg));
  form.append('dat_file', new Blob([readFileSync(item.dat)]), basename(item.dat));
  const start = performance.now();
  const response = await fetch(`${api}/api/upload`, { method: 'POST', body: form });
  const result = await response.json();
  if (!response.ok) throw new Error(`Upload failed: ${response.status} ${JSON.stringify(result)}`);
  sessions.push({ ...item, ...result, upload_ms: Math.round(performance.now() - start) });
  process.stdout.write(JSON.stringify({ raw_bytes: item.raw_bytes, ...result,
    upload_ms: sessions.at(-1).upload_ms }) + '\n');
}
writeFileSync(outputPath, JSON.stringify(sessions, null, 2));
