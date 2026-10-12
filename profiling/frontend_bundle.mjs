// Run after: npm.cmd run build -- --manifest (in webapp/frontend).
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { gzipSync } from 'node:zlib';

const dist = fileURLToPath(new URL('../webapp/frontend/dist/', import.meta.url));
const manifest = JSON.parse(readFileSync(resolve(dist, '.vite/manifest.json'), 'utf8'));
const entries = Object.keys(manifest).filter(key => manifest[key].isEntry);
if (entries.length !== 1) throw new Error(`Expected one application entry; found ${entries.length}`);

function staticClosure(key, visited = new Set()) {
  if (visited.has(key)) return visited;
  if (!manifest[key]) throw new Error(`Missing manifest dependency: ${key}`);
  visited.add(key);
  for (const imported of manifest[key].imports ?? []) staticClosure(imported, visited);
  return visited;
}

function measure(keys) {
  const files = [...new Set([...keys].map(key => manifest[key].file))].filter(file => file.endsWith('.js'));
  let rawBytes = 0;
  let gzipBytes = 0;
  for (const file of files) {
    const buffer = readFileSync(resolve(dist, file));
    rawBytes += buffer.length;
    gzipBytes += gzipSync(buffer).length;
  }
  return { files, rawBytes, gzipBytes };
}

const initial = staticClosure(entries[0]);
const routes = Object.keys(manifest).filter(key => manifest[key].isDynamicEntry);
console.log(JSON.stringify({
  scope: 'JavaScript only; gzip estimate, not measured network transfer or browser latency',
  initial: measure(initial),
  // Additional JS for each route on a cold visit after the entry has loaded.
  routes: Object.fromEntries(routes.map(key => [key,
    measure([...staticClosure(key)].filter(dependency => !initial.has(dependency))),
  ])),
  total: measure(Object.keys(manifest)),
}, null, 2));
