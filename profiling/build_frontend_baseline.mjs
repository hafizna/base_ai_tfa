// Build the committed frontend separately; never overwrite the working tree.
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = fileURLToPath(new URL('../', import.meta.url));
const frontend = resolve(repo, 'webapp/frontend');
// Keep the measured pre-optimization baseline reproducible after this is committed.
const baselineRef = process.argv[2] ?? 'ce823d9eaa308bed13d3848f0a360e14585b4998';
const require = createRequire(resolve(frontend, 'package.json'));
const { build } = await import(pathToFileURL(require.resolve('vite')).href);
const { default: react } = await import(pathToFileURL(require.resolve('@vitejs/plugin-react')).href);
const original = new Map([
  'webapp/frontend/src/App.tsx',
  'webapp/frontend/src/components/plot/PlotlyChart.tsx',
].map(path => [resolve(repo, path).replaceAll('\\', '/'),
  execFileSync('git', ['show', `${baselineRef}:${path}`], { cwd: repo, encoding: 'utf8' }),
]));

await build({
  root: frontend,
  configFile: false,
  plugins: [{
    name: 'committed-loading-baseline',
    enforce: 'pre',
    load(id) { return original.get(id.split('?')[0].replaceAll('\\', '/')) ?? null; },
  }, react()],
  build: {
    outDir: resolve(repo, 'tmp/performance-before'),
    emptyOutDir: false,
    manifest: true,
  },
});
process.stdout.write(`Baseline commit: ${execFileSync('git', ['rev-parse', baselineRef], { cwd: repo, encoding: 'utf8' }).trim()}\n`);
