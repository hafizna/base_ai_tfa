// Production-JS smoke check in a simulated browser DOM, not a real GPU/browser.
// Run after building with --manifest:
// node --experimental-vm-modules profiling/verify_plotly_bundle.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SourceTextModule } from 'node:vm';

const frontend = fileURLToPath(new URL('../webapp/frontend/', import.meta.url));
const require = createRequire(resolve(frontend, 'package.json'));
const { JSDOM, VirtualConsole } = require('jsdom');
const dist = resolve(frontend, 'dist');
const manifest = JSON.parse(readFileSync(resolve(dist, '.vite/manifest.json'), 'utf8'));
const chart = Object.values(manifest).find(entry => entry.file.includes('/PlotlyChart-') && entry.file.endsWith('.js'));
assert.ok(chart, 'Expected shared PlotlyChart chunk in the production manifest');
const console = new VirtualConsole();
const dom = new JSDOM('<!doctype html><div id="plot"></div>', {
  url: 'http://localhost/', runScripts: 'outside-only', pretendToBeVisual: true,
  virtualConsole: console,
});
const context = dom.getInternalVMContext();
assert.equal(typeof dom.window.Buffer, 'undefined');
assert.equal(typeof dom.window.global, 'undefined');
const modules = new Map();
function getModule(path) {
  if (!modules.has(path)) {
    modules.set(path, new SourceTextModule(readFileSync(path, 'utf8'), {
      context, identifier: path,
    }));
  }
  return modules.get(path);
}

try {
  const entry = getModule(resolve(dist, chart.file));
  await entry.link((specifier, importer) => {
    assert.ok(specifier.startsWith('./'), `Unexpected external dependency: ${specifier}`);
    return getModule(resolve(dirname(importer.identifier), specifier));
  });
  await entry.evaluate();
  const Plotly = dom.window.Plotly;
  assert.deepEqual(Object.keys(Plotly.PlotSchema.get().traces).sort(), ['histogram', 'scatter', 'scattergl']);
  const plot = dom.window.document.getElementById('plot');
  await Plotly.newPlot(plot, [
    { type: 'scatter', x: [0, 1, 2], y: [0, 2, 0], mode: 'lines' },
    { type: 'histogram', x: [0.1, 0.2, 1.1, 1.2], xbins: { start: 0, end: 2, size: 1 } },
  ], {
    width: 600, height: 400,
    shapes: [{ type: 'circle', x0: 0, y0: 0, x1: 1, y1: 1 }],
    annotations: [{ x: 1, y: 2, text: 'Trip', showarrow: false }],
  }, { staticPlot: true });
  assert.ok(plot.querySelector('.scatterlayer .js-line'));
  assert.equal(plot.querySelectorAll('.barlayer .point').length, 2);
  assert.ok(plot.querySelector('.shapelayer path'));
  await Plotly.relayout(plot, { 'xaxis.range': [0, 1] });
  assert.equal(JSON.stringify(plot.layout.xaxis.range), '[0,1]');
  const svg = await Plotly.toImage(plot, { format: 'svg', width: 600, height: 400 });
  assert.match(svg, /^data:image\/svg\+xml/);
  assert.ok(decodeURIComponent(svg).includes('Trip'));
  Plotly.purge(plot);
  process.stdout.write('Production Plotly smoke passed: trace registry, SVG render, zoom and SVG export. WebGL/PNG/PDF require a real browser.\n');
} finally {
  dom.window.close();
}
