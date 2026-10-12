import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { startPreviews } from './browser_previews.mjs';

const repo = fileURLToPath(new URL('../', import.meta.url));
const require = createRequire(resolve(repo, 'tmp/browser-tools/package.json'));
const { chromium } = require('playwright');
const executablePath = process.env.CHROMIUM_EXECUTABLE;
const output = resolve(repo, 'tmp/browser-validation');
mkdirSync(output, { recursive: true });
const sessions = JSON.parse(readFileSync(resolve(output, 'sessions.json'), 'utf8'));
const repeats = Number(process.env.PERF_REPEATS ?? 3);
const resultsFile = resolve(output, repeats ? 'browser-results.json' : 'probe-results.json');
const report = { browser: null, method: 'Loopback gzip; unthrottled headless; JS heap after GC; cold context then reload; same stored records; no backend change', samples: [], checks: [] };
report.apiReplay = process.env.PERF_REPLAY_API === '1';
const previews = await startPreviews();
let browser;

async function metrics(page, cdp, start) {
  const readyMs = performance.now() - start;
  await cdp.send('HeapProfiler.collectGarbage');
  const { metrics: values } = await cdp.send('Performance.getMetrics');
  const heap = values.find(item => item.name === 'JSHeapUsedSize')?.value;
  const resources = await page.evaluate(() => {
    const entries = performance.getEntriesByType('resource');
    return {
      jsTransferBytes: entries.filter(item => /\.js(?:\?|$)/.test(item.name)).reduce((sum, item) => sum + item.transferSize, 0),
      apiTransferBytes: entries.filter(item => item.name.includes('/api/')).reduce((sum, item) => sum + item.transferSize, 0),
      totalTransferBytes: entries.reduce((sum, item) => sum + item.transferSize, 0),
      graphCount: document.querySelectorAll('.js-plotly-plot').length,
    };
  });
  return { readyMs: Math.round(readyMs), jsHeapBytes: heap, ...resources };
}

async function waitReady(page, route) {
  if (route === '/') await page.getByText('DFR Analyser', { exact: true }).waitFor();
  else await page.waitForFunction(() => [...document.querySelectorAll('.js-plotly-plot')].some(plot => plot._fullLayout), null, { timeout: 120000 });
  await page.waitForLoadState('networkidle', { timeout: 120000 });
  if (route !== '/') await page.waitForFunction(() => {
    const text = document.body.innerText;
    return !['Extracting features...', 'Menghitung...', 'Membaca urutan kejadian...', 'Computing...']
      .some(label => text.includes(label));
  }, null, { timeout: 120000 });
  await page.waitForLoadState('networkidle', { timeout: 120000 });
  await page.evaluate(() => document.fonts.ready);
}

try {
  browser = await chromium.launch({ headless: true, executablePath, args: ['--use-angle=swiftshader'] });
  report.browser = browser.version();
  if (repeats && report.apiReplay) {
    // Prime any analysis POSTs with real API results before measuring either build.
    for (const item of sessions) {
      const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
      const page = await context.newPage();
      const route = `/workspace/${item.suggested_relay_type ?? '21'}/${item.analysis_id}`;
      await page.goto(`http://127.0.0.1:4174${route}`);
      await waitReady(page, route);
      await context.close();
      process.stdout.write(`Primed API responses for ${item.raw_bytes} byte record\n`);
    }
  }
  for (let iteration = 0; iteration < repeats; iteration++) {
    // Alternate ordering to reduce systematic warmup/order bias.
    for (const variant of iteration % 2 ? ['after', 'before'] : ['before', 'after']) {
      const port = variant === 'before' ? 4174 : 4175;
      for (const item of [null, ...sessions]) {
        const relay = item?.suggested_relay_type ?? '21';
        const route = item ? `/workspace/${relay}/${item.analysis_id}` : '/';
        const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
        const page = await context.newPage();
        page.setDefaultTimeout(120000);
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        const cdp = await context.newCDPSession(page);
        await cdp.send('Performance.enable');
        for (const mode of ['cold', 'warm']) {
          const start = performance.now();
          if (mode === 'cold') await page.goto(`http://127.0.0.1:${port}${route}`);
          else await page.reload();
          await waitReady(page, route);
          const sample = { variant, iteration, mode, rawBytes: item?.raw_bytes ?? 0,
            ...await metrics(page, cdp, start), errors: [...errors] };
          report.samples.push(sample);
          writeFileSync(resultsFile, JSON.stringify(report, null, 2));
          process.stdout.write(JSON.stringify(sample) + '\n');
          assert.deepEqual(errors, [], 'Unexpected application JavaScript error');
        }
        if (iteration === 0 && item) {
          const snapshot = await page.evaluate(() => [...document.querySelectorAll('.js-plotly-plot')].filter(plot => plot._fullLayout).map(plot => ({
            data: plot.data.map(trace => ({ type: trace.type ?? 'scatter', name: trace.name, x: Array.from(trace.x ?? []), y: Array.from(trace.y ?? []), fill: trace.fill })),
            shapes: plot.layout.shapes,
          })));
          const fingerprint = createHash('sha256').update(JSON.stringify(snapshot)).digest('hex');
          report.checks.push({ variant, rawBytes: item.raw_bytes, graphFingerprint: fingerprint, graphCount: snapshot.length });
          await page.screenshot({ path: resolve(output, `${variant}-${item.target_bytes}-workspace.png`), fullPage: true });
          const plotIndex = await page.evaluate(() => [...document.querySelectorAll('.js-plotly-plot')]
            .findIndex(plot => plot.data?.some(trace => trace.x?.length > 10 && trace.y?.length > 10)));
          assert.ok(plotIndex >= 0, 'Expected a populated waveform plot');
          const firstPlot = page.locator('.js-plotly-plot').nth(plotIndex);
          await firstPlot.scrollIntoViewIfNeeded();
          const hoverPoint = await firstPlot.evaluate(plot => {
            const trace = plot.data.find(trace => trace.x?.length > 10 && trace.y?.length > 10);
            const [start, end] = plot._fullLayout.xaxis.range;
            const middle = (start + end) / 2;
            const index = Math.max(0, trace.x.findIndex(value => value >= middle));
            const rect = plot.getBoundingClientRect();
            return { x: rect.left + plot._fullLayout.xaxis._offset + plot._fullLayout.xaxis.c2p(trace.x[index]),
              y: rect.top + plot._fullLayout.yaxis._offset + plot._fullLayout.yaxis.c2p(trace.y[index]) };
          });
          await page.mouse.move(hoverPoint.x, hoverPoint.y);
          await firstPlot.locator('.hoverlayer').locator('text').first().waitFor({ state: 'visible' });
          const zoom = await firstPlot.evaluate(async plot => {
            const [start, end] = plot._fullLayout.xaxis.range;
            const range = [start + (end - start) * 0.25, start + (end - start) * 0.75];
            await window.Plotly.relayout(plot, { 'xaxis.range': range });
            return { expected: range, actual: plot.layout.xaxis.range };
          });
          assert.deepEqual(zoom.actual, zoom.expected);
          report.checks.push({ variant, rawBytes: item.raw_bytes, waveformHover: true, waveformZoom: true });
          if (item.target_bytes === 1_000_000) {
            await page.getByRole('button', { name: 'Plot TCC', exact: true }).click();
            await page.locator('[data-pdf-chart-id="overcurrent_overlay"][data-pdf-ready="true"]').waitFor();
            const pdfRequest = page.waitForResponse(response => response.url().includes('/api/report/') && response.request().method() === 'POST');
            const downloadEvent = page.waitForEvent('download');
            const [response, download] = await Promise.all([pdfRequest, downloadEvent,
              page.getByRole('button', { name: /Download PDF Report/ }).click()]);
            assert.equal(response.status(), 200, 'PDF endpoint must succeed');
            const body = response.request().postDataJSON();
            assert.ok(body.charts.length > 0, 'PDF must contain captured PNG charts');
            const path = resolve(output, `${variant}-report.pdf`);
            await download.saveAs(path);
            assert.equal(readFileSync(path).subarray(0, 5).toString(), '%PDF-');
            report.checks.push({ variant, pdf: 'passed', capturedCharts: body.charts.map(chart => chart.id), pdfBytes: readFileSync(path).length });
          }
        }
        await context.close();
      }
    }
  }
  for (const item of repeats ? sessions : []) {
    const before = report.checks.find(check => check.variant === 'before' && check.rawBytes === item.raw_bytes);
    const after = report.checks.find(check => check.variant === 'after' && check.rawBytes === item.raw_bytes);
    assert.equal(after.graphFingerprint, before.graphFingerprint, `Graph data changed for ${item.raw_bytes} byte record`);
  }
  // Isolate compatibility checks for histogram and actual WebGL rendering.
  for (const variant of ['before', 'after']) {
    const port = variant === 'before' ? 4174 : 4175;
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    page.setDefaultTimeout(120000);
    await page.goto(`http://127.0.0.1:${port}/__plotly_probe`);
    await page.waitForFunction(() => window.probeReady);
    const checks = await page.evaluate(async () => {
      const Plotly = window.Plotly;
      const host = document.getElementById('probe');
      const result = [];
      for (const type of ['histogram', 'scattergl']) {
        const plot = document.createElement('div'); host.append(plot);
        const data = type === 'histogram'
          ? [{ type, x: [17, 17.1, 17.2, 17.3, 17.4], marker: { color: '#ff0000' }, xbins: { start: 17, end: 17.5, size: 0.1 } }]
          : [{ type, x: Array.from({ length: 3600 }, (_, i) => i / 10), y: Array.from({ length: 3600 }, (_, i) => Math.sin(i / 10)), mode: 'lines+markers', line: { color: '#ff0000' }, marker: { color: '#ff0000' } }];
        await Plotly.newPlot(plot, data, { width: 700, height: 450 });
        if (type === 'histogram' && !plot.querySelector('.barlayer .point')) throw new Error('Histogram did not render');
        if (type === 'scattergl' && !plot.querySelector('.gl-container canvas')) throw new Error('WebGL canvas missing');
        await Plotly.relayout(plot, { 'xaxis.range': type === 'histogram' ? [17, 17.5] : [0, 10] });
        Plotly.Fx.hover(plot, type === 'histogram' ? [{ curveNumber: 0, pointNumber: 1 }]
          : { xval: 0.1, yval: Math.sin(0.1) });
        await new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)));
        if (!plot.querySelector('.hoverlayer').textContent.trim()) throw new Error(`${type} hover missing`);
        const png = await Plotly.toImage(plot, { format: 'png', width: 700, height: 450 });
        if (!png.startsWith('data:image/png;base64,')) throw new Error('PNG capture missing');
        const image = new Image();
        await new Promise((done, reject) => { image.onload = done; image.onerror = reject; image.src = png; });
        const canvas = document.createElement('canvas'); canvas.width = image.width; canvas.height = image.height;
        const ctx = canvas.getContext('2d'); ctx.drawImage(image, 0, 0);
        const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        let redPixels = 0;
        for (let i = 0; i < pixels.length; i += 4) if (pixels[i] > 180 && pixels[i + 1] < 100 && pixels[i + 2] < 100) redPixels++;
        if (redPixels < 20) throw new Error(`${type} PNG does not contain rendered trace pixels`);
        result.push({ type, pngBytes: atob(png.split(',')[1]).length, redPixels, zoom: plot.layout.xaxis.range, hover: true });
        Plotly.purge(plot); plot.remove();
      }
      return result;
    });
    report.checks.push({ variant, plotCompatibility: checks });
    process.stdout.write(JSON.stringify({ variant, plotCompatibility: checks }) + '\n');
    await context.close();
  }
  report.status = 'passed';
} catch (error) {
  report.status = 'failed'; report.failure = error.stack;
  throw error;
} finally {
  writeFileSync(resultsFile, JSON.stringify(report, null, 2));
  if (browser) await browser.close();
  await previews.close();
}
