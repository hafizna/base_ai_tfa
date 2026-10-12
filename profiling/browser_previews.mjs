import { createServer, request } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';

const repo = fileURLToPath(new URL('../', import.meta.url));
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.svg': 'image/svg+xml', '.woff': 'font/woff', '.woff2': 'font/woff2', '.json': 'application/json' };

export async function startPreviews() {
  const servers = [];
  const snapshots = new Map();
  const pending = new Map();
  function key(method, path, body = Buffer.alloc(0)) {
    return `${method}:${path}:${createHash('sha256').update(body).digest('hex')}`;
  }
  if (process.env.PERF_REPLAY_API === '1') {
    const saved = JSON.parse(await readFile(resolve(repo, 'tmp/browser-validation/api-responses.json'), 'utf8'));
    for (const item of saved) snapshots.set(key(item.method, item.path), { ...item, body: Buffer.from(item.body, 'base64') });
  }
  for (const [variant, port, directory] of [
    ['before', 4174, resolve(repo, 'tmp/performance-before')],
    ['after', 4175, resolve(repo, 'webapp/frontend/dist')],
  ]) {
    const manifest = JSON.parse(await readFile(resolve(directory, '.vite/manifest.json'), 'utf8'));
    const entry = Object.values(manifest).find(item => item.isEntry);
    const chart = Object.values(manifest).find(item => item.file.includes('/PlotlyChart-') && item.file.endsWith('.js'));
    const cache = new Map();
    const server = createServer(async (req, res) => {
      if (req.url.startsWith('/api/')) {
        if (process.env.PERF_REPLAY_API === '1' && (req.method === 'GET' || req.url.startsWith('/api/analyze/'))) {
          const chunks = [];
          for await (const chunk of req) chunks.push(chunk);
          const body = Buffer.concat(chunks);
          const id = key(req.method, req.url, body);
          if (!snapshots.has(id) && !pending.has(id)) {
            pending.set(id, new Promise((done, reject) => {
              const upstreamRequest = request({ hostname: '127.0.0.1', port: 8901, path: req.url,
                method: req.method, headers: { ...req.headers, host: '127.0.0.1:8901', 'accept-encoding': 'gzip' } }, upstream => {
                const responseChunks = [];
                upstream.on('data', chunk => responseChunks.push(chunk));
                upstream.on('end', () => {
                  const response = { status: upstream.statusCode, headers: upstream.headers, body: Buffer.concat(responseChunks) };
                  if (response.status === 200) snapshots.set(id, response);
                  done(response);
                });
                upstream.on('error', reject);
              });
              upstreamRequest.on('error', reject); upstreamRequest.end(body);
            }).finally(() => pending.delete(id)));
          }
          try {
            const response = snapshots.get(id) ?? await pending.get(id);
            res.writeHead(response.status, response.headers); res.end(response.body);
          } catch (error) { res.writeHead(502); res.end(error.message); }
          return;
        }
        const proxy = request({ hostname: '127.0.0.1', port: 8901, path: req.url,
          method: req.method, headers: { ...req.headers, host: '127.0.0.1:8901' } }, upstream => {
          res.writeHead(upstream.statusCode, upstream.headers);
          upstream.pipe(res);
        });
        proxy.on('error', error => { res.writeHead(502); res.end(error.message); });
        req.pipe(proxy);
        return;
      }
      try {
        const url = new URL(req.url, `http://127.0.0.1:${port}`);
        const pathname = decodeURIComponent(url.pathname);
        let data, contentType;
        if (pathname === '/__plotly_probe') {
          data = Buffer.from(`<!doctype html><html><body><div id="root"></div><div id="probe"></div>
            <script type="module">await import('/${entry.file}');
            ${chart ? `await import('/${chart.file}');` : ''}
            window.probeReady = true;</script></body></html>`);
          contentType = mime['.html'];
        } else {
          let path = resolve(directory, '.' + pathname);
          if (relative(directory, path).startsWith('..')) { res.writeHead(403); res.end(); return; }
          try { data = await readFile(path); } catch { path = resolve(directory, 'index.html'); data = await readFile(path); }
          contentType = mime[extname(path)] ?? 'application/octet-stream';
        }
        const compressed = /javascript|text\/|json|svg/.test(contentType) && /gzip/.test(req.headers['accept-encoding'] ?? '');
        const key = pathname + (compressed ? ':gzip' : ':plain');
        if (!cache.has(key)) cache.set(key, compressed ? gzipSync(data) : data);
        const body = cache.get(key);
        res.writeHead(200, { 'Content-Type': contentType, 'Content-Length': body.length,
          'Cache-Control': pathname.startsWith('/assets/') ? 'public,max-age=3600' : 'no-store',
          ...(compressed ? { 'Content-Encoding': 'gzip', Vary: 'Accept-Encoding' } : {}) });
        res.end(body);
      } catch (error) { res.writeHead(500); res.end(error.message); }
    });
    await new Promise((ready, reject) => { server.once('error', reject); server.listen(port, '127.0.0.1', ready); });
    servers.push({ variant, port, server });
  }
  return { servers, close: () => Promise.all(servers.map(({ server }) => new Promise(done => server.close(done)))) };
}
