# Profiling

Lightweight performance instrumentation for the FastAPI backend.

## Frontend bundle baseline

Build with a manifest from `webapp/frontend`:

```powershell
npm.cmd run build -- --manifest
```

Then, from the repository root:

```powershell
node profiling/frontend_bundle.mjs
```

The report includes the entry's static JavaScript dependencies, additional
dependencies for each lazy page, and all JavaScript chunks. Gzip sizes are
estimates using Node defaults. It excludes CSS/fonts and does not measure
network latency or browser rendering. See
[`docs/performance-baseline-2026-10-12.md`](../docs/performance-baseline-2026-10-12.md)
for the first before/after measurement.

Verify the built Plotly chunk in a simulated DOM without Node globals:

```powershell
node --experimental-vm-modules profiling/verify_plotly_bundle.mjs
```

This checks trace registration, SVG rendering, zoom, and SVG export from the
actual production JavaScript. WebGL rendering, hover, PNG capture and PDF export
still need a real-browser check. It does not measure browser performance.

## Prepare real-browser comparison

Build the committed loading behavior separately, keeping current edits intact:

```powershell
node profiling/build_frontend_baseline.mjs
```

The baseline is written to ignored `tmp/performance-before`. It uses App and
PlotlyChart from pre-optimization commit `ce823d9eaa308bed13d3848f0a360e14585b4998`
with the installed dependencies; the current frontend build remains in
`webapp/frontend/dist`. Pass another Git ref as the script's first argument
if needed. The default stays fixed after committing the optimization.

Select three local CFG/DAT pairs closest to 1/5/10 MB, then run an isolated API:

```powershell
python -B profiling/select_browser_cases.py 'C:/path/to/raw_data'
python -B profiling/browser_test_backend.py --port 8901
```

In a second terminal, create shared test sessions:

```powershell
node profiling/prepare_browser_sessions.mjs
```

Session IDs and local raw paths are saved only under ignored
`tmp/browser-validation`. Reuse these IDs in both frontend builds. The test API
binds to loopback, clears DATABASE_URL, isolates temporary/session and incident
storage, and disables training retention. Python may require execution outside
the restricted sandbox in this environment. This preparation does not itself
validate browser rendering or latency.

## Run headless Chromium validation

Install test tooling separately from application dependencies, then freeze
read-only API responses (the isolated API above must be running):

```powershell
npm.cmd install --prefix tmp/browser-tools --ignore-scripts playwright@1.58.2
node profiling/record_api_responses.mjs
```

Choose an installed Chromium executable, or install the matching Chromium using
`node tmp/browser-tools/node_modules/playwright/cli.js install chromium` and
omit CHROMIUM_EXECUTABLE. The completed local run used an already-installed
Chromium 153, explicitly selected by executable path.

```powershell
$env:CHROMIUM_EXECUTABLE = 'C:/path/to/chrome.exe'
$env:PERF_REPEATS = '3'
$env:PERF_REPLAY_API = '1'
node profiling/browser_validation.mjs
node profiling/summarize_browser_results.mjs
python -B profiling/review_browser_pdfs.py
```

The harness starts temporary gzip preview servers on loopback ports 4174/4175
and closes them after testing. It checks real-record trace equivalence,
waveform hover/zoom, histogram/WebGL PNG pixels, and actual OCR PDF download.
PDF checks require the OCR case and its Plot TCC action. Test artifacts remain
under ignored `tmp/browser-validation`. API replay isolates frontend costs;
it must not be presented as live backend or production performance. Set
PERF_REPEATS=0 for just the deterministic trace compatibility checks.
PyMuPDF renders the downloaded PDFs for visual review.

Method references: [Playwright browsers](https://playwright.dev/docs/browsers)
and [CDP sessions](https://playwright.dev/docs/api/class-cdpsession).

## What's here

| File | Purpose |
|---|---|
| `profile_request.ps1` | Start uvicorn under py-spy, hit a representative endpoint, save flamegraph SVG |
| `profile_request.sh`  | POSIX counterpart for Linux/macOS deploys |
| `flamegraphs/`        | (gitignored) py-spy SVG output |

## Quick start (Windows / PowerShell)

```powershell
# Install once
python -m pip install --user py-spy

# Default profile: 30 s sample of uvicorn while running the bundled smoke request
./profiling/profile_request.ps1

# Profile a specific upload (longer sample, custom cfg/dat pair)
./profiling/profile_request.ps1 -DurationSec 60 `
  -CfgFile "data/synthetic/transformer/INRUSH/...INRUSH...cfg" `
  -DatFile "data/synthetic/transformer/INRUSH/...INRUSH...dat"
```

Open the resulting `profiling/flamegraphs/profile_*.svg` in a browser. Wider
boxes = more wall-clock time.

## What to look for

The hypothesis going into Option A:

- **Pickle load on first request** — should disappear after the lifespan
  preload (see `webapp/api/main.py` startup hook).
- **COMTRADE parse loops** — `core/comtrade_parser.py` reading `.dat` bytes.
- **Digital sequence loop** — `webapp/api/ml_predict.py:_digital_sequence_features`,
  still pure Python.
- **JSON serialization of waveform arrays** — `webapp/api/routers/upload.py`,
  emits 100k+ float samples per channel.

If something else dominates, the optimization plan changes accordingly.
