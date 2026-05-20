# Profiling

Lightweight performance instrumentation for the FastAPI backend.

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
