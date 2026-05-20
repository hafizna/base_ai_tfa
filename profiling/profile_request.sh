#!/usr/bin/env bash
# POSIX counterpart to profile_request.ps1 — for Linux/macOS dev or CI.
#
# Usage:
#   ./profiling/profile_request.sh [duration_sec] [port] [cfg] [dat]
#
# Defaults: 30 seconds, port 8901, synthetic INRUSH file.

set -euo pipefail

duration="${1:-30}"
port="${2:-8901}"
cfg="${3:-data/synthetic/transformer/INRUSH/GI_TAMBUN_150KV_INRUSH_20260408_114808_667.cfg}"
dat="${4:-data/synthetic/transformer/INRUSH/GI_TAMBUN_150KV_INRUSH_20260408_114808_667.dat}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

out_dir="profiling/flamegraphs"
mkdir -p "$out_dir"
stamp="$(date +%Y%m%d_%H%M%S)"
svg="$out_dir/profile_${stamp}.svg"

if ! command -v py-spy >/dev/null 2>&1; then
    echo "py-spy not on PATH. Install: pip install --user py-spy" >&2
    exit 1
fi

echo "Launching uvicorn on port $port..."
python -m uvicorn webapp.api.main:app --host 127.0.0.1 --port "$port" \
    --workers 1 --log-level warning > "$out_dir/uvicorn.log" 2>&1 &
uvicorn_pid=$!
trap 'kill $uvicorn_pid 2>/dev/null || true' EXIT

# Wait for /api/health
for _ in $(seq 1 60); do
    sleep 0.5
    if curl -sf "http://127.0.0.1:$port/api/health" >/dev/null 2>&1; then
        break
    fi
done

echo "Server ready. Starting py-spy sample for ${duration}s..."
py-spy record -o "$svg" -d "$duration" -f flamegraph -p "$uvicorn_pid" --idle &
spy_pid=$!

iterations=$(( duration / 5 > 0 ? duration / 5 : 1 ))
echo "Firing $iterations upload(s)..."
for _ in $(seq 1 "$iterations"); do
    resp="$(curl -s -X POST "http://127.0.0.1:$port/api/upload" \
        -F "cfg_file=@${cfg}" -F "dat_file=@${dat}" || true)"
    aid="$(echo "$resp" | python -c "import json,sys; print(json.load(sys.stdin).get('analysis_id',''))" 2>/dev/null || true)"
    if [ -n "$aid" ]; then
        curl -s -X POST "http://127.0.0.1:$port/api/analyze/21/ai-analysis" \
            -H "Content-Type: application/json" \
            -d "{\"analysis_id\":\"$aid\",\"fault_inception_angle_deg\":0,\"fault_duration_ms\":0,\"prefault_load_a\":0,\"impedance_at_trip_ohm\":0,\"waveform_asymmetry\":0,\"dc_offset\":0,\"ar_result\":null}" >/dev/null || true
    fi
    sleep 0.25
done

wait $spy_pid
echo "Flamegraph written: $svg"
