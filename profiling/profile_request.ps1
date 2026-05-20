<#
.SYNOPSIS
    Profile the FastAPI backend with py-spy while exercising a real request.

.DESCRIPTION
    Launches uvicorn in the background, waits for /api/health to respond, then
    attaches py-spy to capture a flamegraph for DurationSec seconds while a
    smoke request runs against the server.

.PARAMETER DurationSec
    py-spy sampling duration in seconds. Default 30.

.PARAMETER Port
    Local port for uvicorn. Default 8901 (avoid conflict with dev server on 8000).

.PARAMETER CfgFile
    Optional .cfg path for the upload smoke request. Defaults to a synthetic file.

.PARAMETER DatFile
    Optional .dat path; must accompany CfgFile.

.PARAMETER NoLoad
    Skip the smoke request — only sample idle uvicorn (useful to measure
    cold-start cost of imports + model load on its own).

.EXAMPLE
    ./profiling/profile_request.ps1
    ./profiling/profile_request.ps1 -DurationSec 60
    ./profiling/profile_request.ps1 -CfgFile path/to/foo.cfg -DatFile path/to/foo.dat
#>

[CmdletBinding()]
param(
    [int]$DurationSec = 30,
    [int]$Port = 8901,
    [string]$CfgFile,
    [string]$DatFile,
    [switch]$NoLoad
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

# --- Locate py-spy (binary, not Python module) ---
$pySpy = Get-Command py-spy -ErrorAction SilentlyContinue
if (-not $pySpy) {
    $userScripts = Join-Path $env:LOCALAPPDATA "Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\py-spy.exe"
    if (Test-Path $userScripts) {
        $pySpy = $userScripts
    } else {
        # Fallback: derive from current Python's user scripts dir
        $userScriptsDir = python -c "import sysconfig; print(sysconfig.get_paths()['scripts'])" 2>$null
        $candidate = Join-Path $userScriptsDir "py-spy.exe"
        if (Test-Path $candidate) {
            $pySpy = $candidate
        }
    }
    if (-not $pySpy) {
        Write-Error "py-spy not found. Install with: python -m pip install --user py-spy"
        exit 1
    }
} else {
    $pySpy = $pySpy.Source
}
Write-Host "py-spy: $pySpy"

# --- Default smoke file (synthetic INRUSH) ---
if (-not $CfgFile) {
    $CfgFile = "data/synthetic/transformer/INRUSH/GI_TAMBUN_150KV_INRUSH_20260408_114808_667.cfg"
    $DatFile = "data/synthetic/transformer/INRUSH/GI_TAMBUN_150KV_INRUSH_20260408_114808_667.dat"
}
if (-not $NoLoad) {
    if (-not (Test-Path $CfgFile)) { Write-Error "CfgFile not found: $CfgFile"; exit 1 }
    if (-not (Test-Path $DatFile)) { Write-Error "DatFile not found: $DatFile"; exit 1 }
    Write-Host "Smoke file: $CfgFile"
}

# --- Output paths ---
$outDir = Join-Path $PSScriptRoot "flamegraphs"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$svg = Join-Path $outDir "profile_${stamp}.svg"

# --- Launch uvicorn ---
Write-Host "Launching uvicorn on port $Port..."
$uvicornArgs = @(
    "-m", "uvicorn", "webapp.api.main:app",
    "--host", "127.0.0.1", "--port", "$Port",
    "--workers", "1", "--log-level", "warning"
)
$uvicorn = Start-Process -FilePath "python" -ArgumentList $uvicornArgs -PassThru -WindowStyle Hidden -RedirectStandardError "$outDir\uvicorn_stderr.log"

try {
    # Wait for /api/health up to 30 s
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) { throw "uvicorn did not become ready in 30 s" }
    Write-Host "Server ready. Starting py-spy sample for ${DurationSec}s..."

    # --- Start py-spy in background ---
    $pySpyArgs = @("record", "-o", $svg, "-d", "$DurationSec", "-f", "flamegraph", "-p", "$($uvicorn.Id)", "--idle")
    $spy = Start-Process -FilePath $pySpy -ArgumentList $pySpyArgs -PassThru -WindowStyle Hidden

    # --- Fire the smoke request(s) while py-spy samples ---
    if (-not $NoLoad) {
        # Repeat upload to get multiple samples of the hot path
        $iterations = [Math]::Max(1, [Math]::Floor($DurationSec / 5))
        Write-Host "Firing $iterations upload(s)..."
        for ($i = 0; $i -lt $iterations; $i++) {
            try {
                $cfgFs = [System.IO.File]::OpenRead((Resolve-Path $CfgFile))
                $datFs = [System.IO.File]::OpenRead((Resolve-Path $DatFile))
                # FastAPI accepts multipart; use curl.exe which ships with Windows 10+
                $r = & curl.exe -s -X POST "http://127.0.0.1:$Port/api/upload" `
                    -F "cfg_file=@$(Resolve-Path $CfgFile)" `
                    -F "dat_file=@$(Resolve-Path $DatFile)"
                $cfgFs.Close(); $datFs.Close()
                $obj = $r | ConvertFrom-Json -ErrorAction SilentlyContinue
                if ($obj.analysis_id) {
                    # Also exercise the AI fault analysis endpoint
                    & curl.exe -s -X POST "http://127.0.0.1:$Port/api/analyze/21/ai-analysis" `
                        -H "Content-Type: application/json" `
                        -d "{`"analysis_id`":`"$($obj.analysis_id)`",`"fault_inception_angle_deg`":0,`"fault_duration_ms`":0,`"prefault_load_a`":0,`"impedance_at_trip_ohm`":0,`"waveform_asymmetry`":0,`"dc_offset`":0,`"ar_result`":null}" | Out-Null
                }
            } catch {
                Write-Warning "Smoke iteration $i failed: $_"
            }
            Start-Sleep -Milliseconds 250
        }
    }

    # Wait for py-spy to finish its window
    $spy.WaitForExit()
    Write-Host "Flamegraph written: $svg"
} finally {
    if ($uvicorn -and -not $uvicorn.HasExited) {
        Stop-Process -Id $uvicorn.Id -Force -ErrorAction SilentlyContinue
        Write-Host "uvicorn stopped."
    }
}
