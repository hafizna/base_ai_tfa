import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  computeDoubleEndedFL,
  fetchAnalysis,
  fetchDoubleEndedAlignEstimate,
  fetchDoubleEndedLoopSuggestion,
  fetchDoubleEndedSuggestShift,
  uploadComtrade,
  type DoubleEndedComputeResult,
  type DoubleEndedLoopSuggestion,
} from "../api/client";
import type { ComtradeData } from "../context/AnalysisContext";
import CTVTRatioCorrection from "../components/panels/CTVTRatioCorrection";
import FaultLocatorHistogram from "../components/relay/relay21de/FaultLocatorHistogram";
import Plot from "../components/plot/PlotlyChart";
import styles from "./DoubleEndedFL.module.css";

type Side = "a" | "b";

interface TerminalState {
  file: File | null;
  loading: boolean;
  error: string | null;
  analysisId: string | null;
  comtrade: ComtradeData | null;
  invertCurrent: boolean;
  invertPhaseSequence: boolean;
}

const EMPTY_TERMINAL: TerminalState = {
  file: null,
  loading: false,
  error: null,
  analysisId: null,
  comtrade: null,
  invertCurrent: false,
  invertPhaseSequence: false,
};

const LOOP_OPTIONS = [
  { value: "ZA", label: "A-N (ground)" },
  { value: "ZB", label: "B-N (ground)" },
  { value: "ZC", label: "C-N (ground)" },
  { value: "ZAB", label: "A-B (phase)" },
  { value: "ZBC", label: "B-C (phase)" },
  { value: "ZCA", label: "C-A (phase)" },
];

function fileExt(file: File) {
  return file.name.split(".").pop()?.toLowerCase() ?? "";
}

function fileStem(file: File) {
  return file.name.replace(/\.[^.]+$/, "").toLowerCase();
}

/** Single-file (.cff) or .cfg+.dat pair drop zone for one terminal — same
 * validation rules as Upload.tsx's ComtradeDropZone, reused here per side
 * since this page needs two independent COMTRADE uploads rather than one. */
function TerminalDropZone({
  label,
  onFilesReady,
}: {
  label: string;
  onFilesReady: (files: File[]) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [names, setNames] = useState<string[]>([]);

  function handleFiles(list: FileList | null) {
    if (!list) return;
    const files = Array.from(list);
    setNames(files.map((f) => f.name));
    onFilesReady(files);
  }

  return (
    <div
      className={`${styles.dropzone} ${over ? styles.dropzoneOver : ""} ${names.length ? styles.dropzoneFilled : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        handleFiles(e.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".cff,.CFF,.cfg,.CFG,.dat,.DAT"
        multiple
        style={{ display: "none" }}
        onChange={(e) => handleFiles(e.target.files)}
      />
      <span className={styles.dropLabel}>{label}</span>
      {names.length ? (
        <span className={styles.fileName}>{names.join(" + ")}</span>
      ) : (
        <span className={styles.dropHint}>Click or drag .cff, or .cfg + .dat</span>
      )}
    </div>
  );
}

export default function DoubleEndedFL() {
  const navigate = useNavigate();

  const [terminalA, setTerminalA] = useState<TerminalState>(EMPTY_TERMINAL);
  const [terminalB, setTerminalB] = useState<TerminalState>(EMPTY_TERMINAL);

  const [loop, setLoop] = useState("ZA");
  const [loopTouchedManually, setLoopTouchedManually] = useState(false);
  const [loopSuggestion, setLoopSuggestion] = useState<DoubleEndedLoopSuggestion | null>(null);
  const [lineLenKm, setLineLenKm] = useState<string>("");
  const [r1, setR1] = useState<string>("0.05");
  const [x1, setX1] = useState<string>("0.4");

  const [manualShiftMs, setManualShiftMs] = useState<number>(0);
  const [estimateNote, setEstimateNote] = useState<string | null>(null);
  const [estimateLoading, setEstimateLoading] = useState(false);
  const [detectedInceptionAS, setDetectedInceptionAS] = useState<number | null>(null);
  const [syncOverlay, setSyncOverlay] = useState<{
    phase: string;
    channelA: string;
    channelB: string;
    lineA: string | null;
    lineB: string | null;
  } | null>(null);
  const [shiftSearchLoading, setShiftSearchLoading] = useState(false);
  const [shiftSearchNote, setShiftSearchNote] = useState<string | null>(null);

  const [computing, setComputing] = useState(false);
  const [computeError, setComputeError] = useState<string | null>(null);
  const [result, setResult] = useState<DoubleEndedComputeResult | null>(null);

  const bothUploaded = Boolean(terminalA.comtrade && terminalB.comtrade);

  async function handleTerminalFiles(side: Side, files: File[]) {
    const setState = side === "a" ? setTerminalA : setTerminalB;
    const cff = files.find((f) => fileExt(f) === "cff") ?? null;
    const cfg = files.find((f) => fileExt(f) === "cfg") ?? null;
    const dat = files.find((f) => fileExt(f) === "dat") ?? null;

    if (!cff && !(cfg && dat)) {
      setState((prev) => ({ ...prev, error: "Select one .cff file, or a matching .cfg + .dat pair." }));
      return;
    }
    if (cfg && dat && fileStem(cfg) !== fileStem(dat)) {
      setState((prev) => ({ ...prev, error: "The .cfg and .dat filenames do not match." }));
      return;
    }

    setState((prev) => ({ ...prev, loading: true, error: null }));
    setResult(null);
    try {
      const uploadFiles = cff ? [cff] : [cfg!, dat!];
      const uploaded = await uploadComtrade(uploadFiles);
      const comtrade = await fetchAnalysis(uploaded.analysis_id);
      setState((prev) => ({
        ...prev,
        loading: false,
        analysisId: uploaded.analysis_id,
        comtrade,
      }));
    } catch (err: unknown) {
      const response = (err as { response?: { data?: { detail?: string } } }).response;
      setState((prev) => ({
        ...prev,
        loading: false,
        error: response?.data?.detail ?? "Upload failed — check the file pair and try again.",
      }));
    }
  }

  // Absolute trigger timestamps come straight from each record's own CFG and
  // are frequently wrong or un-synced between two independently-owned DFRs
  // (wrong clock, wrong timezone, or a date-format bug in the source file) —
  // the align-estimate endpoint already says so in its own docstring. A
  // genuine two-terminal sync offset for a transmission line is sub-second
  // (propagation delay + trigger jitter); anything beyond this is far more
  // likely a bad timestamp than real clock drift, so it's shown as
  // information only — never silently written into the shift field the
  // calculation actually uses.
  const PLAUSIBLE_SHIFT_MS = 30_000;

  async function fetchEstimate(idA: string, idB: string) {
    setEstimateLoading(true);
    setSyncOverlay(null);
    try {
      const est = await fetchDoubleEndedAlignEstimate(idA, idB);
      setDetectedInceptionAS(est.inception_time_a_s);
      if (est.sync_phase && est.sync_channel_a && est.sync_channel_b) {
        setSyncOverlay({
          phase: est.sync_phase,
          channelA: est.sync_channel_a,
          channelB: est.sync_channel_b,
          lineA: est.line_a ?? null,
          lineB: est.line_b ?? null,
        });
      }
      if (
        est.estimate_available &&
        est.estimated_shift_ms != null &&
        Math.abs(est.estimated_shift_ms) <= PLAUSIBLE_SHIFT_MS
      ) {
        setManualShiftMs(Math.round(est.estimated_shift_ms * 10) / 10);
        setEstimateNote(est.estimate_reason);
      } else if (est.estimate_available && est.estimated_shift_ms != null) {
        setEstimateNote(
          `Estimate from record timestamps is ${(est.estimated_shift_ms / 1000).toFixed(1)}s — ` +
          "too large to be a real two-terminal sync offset, so it was NOT applied. " +
          "One or both records likely have a wrong/un-synced clock. Set the shift from the cursors below."
        );
      } else {
        setEstimateNote(est.estimate_reason);
      }
    } catch {
      setEstimateNote("Could not compute a starting estimate — set the shift manually from the cursors below.");
    } finally {
      setEstimateLoading(false);
    }
  }

  // Fire the estimate fetch once both sides have finished uploading. Runs as
  // an effect (not during render) so it only ever fires as a response to the
  // analysis IDs actually changing, matching the rules of hooks.
  const estimateRequestedFor = useRef<string | null>(null);
  const analysisIdA = terminalA.analysisId;
  const analysisIdB = terminalB.analysisId;
  useEffect(() => {
    if (!analysisIdA || !analysisIdB) return;
    const key = `${analysisIdA}:${analysisIdB}`;
    if (estimateRequestedFor.current === key) return;
    estimateRequestedFor.current = key;
    void fetchEstimate(analysisIdA, analysisIdB);
  }, [analysisIdA, analysisIdB]);

  // Suggest the loop from the faulted phases each terminal's reasoning chain
  // reads (rule F7.4: only a loop that carries fault current gives a valid
  // two-ended location). The backend prefers the strong end's reading when
  // one end is weak infeed. Only ever pre-fills the field the FIRST time a
  // suggestion arrives for this pair; once the user has touched the loop
  // selector themselves, their choice is never overwritten.
  const loopSuggestionRequestedFor = useRef<string | null>(null);
  useEffect(() => {
    if (!analysisIdA || !analysisIdB) return;
    const key = `${analysisIdA}:${analysisIdB}`;
    if (loopSuggestionRequestedFor.current === key) return;
    loopSuggestionRequestedFor.current = key;
    fetchDoubleEndedLoopSuggestion(analysisIdA, analysisIdB)
      .then((suggestion) => {
        if (!suggestion.loop) return;
        setLoopSuggestion(suggestion);
        if (!loopTouchedManually) setLoop(suggestion.loop);
      })
      .catch(() => {
        // The suggestion is a convenience, not a requirement — leave the
        // loop selector on its default/manual value if it fails.
      });
    // loopTouchedManually intentionally omitted: this effect's identity is
    // keyed on the analysis-id pair, not on that flag — re-running it every
    // time the user touches the selector would refetch the suggestion for
    // no reason. The flag is still read fresh via closure each run.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysisIdA, analysisIdB]);

  const syncTraces = useMemo(() => {
    if (!terminalA.comtrade || !terminalB.comtrade) return null;
    // The backend names the phase current to overlay on each record's
    // disturbed line. Without it, fall back to the first IA channel, which on
    // a DFR recording two lines can belong to the healthy line.
    const named = (comtrade: ComtradeData, name: string | undefined) =>
      name ? comtrade.analog_channels.find((c) => c.name === name) : undefined;
    const iaChannel = named(terminalA.comtrade, syncOverlay?.channelA)
      ?? terminalA.comtrade.analog_channels.find((c) => c.canonical_name === "IA");
    const ibChannel = named(terminalB.comtrade, syncOverlay?.channelB)
      ?? terminalB.comtrade.analog_channels.find((c) => c.canonical_name === "IA");
    if (!iaChannel || !ibChannel) return null;
    const tA = terminalA.comtrade.time;
    // Apply the current manual shift to B's time axis for display: positive
    // shift means "B is late relative to A", so subtract it from B's own
    // time values to slide B's trace earlier onto A's axis — matching the
    // backend's shift_s = -manual_shift_ms/1000 convention exactly.
    const tB = terminalB.comtrade.time.map((t) => t - manualShiftMs / 1000);
    const label = iaChannel.canonical_name || "IA";
    return { tA, iaSamples: iaChannel.samples, tB, ibSamples: ibChannel.samples, label };
  }, [terminalA.comtrade, terminalB.comtrade, manualShiftMs, syncOverlay]);

  async function handleFindOptimalShift() {
    if (!terminalA.analysisId || !terminalB.analysisId) return;
    const lineLen = parseFloat(lineLenKm);
    const r1v = parseFloat(r1);
    const x1v = parseFloat(x1);
    if (!Number.isFinite(lineLen) || lineLen <= 0 || !Number.isFinite(r1v) || !Number.isFinite(x1v)) {
      setShiftSearchNote("Fill in line length and R1/X1 (step 4) before searching for a shift.");
      return;
    }

    setShiftSearchLoading(true);
    setShiftSearchNote(null);
    try {
      const res = await fetchDoubleEndedSuggestShift({
        analysisIdA: terminalA.analysisId,
        analysisIdB: terminalB.analysisId,
        loop,
        lineLenKm: lineLen,
        r1OhmPerKm: r1v,
        x1OhmPerKm: x1v,
        invertIA: terminalA.invertCurrent,
        invertIB: terminalB.invertCurrent,
        invertPhaseSequenceA: terminalA.invertPhaseSequence,
        invertPhaseSequenceB: terminalB.invertPhaseSequence,
      });
      if (res.shift_ms != null) {
        setManualShiftMs(Math.round(res.shift_ms * 100) / 100);
      }
      setShiftSearchNote(res.reason);
    } catch (err: unknown) {
      const response = (err as { response?: { data?: { detail?: string } } }).response;
      setShiftSearchNote(response?.data?.detail ?? "Shift search failed — check inputs and try again.");
    } finally {
      setShiftSearchLoading(false);
    }
  }

  async function handleCompute() {
    if (!terminalA.analysisId || !terminalB.analysisId) return;
    const lineLen = parseFloat(lineLenKm);
    const r1v = parseFloat(r1);
    const x1v = parseFloat(x1);
    if (!Number.isFinite(lineLen) || lineLen <= 0) {
      setComputeError("Enter a valid line length (km).");
      return;
    }
    if (!Number.isFinite(r1v) || !Number.isFinite(x1v)) {
      setComputeError("Enter valid R1/X1 (ohm per km).");
      return;
    }

    setComputing(true);
    setComputeError(null);
    setResult(null);
    try {
      const res = await computeDoubleEndedFL({
        analysisIdA: terminalA.analysisId,
        analysisIdB: terminalB.analysisId,
        loop,
        lineLenKm: lineLen,
        r1OhmPerKm: r1v,
        x1OhmPerKm: x1v,
        manualShiftMs,
        invertIA: terminalA.invertCurrent,
        invertIB: terminalB.invertCurrent,
        invertPhaseSequenceA: terminalA.invertPhaseSequence,
        invertPhaseSequenceB: terminalB.invertPhaseSequence,
      });
      setResult(res);
    } catch (err: unknown) {
      const response = (err as { response?: { data?: { detail?: string } } }).response;
      setComputeError(response?.data?.detail ?? "Computation failed — check inputs and synchronization.");
    } finally {
      setComputing(false);
    }
  }

  return (
    <div className={styles.page}>
      <button className={styles.back} onClick={() => navigate("/")} type="button">
        Back to relay selection
      </button>

      <h1 className={styles.title}>Double Ended Fault Locator</h1>
      <p className={styles.subtitle}>
        Two-terminal distance calculation via Kirchhoff's voltage law — combines both line ends' voltage and
        current so fault resistance and zero-sequence compensation (K0) never enter the calculation, unlike
        single-ended distance readings.
      </p>

      <div className={styles.step}>
        <div className={styles.stepHeader}>
          <span className={styles.stepNum}>1</span>
          <span className={styles.stepTitle}>Upload both terminals</span>
          {bothUploaded && <span className={styles.stepDone}>Both loaded</span>}
        </div>
        <div className={styles.twoCol}>
          <div className={styles.terminalCard}>
            <div className={styles.terminalLabel}>Terminal A</div>
            <TerminalDropZone label=".cff / .cfg + .dat" onFilesReady={(files) => void handleTerminalFiles("a", files)} />
            {terminalA.loading && <div className={styles.estimateHint}>Parsing…</div>}
            {terminalA.error && <div className={styles.error}>{terminalA.error}</div>}
            {terminalA.comtrade && (
              <div className={styles.estimateHint} style={{ marginTop: 6 }}>
                {terminalA.comtrade.station_name || "Terminal A"} — {terminalA.comtrade.total_samples} samples
              </div>
            )}
          </div>
          <div className={styles.terminalCard}>
            <div className={styles.terminalLabel}>Terminal B</div>
            <TerminalDropZone label=".cff / .cfg + .dat" onFilesReady={(files) => void handleTerminalFiles("b", files)} />
            {terminalB.loading && <div className={styles.estimateHint}>Parsing…</div>}
            {terminalB.error && <div className={styles.error}>{terminalB.error}</div>}
            {terminalB.comtrade && (
              <div className={styles.estimateHint} style={{ marginTop: 6 }}>
                {terminalB.comtrade.station_name || "Terminal B"} — {terminalB.comtrade.total_samples} samples
              </div>
            )}
          </div>
        </div>
      </div>

      {bothUploaded && (
        <div className={styles.step}>
          <div className={styles.stepHeader}>
            <span className={styles.stepNum}>2</span>
            <span className={styles.stepTitle}>CT / PT ratio and phase sequence per terminal</span>
          </div>
          <div className={styles.twoCol}>
            <div className={styles.terminalCard}>
              <div className={styles.terminalLabel}>Terminal A</div>
              {terminalA.comtrade && terminalA.analysisId && (
                <CTVTRatioCorrection
                  analysisId={terminalA.analysisId}
                  comtrade={terminalA.comtrade}
                  onUpdate={(updated) => setTerminalA((prev) => ({ ...prev, comtrade: updated }))}
                />
              )}
              <label className={styles.toggleRow}>
                <input
                  type="checkbox"
                  checked={terminalA.invertCurrent}
                  onChange={(e) => setTerminalA((prev) => ({ ...prev, invertCurrent: e.target.checked }))}
                />
                Invert current polarity (CT direction)
              </label>
              <label className={styles.toggleRow}>
                <input
                  type="checkbox"
                  checked={terminalA.invertPhaseSequence}
                  onChange={(e) => setTerminalA((prev) => ({ ...prev, invertPhaseSequence: e.target.checked }))}
                />
                Invert phase sequence (B/C swapped)
              </label>
            </div>
            <div className={styles.terminalCard}>
              <div className={styles.terminalLabel}>Terminal B</div>
              {terminalB.comtrade && terminalB.analysisId && (
                <CTVTRatioCorrection
                  analysisId={terminalB.analysisId}
                  comtrade={terminalB.comtrade}
                  onUpdate={(updated) => setTerminalB((prev) => ({ ...prev, comtrade: updated }))}
                />
              )}
              <label className={styles.toggleRow}>
                <input
                  type="checkbox"
                  checked={terminalB.invertCurrent}
                  onChange={(e) => setTerminalB((prev) => ({ ...prev, invertCurrent: e.target.checked }))}
                />
                Invert current polarity (CT direction)
              </label>
              <label className={styles.toggleRow}>
                <input
                  type="checkbox"
                  checked={terminalB.invertPhaseSequence}
                  onChange={(e) => setTerminalB((prev) => ({ ...prev, invertPhaseSequence: e.target.checked }))}
                />
                Invert phase sequence (B/C swapped)
              </label>
            </div>
          </div>
        </div>
      )}

      {bothUploaded && (
        <div className={styles.step}>
          <div className={styles.stepHeader}>
            <span className={styles.stepNum}>3</span>
            <span className={styles.stepTitle}>Synchronize terminal B against terminal A</span>
          </div>
          <p className={styles.estimateHint}>
            The two records were triggered independently — align terminal B's fault inception onto terminal A's by
            adjusting the shift below until both current traces step at the same time. This shift is the
            authoritative sync value used by the calculation; it is never guessed automatically.
          </p>
          {syncOverlay && (
            <p className={styles.estimateHint}>
              Overlay: phase {syncOverlay.phase} current — the phase whose fault step is clearest at both ends — on{" "}
              {syncOverlay.lineA ? `line ${syncOverlay.lineA}` : "terminal A"} ({syncOverlay.channelA}) and{" "}
              {syncOverlay.lineB ? `line ${syncOverlay.lineB}` : "terminal B"} ({syncOverlay.channelB}).
            </p>
          )}
          {syncTraces && (
            <div className={styles.syncPlot}>
              <Plot
                data={[
                  {
                    x: syncTraces.tA,
                    y: syncTraces.iaSamples,
                    type: "scatter",
                    mode: "lines",
                    name: `A — ${syncTraces.label}`,
                    line: { color: "#ef4444", width: 1.2 },
                  },
                  {
                    x: syncTraces.tB,
                    y: syncTraces.ibSamples,
                    type: "scatter",
                    mode: "lines",
                    name: `B — ${syncTraces.label} (shifted)`,
                    line: { color: "#3b82f6", width: 1.2 },
                  },
                ]}
                layout={{
                  autosize: true,
                  height: 280,
                  margin: { l: 50, r: 20, t: 20, b: 40 },
                  xaxis: {
                    title: { text: "Time (s, terminal A reference)" },
                    gridcolor: "#e2e8f0",
                    range: detectedInceptionAS == null
                      ? undefined
                      : [detectedInceptionAS - 0.08, detectedInceptionAS + 0.16],
                    rangeslider: { visible: true },
                  },
                  yaxis: { title: { text: `${syncTraces.label} (A, primary)` }, gridcolor: "#e2e8f0" },
                  legend: { orientation: "h" },
                  paper_bgcolor: "#ffffff",
                  plot_bgcolor: "#ffffff",
                }}
                config={{ responsive: true, displaylogo: false }}
                style={{ width: "100%" }}
                useResizeHandler
              />
            </div>
          )}
          <div className={styles.syncReadout}>
            <div className={styles.field} style={{ maxWidth: 200 }}>
              <label htmlFor="shift-ms">Shift record B by (ms)</label>
              <input
                id="shift-ms"
                type="number"
                step="0.1"
                value={manualShiftMs}
                onChange={(e) => setManualShiftMs(parseFloat(e.target.value) || 0)}
              />
            </div>
            <span className={styles.shiftBadge}>{manualShiftMs.toFixed(1)} ms</span>
            {estimateLoading && <span className={styles.estimateHint}>Estimating starting offset…</span>}
            {estimateNote && !estimateLoading && <span className={styles.estimateHint}>{estimateNote}</span>}
          </div>
          <div className={styles.syncReadout} style={{ marginTop: 8 }}>
            <button
              type="button"
              className={styles.buttonSecondary}
              onClick={() => void handleFindOptimalShift()}
              disabled={shiftSearchLoading || !lineLenKm}
            >
              {shiftSearchLoading ? "Searching…" : "Find shift with lowest residual"}
            </button>
            {shiftSearchNote && <span className={styles.estimateHint}>{shiftSearchNote}</span>}
          </div>
          <p className={styles.estimateHint} style={{ marginTop: 4 }}>
            This searches for the shift that makes both terminals' equations most self-consistent (uses the
            line length and R1/X1 from step 4 below, and the loop selected there) — grounded in the same
            physics the final result relies on, unlike the coarse estimate above. Still only a suggestion:
            confirm it against the waveform overlay before trusting it.
          </p>
        </div>
      )}

      {bothUploaded && (
        <div className={styles.step}>
          <div className={styles.stepHeader}>
            <span className={styles.stepNum}>4</span>
            <span className={styles.stepTitle}>Line parameters and loop</span>
          </div>
          <div className={styles.fieldRow}>
            <div className={styles.field}>
              <label htmlFor="loop-select">Loop</label>
              <select
                id="loop-select"
                value={loop}
                onChange={(e) => {
                  setLoopTouchedManually(true);
                  setLoop(e.target.value);
                }}
              >
                {LOOP_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
              {loopSuggestion && (
                <span className={styles.estimateHint}>
                  {loopSuggestion.loop === loop
                    ? `Auto-detected: ${loopSuggestion.reason}`
                    : `Suggested loop ${loopSuggestion.loop}: ${loopSuggestion.reason} You selected a different loop.`}
                  {loopSuggestion.agree === false ? " The terminals disagree — check both records before trusting the loop." : ""}
                </span>
              )}
              {loopSuggestion && loopSuggestion.weak_infeed_terminals.length > 0 && (
                <span className={styles.estimateHint}>
                  {`Terminal ${loopSuggestion.weak_infeed_terminals.join(" and ")} is a weak-infeed end (little fault current), `}
                  {loopSuggestion.source_terminal && !loopSuggestion.weak_infeed_terminals.includes(loopSuggestion.source_terminal)
                    ? `so the loop comes from terminal ${loopSuggestion.source_terminal}'s reading.`
                    : "so its faulted phases are read from the voltage sag alone."}
                </span>
              )}
            </div>
            <div className={styles.field}>
              <label htmlFor="line-len">Line length (km)</label>
              <input id="line-len" type="number" step="0.01" value={lineLenKm} onChange={(e) => setLineLenKm(e.target.value)} />
            </div>
            <div className={styles.field}>
              <label htmlFor="r1">R1 (ohm/km)</label>
              <input id="r1" type="number" step="0.001" value={r1} onChange={(e) => setR1(e.target.value)} />
            </div>
            <div className={styles.field}>
              <label htmlFor="x1">X1 (ohm/km)</label>
              <input id="x1" type="number" step="0.001" value={x1} onChange={(e) => setX1(e.target.value)} />
            </div>
          </div>
          <div style={{ marginTop: 16 }}>
            <button className={styles.button} onClick={() => void handleCompute()} disabled={computing} type="button">
              {computing ? "Computing…" : "Run Fault Locator"}
            </button>
          </div>
          {computeError && <div className={styles.error}>{computeError}</div>}
        </div>
      )}

      {result && (
        <div className={styles.resultCard}>
          {result.loop_carries_fault === false && (
            <div className={styles.unreliableBanner}>
              <strong>This is not a valid fault location.</strong> The fault draws almost no current in loop{" "}
              {result.loop}, so the two-ended equations hold for any distance — a small residual here means
              nothing. Pick the loop of the faulted phases in step 4 and re-run.
            </div>
          )}
          {result.loop_carries_fault !== false &&
            (Math.abs(result.m_residual_imag) > 0.15 || result.kvl_residual > 0.15) && (
            <div className={styles.unreliableBanner}>
              <strong>This result is not reliable yet.</strong> The two terminals' equations never found a
              consistent multi-window solution (RMS Im(m)={result.m_residual_imag.toFixed(3)}, KVL residual={" "}
              {result.kvl_residual.toFixed(3)}) — the numbers below will keep
              changing unpredictably if you adjust the line length, because the underlying disagreement between
              terminal A and terminal B is not a line-length problem. Go back to step 3 and drag the sync shift
              until terminal B's current step visually lines up with terminal A's, then re-run. See the warning
              below for what to check next if that doesn't resolve it.
            </div>
          )}
          <div className={styles.stepTitle}>Double-ended fault location result</div>
          <div className={styles.resultGrid}>
            <div className={styles.resultStat}>
              <div className={styles.resultStatLabel}>Distance from A</div>
              <div className={styles.resultStatValue}>{result.distance_km.toFixed(2)} km</div>
            </div>
            <div className={styles.resultStat}>
              <div className={styles.resultStatLabel}>% of line</div>
              <div className={styles.resultStatValue}>{result.distance_pct.toFixed(1)}%</div>
            </div>
            <div className={styles.resultStat}>
              <div className={styles.resultStatLabel}>Fault current</div>
              <div className={styles.resultStatValue}>{(result.fault_current_a / 1000).toFixed(2)} kA</div>
            </div>
            <div className={styles.resultStat}>
              <div className={styles.resultStatLabel}>Loop</div>
              <div className={styles.resultStatValue}>{result.loop}</div>
            </div>
            <div className={styles.resultStat}>
              <div className={styles.resultStatLabel}>Calculation basis</div>
              <div className={styles.resultStatValue}>
                {result.calculation_basis === "negative_sequence" ? "Negative sequence" : "Phase loop"}
              </div>
            </div>
            <div className={styles.resultStat}>
              <div className={styles.resultStatLabel}>Selected windows</div>
              <div className={styles.resultStatValue}>{result.selected_window_count}</div>
            </div>
            <div className={styles.resultStat}>
              <div className={styles.resultStatLabel}>Distance spread</div>
              <div className={styles.resultStatValue}>{result.distance_spread_km.toFixed(2)} km</div>
            </div>
            <div className={styles.resultStat}>
              <div className={styles.resultStatLabel}>KVL residual</div>
              <div className={styles.resultStatValue}>{result.kvl_residual.toFixed(3)}</div>
            </div>
          </div>
          {result.warnings.length > 0 && (
            <ul className={styles.warningList}>
              {result.warnings.map((w, i) => <li key={i}>{w}</li>)}
            </ul>
          )}
          {result.distance_histogram_km.length > 0 && (
            <FaultLocatorHistogram
              histogram={result.distance_histogram_km}
              lineLenKm={parseFloat(lineLenKm)}
              twoEnded={{ distanceKm: result.distance_km, faultCurrentA: result.fault_current_a, loop: result.loop }}
              singleEndedA={result.single_ended_a}
              singleEndedB={result.single_ended_b}
            />
          )}
        </div>
      )}
    </div>
  );
}
