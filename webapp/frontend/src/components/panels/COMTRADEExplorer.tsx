import { useEffect, useMemo, useState } from "react";

import type { AnalogChannel, ComtradeData } from "../../context/AnalysisContext";
import Plot from "../plot/PlotlyChart";
import styles from "./Panel.module.css";

interface Props {
  comtrade: ComtradeData;
}

const MAX_PLOT_POINTS = 5000;
const PLOT_LEFT_MARGIN = 120; // shared margin keeps analog & digital x-axes aligned

/** Find intervals where digital channel is active (state=1). */
function findActiveIntervals(timeMs: number[], samples: number[]): [number, number][] {
  const intervals: [number, number][] = [];
  let inActive = false;
  let startT = 0;
  const len = Math.min(timeMs.length, samples.length);
  for (let i = 0; i < len; i++) {
    if (samples[i] === 1 && !inActive) { startT = timeMs[i]; inActive = true; }
    else if (samples[i] === 0 && inActive) { intervals.push([startT, timeMs[i]]); inActive = false; }
  }
  if (inActive && len > 0) intervals.push([startT, timeMs[len - 1]]);
  return intervals;
}

/** Build Gantt-style bar traces for digital channels. Gridlines handle the row lines. */
function buildDigitalBarTraces(
  channels: { name: string; samples: number[] }[],
  timeMs: number[],
): Plotly.Data[] {
  const traces: Plotly.Data[] = [];
  const n = channels.length;

  channels.forEach((ch, idx) => {
    const rowY = n - 1 - idx;
    const intervals = findActiveIntervals(timeMs, ch.samples);

    intervals.forEach(([t0, t1]) => {
      traces.push({
        x: [t0, t0, t1, t1, t0],
        y: [rowY - 0.38, rowY + 0.38, rowY + 0.38, rowY - 0.38, rowY - 0.38],
        type: "scatter",
        mode: "lines",
        fill: "toself",
        fillcolor: "rgba(251, 146, 60, 0.88)",
        line: { width: 0 },
        showlegend: false,
        name: ch.name,
        hovertemplate: `<b>${ch.name}</b><br>${t0.toFixed(1)} → ${t1.toFixed(1)} ms<extra></extra>`,
      } as Plotly.Data);
    });
  });

  return traces;
}

function buildSampledSeries(time: number[], samples: number[], maxPoints: number) {
  if (!time.length || !samples.length) {
    return { x: [], y: [] };
  }

  if (time.length <= maxPoints) {
    return { x: time, y: samples };
  }

  const step = Math.ceil(time.length / maxPoints);
  const x: number[] = [];
  const y: number[] = [];

  for (let i = 0; i < time.length; i += step) {
    x.push(time[i]);
    y.push(samples[i]);
  }

  if (x[x.length - 1] !== time[time.length - 1]) {
    x.push(time[time.length - 1]);
    y.push(samples[samples.length - 1]);
  }

  return { x, y };
}

function inferChannelColor(name: string, measurement: string) {
  const upper = name.toUpperCase();

  if (measurement === "voltage") {
    if (upper.includes("VA") || upper.includes("AN") || upper.endsWith("A")) return "#ef4444";
    if (upper.includes("VB") || upper.includes("BN") || upper.endsWith("B")) return "#3b82f6";
    if (upper.includes("VC") || upper.includes("CN") || upper.endsWith("C")) return "#22c55e";
    return "#8b5cf6";
  }

  if (upper.includes("IA") || upper.endsWith("A")) return "#f97316";
  if (upper.includes("IB") || upper.endsWith("B")) return "#06b6d4";
  if (upper.includes("IC") || upper.endsWith("C")) return "#a855f7";
  if (upper.includes("IN") || upper.includes("I0") || upper.includes("IE")) return "#eab308";
  return "#64748b";
}

function channelLabel(name: string, canonical: string) {
  return canonical || name;
}

function parseRelayoutRange(event: Record<string, unknown>) {
  const start = event["xaxis.range[0]"];
  const end = event["xaxis.range[1]"];

  if (typeof start === "number" && typeof end === "number") {
    return [start, end] as [number, number];
  }

  if (event["xaxis.autorange"]) {
    return null;
  }

  return undefined;
}

/** Extract dot-separated prefix from a channel name, e.g. "HVS.Ia" → "HVS". */
function extractPrefix(name: string): string {
  const idx = name.indexOf(".");
  return idx > 0 ? name.slice(0, idx) : "";
}

/**
 * Extract Siemens 7UT-style trailing winding-side suffix.
 *   "Current IA.a" → "a"  (Side 1, conventionally HV)
 *   "Current IB.b" → "b"  (Side 2, conventionally LV)
 *   "Current IC.c" → "c"  (Side 3, conventionally TV)
 * Returns "" if no single-letter trailing suffix is present.
 */
function extractWindingSuffix(name: string): string {
  const m = name.match(/\.([abc])\s*$/i);
  return m ? m[1].toLowerCase() : "";
}

/** Map Siemens-style suffix to canonical short side code used elsewhere in the explorer. */
const SUFFIX_SIDE_SHORT: Record<string, string> = { a: "HVS", b: "LVS", c: "TVS" };
const SUFFIX_SIDE_LABEL: Record<string, string> = {
  a: "Side 1 (HV)",
  b: "Side 2 (LV)",
  c: "Side 3 (TV)",
};

/** Human-readable label for a winding prefix. */
function windingLabel(prefix: string): string {
  const upper = prefix.toUpperCase();
  if (upper.startsWith("HVS") || upper === "HV") return `${prefix} (HV Side)`;
  if (upper.startsWith("LVS") || upper === "LV") return `${prefix} (LV Side)`;
  if (upper.startsWith("MVS") || upper === "MV") return `${prefix} (MV Side)`;
  return prefix;
}

interface ChannelGroup {
  label: string;
  shortLabel: string;
  channels: AnalogChannel[];
}

/**
 * Detect logical channel groups — either transformer winding sides (HVS/MVS/LVS prefix)
 * or external DFR bays (first repeated canonical name).
 * Returns null if no grouping applies.
 */
function detectChannelGroups(
  channels: AnalogChannel[]
): ChannelGroup[] | null {
  if (channels.length === 0) return null;

  // --- Winding-suffix detection (Siemens 7UT-style ".a" / ".b" / ".c") ---
  // Checked before the prefix path because names like "Current IA.a" / "Current IA.b"
  // share the same dot-prefix ("Current IA") and would otherwise be grouped per phase
  // instead of per winding side.
  const suffixes = channels.map((ch) => extractWindingSuffix(ch.name));
  const uniqueSuffixes = [...new Set(suffixes.filter((s) => s))];
  const withSuffix = suffixes.filter((s) => s).length;
  if (
    uniqueSuffixes.length >= 2 &&
    uniqueSuffixes.length <= 3 &&
    withSuffix >= channels.length * 0.7
  ) {
    return uniqueSuffixes.sort().map((sfx) => ({
      label: SUFFIX_SIDE_LABEL[sfx] ?? `Side ${sfx.toUpperCase()}`,
      shortLabel: SUFFIX_SIDE_SHORT[sfx] ?? sfx.toUpperCase(),
      channels: channels.filter((ch) => extractWindingSuffix(ch.name) === sfx),
    }));
  }

  // --- Winding-prefix detection ---
  const prefixes = channels.map((ch) => extractPrefix(ch.name));
  const uniquePrefixes = [...new Set(prefixes.filter((p) => p))];
  const withPrefix = prefixes.filter((p) => p).length;

  if (
    uniquePrefixes.length >= 2 &&
    uniquePrefixes.length <= 4 &&
    withPrefix >= channels.length * 0.7
  ) {
    return uniquePrefixes.map((prefix) => ({
      label: windingLabel(prefix),
      shortLabel: prefix,
      channels: channels.filter((ch) => extractPrefix(ch.name) === prefix),
    }));
  }

  // --- Legacy bay-boundary detection (first repeated canonical name) ---
  const seen = new Set<string>();
  let boundary: number | null = null;
  for (let i = 0; i < channels.length; i++) {
    const key = channels[i].canonical_name || channels[i].name;
    if (seen.has(key)) { boundary = i; break; }
    seen.add(key);
  }
  if (boundary === null) return null;

  const bay1 = channels.slice(0, boundary);
  const bay2 = channels.slice(boundary);
  const labelFor = (chs: typeof channels) => {
    if (chs.length === 0) return "Bay";
    const parts = chs[0].name.trim().split(/\s+/);
    return parts.length > 1 ? parts.slice(-2).join(" ") : chs[0].name;
  };
  return [
    { label: labelFor(bay1), shortLabel: labelFor(bay1), channels: bay1 },
    { label: labelFor(bay2), shortLabel: labelFor(bay2), channels: bay2 },
  ];
}

/** Peak absolute value across all samples. */
function channelPeak(samples: number[]): number {
  let max = 0;
  for (const v of samples) { const a = Math.abs(v); if (a > max) max = a; }
  return max;
}

/**
 * Compute a symmetric y-axis range that fits all channels.
 * Returns undefined when there are no channels or all values are zero.
 */
function channelsYRange(
  channels: { samples: number[] }[],
  mode: "instantaneous" | "rms",
  cN: number,
): [number, number] | undefined {
  if (!channels.length) return undefined;
  let max = 0;
  for (const ch of channels) {
    const s = mode === "rms" ? computeRms(ch.samples, cN) : ch.samples;
    const p = channelPeak(s);
    if (p > max) max = p;
  }
  if (max === 0) return undefined;
  const pad = max * 0.12;
  return [-(max + pad), max + pad];
}

/** Sliding-window RMS over one electrical cycle. Produces the amplitude envelope. */
function computeRms(samples: number[], cycleN: number): number[] {
  const result = new Float64Array(samples.length);
  const half = Math.floor(cycleN / 2);
  let sumSq = 0;
  // Seed the first window
  const initEnd = Math.min(cycleN, samples.length);
  for (let j = 0; j < initEnd; j++) sumSq += samples[j] * samples[j];
  for (let i = 0; i < samples.length; i++) {
    const winStart = Math.max(0, i - half);
    const winEnd   = Math.min(samples.length, i + half + 1);
    // Recompute from scratch per point (simple; fast enough for ≤6k samples)
    let s = 0;
    for (let j = winStart; j < winEnd; j++) s += samples[j] * samples[j];
    result[i] = Math.sqrt(s / (winEnd - winStart));
  }
  return Array.from(result);
}

export default function COMTRADEExplorer({ comtrade }: Props) {
  const analogChannels = comtrade.analog_channels;
  const statusChannels = comtrade.status_channels;
  const sr = comtrade.sampling_rates[0]?.[0] ?? "-";
  const duration =
    comtrade.time.length > 1
      ? ((comtrade.time[comtrade.time.length - 1] - comtrade.time[0]) * 1000).toFixed(1)
      : "-";

  const sampledTimeMs = useMemo(() => comtrade.time.map((t) => t * 1000), [comtrade.time]);

  // Detect winding-side or multi-bay groups
  const channelGroups = useMemo(() => detectChannelGroups(analogChannels), [analogChannels]);
  const hasGroups = channelGroups !== null;

  // True when dot-prefix groups look like transformer windings (HVS/MVS/LVS)
  const isTransformerRecording = useMemo(
    () =>
      channelGroups !== null &&
      channelGroups.some((g) => /^(HVS?|LVS?|MVS?)$/i.test(g.shortLabel)),
    [channelGroups]
  );

  // -1 = all (default), 0..N-1 = group index
  const [selectedGroup, setSelectedGroup] = useState<number>(-1);

  const visibleAnalog = useMemo(() => {
    if (!hasGroups || selectedGroup === -1) return analogChannels;
    return channelGroups![selectedGroup]?.channels ?? analogChannels;
  }, [analogChannels, hasGroups, selectedGroup, channelGroups]);

  const defaultAnalogIds = useMemo(() => {
    // Exclude voltage from auto-selection for transformer recordings
    const voltageCodes = isTransformerRecording ? [] : ["VA", "VB", "VC"];
    const phaseIds = visibleAnalog
      .filter((ch) => {
        if (!([...voltageCodes, "IA", "IB", "IC", "IN", "I0"].includes(ch.canonical_name || ch.name))) return false;
        // For transformer recordings skip MV side — not installed in PLN 2-winding config
        if (isTransformerRecording && /^MVS?$/i.test(extractPrefix(ch.name))) return false;
        return true;
      })
      .map((ch) => ch.id);

    // Include relay-computed diff/REF channels (pu / In) so they auto-show in their subplot
    const diffIds = visibleAnalog
      .filter((ch) => {
        const u = ch.unit.toLowerCase();
        return u === "pu" || u === "in";
      })
      .filter((ch) => /87[TL]\.i/i.test(ch.name) || /64ref\.i/i.test(ch.name))
      .map((ch) => ch.id);

    return new Set([...phaseIds, ...diffIds]);
  }, [visibleAnalog]);

  // Auto-select channels that were active (any sample = 1) — same logic as SOE
  const defaultDigitalIds = useMemo(() => {
    const active = statusChannels.filter((ch) => ch.samples.some((s) => s === 1));
    // If nothing fired, fall back to first 10 so the panel isn't empty
    const base = active.length > 0 ? active : statusChannels.slice(0, 10);
    return new Set(base.map((ch) => ch.id));
  }, [statusChannels]);

  const [selectedAnalog, setSelectedAnalog] = useState<Set<string>>(
    () => (defaultAnalogIds.size > 0 ? defaultAnalogIds : new Set(analogChannels.slice(0, 6).map((ch) => ch.id)))
  );
  const [selectedDigital, setSelectedDigital] = useState<Set<string>>(defaultDigitalIds);
  const [showDigital, setShowDigital] = useState(true);
  const [sharedRange, setSharedRange] = useState<[number, number] | null>(null);
  const [normalizeToInception, setNormalizeToInception] = useState(false);
  const [displayMode, setDisplayMode] = useState<"instantaneous" | "rms">("instantaneous");
  const [digitalHoverMs, setDigitalHoverMs] = useState<number | null>(null);

  // One electrical cycle in samples — used for RMS window
  const cycleN = useMemo(() => {
    const fs = comtrade.sampling_rates[0]?.[0] ?? 1200;
    const freq = comtrade.frequency ?? 50;
    return Math.max(2, Math.round(fs / freq));
  }, [comtrade]);

  // Detect fault inception time from the first available current channel
  const inceptionTimeMs = useMemo((): number | null => {
    const ch = comtrade.analog_channels.find(
      (c) => c.measurement === "current" && ["IA", "IB", "IC"].includes(c.canonical_name || "")
    );
    if (!ch || ch.samples.length < 10) return null;
    const s = ch.samples;
    const preEnd = Math.max(4, Math.min(Math.floor(s.length / 4), 80));
    const preRms = Math.sqrt(s.slice(0, preEnd).reduce((a, v) => a + v * v, 0) / preEnd);
    let maxAbs = 0;
    for (const v of s) if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    const thr = Math.max(preRms * 2, maxAbs * 0.3, 0.05);
    for (let i = preEnd; i < s.length; i++) {
      if (Math.abs(s[i]) > thr) return comtrade.time[i] * 1000;
    }
    return null;
  }, [comtrade]);

  // Trigger offset from CFG (ms from recording start). 0 = unknown.
  const triggerOffsetMs = comtrade.trigger_time * 1000;

  // Reset zoom & mode when a new file is loaded
  useEffect(() => {
    setSharedRange(null);
    setNormalizeToInception(false);
    setDigitalHoverMs(null);
  }, [comtrade]);

  // Default zoom: ±N ms around trigger (or inception if no CFG trigger)
  const defaultRange = useMemo((): [number, number] | null => {
    const dur = (comtrade.time[comtrade.time.length - 1] ?? 0) * 1000;
    const center = triggerOffsetMs > 20 ? triggerOffsetMs : (inceptionTimeMs ?? null);
    if (center === null || dur <= 600) return null;
    return [Math.max(0, center - 300), Math.min(dur, center + 700)];
  }, [triggerOffsetMs, inceptionTimeMs, comtrade.time]);

  const displayTimeMs = useMemo(() => {
    const offset = normalizeToInception && inceptionTimeMs !== null ? inceptionTimeMs : 0;
    return sampledTimeMs.map((t) => t - offset);
  }, [sampledTimeMs, normalizeToInception, inceptionTimeMs]);

  // Trigger position on the displayed time axis (adjusted for normalization)
  const displayTriggerMs = useMemo((): number | null => {
    if (triggerOffsetMs <= 0) return null;
    const offset = normalizeToInception && inceptionTimeMs !== null ? inceptionTimeMs : 0;
    return triggerOffsetMs - offset;
  }, [triggerOffsetMs, normalizeToInception, inceptionTimeMs]);

  function switchGroup(groupIdx: number) {
    setSelectedGroup(groupIdx);
    const nextChannels =
      groupIdx === -1 ? analogChannels : channelGroups?.[groupIdx]?.channels ?? analogChannels;
    const nextIds = new Set(
      nextChannels
        .filter((ch) => ["VA", "VB", "VC", "IA", "IB", "IC", "IN", "I0"].includes(ch.canonical_name || ch.name))
        .slice(0, 8)
        .map((ch) => ch.id)
    );
    setSelectedAnalog(nextIds.size > 0 ? nextIds : new Set(nextChannels.slice(0, 6).map((ch) => ch.id)));
  }

  function toggleAnalog(id: string) {
    setSelectedAnalog((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  function toggleDigital(id: string) {
    setSelectedDigital((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  function syncRange(event: Record<string, unknown>) {
    const next = parseRelayoutRange(event);
    if (next !== undefined) {
      setSharedRange(next);
    }
  }

  /** Channels whose unit is pu or In (relay-computed diff / REF) — shown in own subplot. */
  function isDiffChannel(ch: { name: string; unit: string }): boolean {
    const u = ch.unit.toLowerCase();
    return u === "pu" || u === "in";
  }

  const selectedVoltage = analogChannels.filter(
    (ch) => ch.measurement === "voltage" && selectedAnalog.has(ch.id)
  );
  const selectedCurrent = analogChannels.filter(
    (ch) => ch.measurement === "current" && !isDiffChannel(ch) && selectedAnalog.has(ch.id)
  );
  const selectedDiff = analogChannels.filter(
    (ch) => isDiffChannel(ch) && selectedAnalog.has(ch.id)
  );
  const selectedStatus = statusChannels.filter((ch) => selectedDigital.has(ch.id));

  const xAxisLabel = normalizeToInception ? "Waktu relatif inception (ms)" : "Waktu (ms)";
  const inceptionDisplayMs = normalizeToInception ? 0 : (inceptionTimeMs ?? null);

  const voltageTraces = buildTraces(selectedVoltage, "voltage");

  function buildTraces(
    channels: typeof analogChannels,
    measurement: string,
  ): Plotly.Data[] {
    return channels.map((ch) => {
      const lbl = channelLabel(ch.name, ch.canonical_name);
      const rawSamples =
        displayMode === "rms" && measurement !== "voltage"
          ? computeRms(ch.samples, cycleN)
          : ch.samples;
      const series = buildSampledSeries(displayTimeMs, rawSamples, MAX_PLOT_POINTS);
      return {
        x: series.x,
        y: series.y,
        type: "scatter",
        mode: "lines",
        name: lbl,
        line: { color: inferChannelColor(lbl, measurement), width: 1.3 },
        hovertemplate: `${lbl}<br>%{y:.3f} ${ch.unit}<br>%{x:.2f} ms<extra></extra>`,
      } as Plotly.Data;
    });
  }

  const currentTraces = buildTraces(selectedCurrent, "current");

  const diffTraces = buildTraces(selectedDiff, "current");

  // Explicit y-ranges so they update correctly on mode/channel changes
  // while remaining locked against user drag/scroll (fixedrange: true)
  const voltageYRange = useMemo(
    () => channelsYRange(selectedVoltage, "instantaneous", cycleN),
    [selectedVoltage, cycleN]
  );
  const currentYRange = useMemo(
    () => channelsYRange(selectedCurrent, displayMode, cycleN),
    [selectedCurrent, displayMode, cycleN]
  );
  const diffYRange = useMemo(
    () => channelsYRange(selectedDiff, displayMode, cycleN),
    [selectedDiff, displayMode, cycleN]
  );

  const digitalBarTraces = useMemo(
    () => buildDigitalBarTraces(selectedStatus, displayTimeMs),
    [selectedStatus, displayTimeMs]
  );

  function digitalStateAtTime(tMs: number | null) {
    if (tMs === null || selectedStatus.length === 0 || displayTimeMs.length === 0) return null;
    let idx = 0;
    while (idx < displayTimeMs.length - 1 && displayTimeMs[idx + 1] <= tMs) idx += 1;
    const active = selectedStatus.filter((ch) => ch.samples[idx] === 1).map((ch) => ch.name);
    return {
      time: tMs,
      active,
      inactiveCount: selectedStatus.length - active.length,
    };
  }

  const digitalHoverReadout = useMemo(
    () => digitalStateAtTime(digitalHoverMs),
    [digitalHoverMs, selectedStatus, displayTimeMs]
  );

  function updateDigitalHover(event: Readonly<Plotly.PlotMouseEvent>) {
    const point = event.points?.[0];
    const xRaw = point?.x;
    const x = typeof xRaw === "number" ? xRaw : Number(xRaw);
    if (!Number.isFinite(x)) return;
    setDigitalHoverMs(x);
  }

  const inceptionShape: Plotly.Shape | null = inceptionDisplayMs !== null ? {
    type: "line",
    x0: inceptionDisplayMs, x1: inceptionDisplayMs,
    yref: "paper", y0: 0, y1: 1,
    line: { color: "#f97316", width: 1.5, dash: "dot" },
  } as Plotly.Shape : null;

  // CFG trigger marker — amber dashed vertical line
  const triggerShape: Plotly.Shape | null = displayTriggerMs !== null ? {
    type: "line",
    x0: displayTriggerMs, x1: displayTriggerMs,
    yref: "paper", y0: 0, y1: 1,
    line: { color: "#f59e0b", width: 1.5, dash: "dash" },
  } as Plotly.Shape : null;

  const digitalHoverShape: Plotly.Shape | null = digitalHoverMs !== null ? {
    type: "line",
    x0: digitalHoverMs,
    x1: digitalHoverMs,
    yref: "paper",
    y0: 0,
    y1: 1,
    line: { color: "#2563eb", width: 1.2 },
  } as Plotly.Shape : null;

  const activeRange = sharedRange ?? defaultRange;

  const analogLayout = (yTitle: string, yRange?: [number, number]): Partial<Plotly.Layout> => ({
    height: 260,
    margin: { l: PLOT_LEFT_MARGIN, r: 20, t: 16, b: 34 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    hovermode: "x unified",
    dragmode: "pan",
    uirevision: "comtrade-sync",
    legend: { orientation: "h", y: 1.12, font: { size: 10 } },
    xaxis: {
      title: { text: xAxisLabel },
      tickfont: { size: 10 },
      autorange: activeRange ? undefined : true,
      range: activeRange ?? undefined,
      zeroline: normalizeToInception,
      zerolinecolor: "#f97316",
      zerolinewidth: 1.5,
    },
    yaxis: {
      title: { text: yTitle },
      tickfont: { size: 10 },
      tickformat: "~s",
      exponentformat: "SI",
      separatethousands: true,
      // Lock y-axis so scroll/drag only affects the time axis.
      // Range is computed from data so it updates when mode or channels change.
      fixedrange: true,
      ...(yRange ? { range: yRange, autorange: false } : { autorange: true }),
    },
    shapes: [
      ...(inceptionShape && !normalizeToInception ? [inceptionShape] : []),
      ...(triggerShape ? [triggerShape] : []),
    ],
  });

  const digitalBarLayout: Partial<Plotly.Layout> = {
    height: Math.max(180, selectedStatus.length * 28 + 60),
    margin: { l: PLOT_LEFT_MARGIN, r: 20, t: 10, b: 40 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    hovermode: "closest",
    dragmode: "pan",
    showlegend: false,
    uirevision: "comtrade-sync",
    xaxis: {
      title: { text: xAxisLabel },
      tickfont: { size: 9 },
      autorange: activeRange ? undefined : true,
      range: activeRange ?? undefined,
      showgrid: true,
      gridcolor: "#e2e8f0",
      zeroline: normalizeToInception,
      zerolinecolor: "#f97316",
      zerolinewidth: 1.5,
    },
    yaxis: {
      // Truncate long labels to keep within the shared margin
      tickvals: selectedStatus.map((_, idx) => selectedStatus.length - 1 - idx),
      ticktext: selectedStatus.map((ch) => ch.name.length > 16 ? ch.name.slice(0, 15) + "…" : ch.name),
      range: [-0.6, selectedStatus.length - 0.4],
      tickfont: { size: 9 },
      showgrid: true,
      gridcolor: "#e2e8f0",
      zeroline: false,
      fixedrange: true,
    },
    shapes: [
      ...(inceptionShape && !normalizeToInception ? [inceptionShape] : []),
      ...(triggerShape ? [triggerShape] : []),
      ...(digitalHoverShape ? [digitalHoverShape] : []),
    ],
  };

  return (
    <div className={styles.waveShell}>
      <div className={styles.waveHead}>
        <div>
          <div className={styles.waveTitle}>COMTRADE Explorer</div>
          <div className={styles.waveSub}>
            Semua plot tersinkron. Garis amber (---) = trigger CFG. Garis oranye (···) = inception terdeteksi.
          </div>
        </div>
        <div className={styles.waveBadges}>
          <span className={styles.waveBadge}>Samples: {comtrade.total_samples}</span>
          <span className={styles.waveBadge}>Fs: {sr} Hz</span>
          <span className={styles.waveBadge}>Durasi: {duration} ms</span>
          {triggerOffsetMs > 0 && (
            <span className={styles.waveBadge} style={{ background: "#fffbeb", color: "#92400e", borderColor: "#fbbf24" }}>
              Pre-fault: {triggerOffsetMs.toFixed(0)} ms
            </span>
          )}
          <span className={styles.waveBadge}>Analog: {analogChannels.length}</span>
          <span className={styles.waveBadge}>Digital: {statusChannels.length}</span>
        </div>
      </div>

      <div className={styles.waveToolbar}>
        <div className={styles.waveBadges}>
          <span className={styles.waveBadge}>Station: {comtrade.station_name}</span>
          <span className={styles.waveBadge}>Device: {comtrade.rec_dev_id}</span>
          <span className={styles.waveBadge}>Revision: {comtrade.rev_year}</span>
        </div>
        <div className={styles.waveToolbarActions}>
          <button type="button" className={styles.waveGhostBtn} onClick={() => setSharedRange(null)}>
            Reset Zoom
          </button>
          <button
            type="button"
            className={`${styles.waveGhostBtn} ${styles.waveModeBtn}`}
            onClick={() => setDisplayMode((m) => m === "instantaneous" ? "rms" : "instantaneous")}
            style={displayMode === "rms" ? { background: "#eff6ff", borderColor: "#3b82f6", color: "#1d4ed8" } : undefined}
          >
            {displayMode === "instantaneous" ? "Instantaneous Chart" : "RMS Chart"}
          </button>
          {inceptionTimeMs !== null && (
            <button
              type="button"
              className={styles.waveGhostBtn}
              onClick={() => { setNormalizeToInception((v) => !v); setSharedRange(null); }}
              style={normalizeToInception ? { background: "#fff7ed", borderColor: "#f97316", color: "#c2410c" } : undefined}
            >
              {normalizeToInception ? "t=0 @ inception ✓" : "Normalkan t=0"}
            </button>
          )}
          <label className={styles.waveDigitalToggle}>
            <input type="checkbox" checked={showDigital} onChange={(e) => setShowDigital(e.target.checked)} />
            Digital
          </label>
        </div>
      </div>

      <div className={styles.waveGrid}>
        <div className={styles.waveChannelList}>
          {hasGroups && (
            <div style={{ display: "flex", gap: 4, marginBottom: 8, flexWrap: "wrap" }}>
              <div style={{ fontSize: "0.65rem", color: "#94a3b8", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", width: "100%", marginBottom: 4 }}>
                {channelGroups!.some((g) => /HVS|LVS|MVS|HV|LV|MV/i.test(g.shortLabel)) ? "Winding Side" : "Bay Filter"}
              </div>
              {[
                ...channelGroups!.map((g, i) => ({ val: i, label: g.shortLabel })),
                { val: -1, label: "Semua" },
              ].map(({ val, label }) => (
                <button
                  key={val}
                  type="button"
                  onClick={() => switchGroup(val)}
                  style={{
                    padding: "3px 10px",
                    fontSize: "0.72rem",
                    fontWeight: 600,
                    borderRadius: 99,
                    border: "1.5px solid",
                    borderColor: selectedGroup === val ? "#3b82f6" : "#cbd5e1",
                    background: selectedGroup === val ? "#eff6ff" : "#fff",
                    color: selectedGroup === val ? "#1d4ed8" : "#64748b",
                    cursor: "pointer",
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
          )}
          <div className={styles.waveGroupTitle}>Analog Channels</div>
          {visibleAnalog.map((ch) => {
            const label = channelLabel(ch.name, ch.canonical_name);
            const prefix = extractPrefix(ch.name);
            const windingSide = prefix ? windingLabel(prefix) : null;
            const peak = channelPeak(ch.samples);
            const peakStr = peak > 0
              ? `peak ${peak < 1 ? peak.toFixed(3) : peak < 100 ? peak.toFixed(2) : peak.toFixed(1)} ${ch.unit}`
              : null;
            return (
              <button
                key={ch.id}
                type="button"
                className={`${styles.waveChannel} ${selectedAnalog.has(ch.id) ? styles.waveChannelActive : ""}`}
                onClick={() => toggleAnalog(ch.id)}
              >
                <span
                  className={styles.waveDot}
                  style={{ background: inferChannelColor(label, ch.measurement) }}
                />
                <span className={styles.waveMeta}>
                  <span className={styles.waveName}>{label}</span>
                  <span className={styles.waveStats}>
                    {windingSide ?? (ch.phase ? `Ph ${ch.phase}` : ch.measurement)}
                    {peakStr && <> · {peakStr}</>}
                  </span>
                </span>
              </button>
            );
          })}

          {statusChannels.length > 0 && (
            <>
              <div className={styles.waveGroupTitle}>Digital Channels</div>
              {statusChannels.map((ch) => (
                <button
                  key={ch.id}
                  type="button"
                  className={`${styles.waveChannel} ${selectedDigital.has(ch.id) ? styles.waveChannelActive : ""}`}
                  onClick={() => toggleDigital(ch.id)}
                >
                  <span className={styles.waveDot} style={{ background: "#60a5fa" }} />
                  <span className={styles.waveMeta}>
                    <span className={styles.waveName}>{ch.name}</span>
                    <span className={styles.waveStats}>{ch.samples.length} samples</span>
                  </span>
                </button>
              ))}
            </>
          )}
        </div>

        <div className={styles.wavePlotStack}>
          {voltageTraces.length > 0 && (
            <div className={styles.waveSubplot}>
              <div className={styles.waveSubplotTitle}>Tegangan</div>
              <Plot
                data={voltageTraces}
                layout={analogLayout(`Tegangan (${selectedVoltage[0]?.unit || "kV"})`, voltageYRange)}
                config={{ displayModeBar: false, responsive: true, scrollZoom: true, doubleClick: "reset" }}
                style={{ width: "100%" }}
                onRelayout={(event) => syncRange(event as Record<string, unknown>)}
              />
            </div>
          )}

          {/* Per-group current subplots when viewing all channels of a multi-winding recording */}
          {hasGroups && selectedGroup === -1 ? (
            channelGroups!.map((group, idx) => {
              const groupCurrents = group.channels.filter(
                (ch) => ch.measurement === "current" && !isDiffChannel(ch) && selectedAnalog.has(ch.id)
              );
              if (groupCurrents.length === 0) return null;
              const traces = buildTraces(groupCurrents, "current");
              const yRange = channelsYRange(groupCurrents, displayMode, cycleN);
              return (
                <div key={idx} className={styles.waveSubplot}>
                  <div className={styles.waveSubplotTitle}>
                    Arus — <span style={{ color: "#64748b", fontWeight: 400 }}>{group.label}</span>
                  </div>
                  <Plot
                    data={traces}
                    layout={analogLayout(`Arus (${groupCurrents[0]?.unit || "A"})`, yRange)}
                    config={{ displayModeBar: false, responsive: true, scrollZoom: true, doubleClick: "reset" }}
                    style={{ width: "100%" }}
                    onRelayout={(event) => syncRange(event as Record<string, unknown>)}
                  />
                </div>
              );
            })
          ) : (
            currentTraces.length > 0 && (
              <div className={styles.waveSubplot}>
                <div className={styles.waveSubplotTitle}>Arus</div>
                <Plot
                  data={currentTraces}
                  layout={analogLayout(`Arus (${selectedCurrent[0]?.unit || "A"})`, currentYRange)}
                  config={{ displayModeBar: false, responsive: true, scrollZoom: true, doubleClick: "reset" }}
                  style={{ width: "100%" }}
                  onRelayout={(event) => syncRange(event as Record<string, unknown>)}
                />
              </div>
            )
          )}

          {/* Differential / REF subplot (pu / In channels) */}
          {diffTraces.length > 0 && (
            <div className={styles.waveSubplot}>
              <div className={styles.waveSubplotTitle}>Differential / REF</div>
              <Plot
                data={diffTraces}
                layout={analogLayout("p.u. / In", diffYRange)}
                config={{ displayModeBar: false, responsive: true, scrollZoom: true, doubleClick: "reset" }}
                style={{ width: "100%" }}
                onRelayout={(event) => syncRange(event as Record<string, unknown>)}
              />
            </div>
          )}

          {showDigital && selectedStatus.length > 0 && (
            <div className={styles.waveSubplot}>
              <div className={styles.waveSubplotTitle} style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                <span>Sinyal Digital</span>
                <span style={{ fontSize: "0.68rem", color: "#94a3b8", fontWeight: 400 }}>(bar = aktif)</span>
                <span style={{ fontSize: "0.72rem", color: "#475569", fontWeight: 500 }}>
                  Hover: {digitalHoverReadout ? `${digitalHoverReadout.time.toFixed(2)} ms` : "-"}
                </span>
                <span style={{ fontSize: "0.72rem", color: "#0f172a", fontWeight: 700 }}>
                  Active: {digitalHoverReadout ? (digitalHoverReadout.active.length ? digitalHoverReadout.active.slice(0, 5).join(", ") : "None") : "-"}
                  {digitalHoverReadout && digitalHoverReadout.active.length > 5 ? ` +${digitalHoverReadout.active.length - 5}` : ""}
                </span>
              </div>
              <Plot
                data={digitalBarTraces}
                layout={digitalBarLayout}
                config={{ displayModeBar: false, responsive: true, scrollZoom: true, doubleClick: "reset" }}
                style={{ width: "100%" }}
                onRelayout={(event) => syncRange(event as Record<string, unknown>)}
                onHover={(event) => updateDigitalHover(event as Readonly<Plotly.PlotMouseEvent>)}
              />
            </div>
          )}
        </div>
      </div>

      {comtrade.warnings.length > 0 && (
        <div className={styles.waveHint}>
          {comtrade.warnings.join(" | ")}
        </div>
      )}
    </div>
  );
}
