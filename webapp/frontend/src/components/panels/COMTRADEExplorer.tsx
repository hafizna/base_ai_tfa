import { useEffect, useMemo, useState } from "react";

import type { AnalogChannel, ComtradeData, StatusChannel } from "../../context/AnalysisContext";
import Plot from "../plot/PlotlyChart";
import styles from "./Panel.module.css";

interface Props {
  comtrade: ComtradeData;
}

const MAX_PLOT_POINTS = 5000;
const PLOT_LEFT_MARGIN = 120; // shared margin keeps analog & digital x-axes aligned
type AnalogViewMode = "stacked" | "grouped";
type SOEPosition = "belowDigital" | "bottom";

interface DigitalEventRow {
  channel: string;
  timeMs: number;
  state: 0 | 1;
  durationMs: number | null;
}

/** Find intervals where digital channel is active (state=1). */
function findActiveIntervals(timeMs: number[], samples: number[]): Array<{ startMs: number; endMs: number; hasOffEdge: boolean }> {
  const intervals: Array<{ startMs: number; endMs: number; hasOffEdge: boolean }> = [];
  let inActive = false;
  let startT = 0;
  const len = Math.min(timeMs.length, samples.length);
  for (let i = 0; i < len; i++) {
    if (samples[i] === 1 && !inActive) { startT = timeMs[i]; inActive = true; }
    else if (samples[i] === 0 && inActive) { intervals.push({ startMs: startT, endMs: timeMs[i], hasOffEdge: true }); inActive = false; }
  }
  if (inActive && len > 0) intervals.push({ startMs: startT, endMs: timeMs[len - 1], hasOffEdge: false });
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

    intervals.forEach(({ startMs: t0, endMs: t1 }) => {
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

function buildDigitalEdgeTrace(
  channels: StatusChannel[],
  timeMs: number[],
): Plotly.Data | null {
  const x: number[] = [];
  const y: number[] = [];
  const customdata: Array<[string, string, string]> = [];
  const symbols: string[] = [];
  const colors: string[] = [];
  const n = channels.length;

  channels.forEach((ch, idx) => {
    const rowY = n - 1 - idx;
    const intervals = findActiveIntervals(timeMs, ch.samples);
    intervals.forEach((interval) => {
      x.push(interval.startMs);
      y.push(rowY);
      customdata.push([ch.name, "ON", interval.startMs.toFixed(2)]);
      symbols.push("triangle-right");
      colors.push("#10b981");

      if (interval.hasOffEdge) {
        x.push(interval.endMs);
        y.push(rowY);
        customdata.push([ch.name, "OFF", interval.endMs.toFixed(2)]);
        symbols.push("triangle-left");
        colors.push("#ef4444");
      }
    });
  });

  if (!x.length) return null;
  return {
    x,
    y,
    customdata,
    type: "scatter",
    mode: "markers",
    name: "Digital edges",
    marker: {
      color: colors,
      symbol: symbols,
      size: 11,
      line: { color: "#ffffff", width: 1 },
    },
    cliponaxis: false,
    showlegend: false,
    hovertemplate: `<b>%{customdata[0]}</b><br>%{customdata[1]} @ %{customdata[2]} ms<extra></extra>`,
  } as Plotly.Data;
}

function buildDigitalEventRows(
  channels: StatusChannel[],
  timeMs: number[],
  range: [number, number] | null,
): DigitalEventRow[] {
  const rows: DigitalEventRow[] = [];

  channels.forEach((ch) => {
    const len = Math.min(timeMs.length, ch.samples.length);
    const offTimes: number[] = [];

    for (let i = 1; i < len; i += 1) {
      const prev = ch.samples[i - 1] ? 1 : 0;
      const next = ch.samples[i] ? 1 : 0;
      if (prev !== next && next === 0) offTimes.push(timeMs[i]);
    }

    for (let i = 1; i < len; i += 1) {
      const prev = ch.samples[i - 1] ? 1 : 0;
      const next = ch.samples[i] ? 1 : 0;
      if (prev === next) continue;
      if (range && (timeMs[i] < range[0] || timeMs[i] > range[1])) continue;

      const offTime = next === 1 ? offTimes.find((t) => t > timeMs[i]) ?? null : null;
      rows.push({
        channel: ch.name,
        timeMs: timeMs[i],
        state: next as 0 | 1,
        durationMs: offTime !== null ? offTime - timeMs[i] : null,
      });
    }
  });

  return rows.sort((a, b) => a.timeMs - b.timeMs || a.channel.localeCompare(b.channel));
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

function phaseFromNameOrCanonical(ch: Pick<AnalogChannel, "name" | "canonical_name" | "phase">): string | null {
  if (ch.phase) return ch.phase;
  const text = `${ch.canonical_name} ${ch.name}`.toUpperCase();
  if (/\b(?:IA|IL1|I1|IDL1|IDIFF_A)\b|\bIL1\s*D\b/.test(text)) return "A";
  if (/\b(?:IB|IL2|I2|IDL2|IDIFF_B)\b|\bIL2\s*D\b/.test(text)) return "B";
  if (/\b(?:IC|IL3|I3|IDL3|IDIFF_C)\b|\bIL3\s*D\b/.test(text)) return "C";
  if (/\b(?:IN|I0|3I0|IDNS)\b/.test(text)) return "N";
  return null;
}

type ChannelRole =
  | "lineCurrent"
  | "remoteCurrent"
  | "windingCurrent"
  | "differential"
  | "restraint"
  | "voltage"
  | "neutralCurrent"
  | "unknown";

/**
 * Detect winding-side marker on a channel name.
 * Recognizes:
 *   - Siemens 7UT612: "iL1-S1", "iL2-S2"
 *   - ABB RET670 / GE T60: "W1 CT IL1", "W2 CT IL2"
 *   - Dotted prefix: "HVS.IA", "LVS.IB"
 *   - Free-text tokens: "REF HV", "REF LV", "MV", "TV"
 * Returns "S1".."S5" or empty.
 */
function extractSidedSuffix(name: string): string {
  const upper = name.toUpperCase();
  let m = upper.match(/-S([1-5])\b/);
  if (m) return `S${m[1]}`;
  m = upper.match(/\bW([1-3])\b/);
  if (m) return `S${m[1]}`;
  if (/\bHVS?\b/.test(upper)) return "S1";
  if (/\bLVS?\b/.test(upper)) return "S2";
  if (/\bMVS?\b/.test(upper)) return "S3";
  if (/\bTVS?\b/.test(upper)) return "S3";
  return "";
}

const SIDED_SUFFIX_LABEL: Record<string, string> = {
  S1: "HV Side",
  S2: "LV Side",
  S3: "TV Side",
  S4: "Side 4",
  S5: "Side 5",
};
const SIDED_SUFFIX_SHORT: Record<string, string> = {
  S1: "HVS",
  S2: "LVS",
  S3: "TVS",
  S4: "S4",
  S5: "S5",
};

function channelRole(ch: Pick<AnalogChannel, "name" | "canonical_name" | "unit" | "measurement" | "phase">): ChannelRole {
  const text = `${ch.name} ${ch.canonical_name}`.toUpperCase();
  const unit = ch.unit.toLowerCase();

  // Explicit name-based detection takes priority over any unit heuristic.
  // Standalone DIFF/BIAS tokens (e.g. "REF DIFF HV") matter for relays whose
  // names don't always carry the leading "I", so detect both forms.
  if (/\b(?:IBIAS|IREST|IRESTR|RESTRAINT|RSTR|BIAS)\b/.test(text)) return "restraint";
  if (
    /\b(?:IDIFF|IDIF|IDL[123]?|LDL|DIFF)\b/.test(text) ||
    /\bIL[123]\s*D\b/.test(text) ||
    /\bIL[123]D\b/.test(text)
  ) {
    return "differential";
  }

  if (ch.measurement === "voltage") return "voltage";

  if (ch.measurement === "current") {
    // Winding-side detection runs before any pu/neutral fallback so that
    // Siemens 7UT612 phase currents recorded in pu (e.g. "iL1-S1") and
    // ABB/GE "W1 CT IL1" / "REF HV" channels all get the windingCurrent role.
    if (extractSidedSuffix(ch.name)) return "windingCurrent";
    if (/\bREM(?:OTE)?\b|\bREM\s+L\b/.test(text)) return "remoteCurrent";
    if (/\b(?:IN|I0|3I0|IDNS)\b/.test(text)) return "neutralCurrent";
    if (unit === "pu" || unit === "in") return "differential";
    return "lineCurrent";
  }
  return "unknown";
}

function channelDisplay(ch: AnalogChannel) {
  const role = channelRole(ch);
  const phase = phaseFromNameOrCanonical(ch);
  const phaseText = phase ? `Ph ${phase}` : null;
  const raw = ch.name.trim();
  const canonical = (ch.canonical_name || "").trim();

  if (role === "differential") {
    return {
      title: phase && phase !== "N" ? `I Diff ${phase}` : raw,
      detail: `Differential current${phaseText ? ` / ${phaseText}` : ""}`,
      raw,
    };
  }

  if (role === "restraint") {
    return {
      title: phase && phase !== "N" ? `I Restraint ${phase}` : "I Restraint / Bias",
      detail: `Restraint or bias current${phaseText ? ` / ${phaseText}` : ""}`,
      raw,
    };
  }

  if (role === "remoteCurrent") {
    return {
      title: canonical && canonical !== raw ? `${canonical} Remote` : raw,
      detail: `Remote-end line current${phaseText ? ` / ${phaseText}` : ""}`,
      raw,
    };
  }

  if (role === "windingCurrent") {
    const sided = extractSidedSuffix(ch.name);
    const sideLabel = sided ? SIDED_SUFFIX_LABEL[sided] : "Winding";
    const sideShort = sided ? SIDED_SUFFIX_SHORT[sided] : "";
    const isRef = /\bREF\b/i.test(ch.name);
    let title: string;
    if (isRef) {
      title = sideShort ? `REF ${sideShort.replace(/S$/, "")}` : raw;
    } else if (phase === "N" && sideShort) {
      title = `IN ${sideShort}`;
    } else if (phase && phase !== "N" && sideShort) {
      title = `I${phase} ${sideShort}`;
    } else {
      title = canonical && canonical !== raw ? canonical : raw;
    }
    const detailKind = isRef ? "REF input current" : "Winding current";
    return {
      title,
      detail: `${sideLabel} ${detailKind}${phaseText ? ` / ${phaseText}` : ""}`,
      raw,
    };
  }

  if (role === "lineCurrent") {
    return {
      title: canonical && canonical !== raw ? canonical : raw,
      detail: `Local line current${phaseText ? ` / ${phaseText}` : ""}`,
      raw,
    };
  }

  if (role === "neutralCurrent") {
    return {
      title: canonical && canonical !== raw ? canonical : raw,
      detail: "Neutral / residual current",
      raw,
    };
  }

  if (role === "voltage") {
    return {
      title: canonical && canonical !== raw ? canonical : raw,
      detail: `Voltage${phaseText ? ` / ${phaseText}` : ""}`,
      raw,
    };
  }

  return {
    title: canonical || raw,
    detail: ch.measurement,
    raw,
  };
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
  // Siemens 7UT8x style: "Current IA.a" / "Current IA.b" / "Current IA.c"
  const dotted = name.match(/\.([abc])\s*$/i);
  if (dotted) return dotted[1].toLowerCase();
  // Reuse the unified side detector so W1/W2/W3 (ABB/GE), -S1..-S5 (Siemens
  // 7UT612), and HV/LV/MV/TV markers all participate in winding grouping.
  const sided = extractSidedSuffix(name);
  if (sided) {
    const idx = Number(sided.slice(1));
    return ["", "a", "b", "c", "d", "e"][idx] || "";
  }
  return "";
}

/** Map Siemens-style suffix to canonical short side code used elsewhere in the explorer. */
const SUFFIX_SIDE_SHORT: Record<string, string> = { a: "HVS", b: "LVS", c: "TVS", d: "S4", e: "S5" };
const SUFFIX_SIDE_LABEL: Record<string, string> = {
  a: "Side 1 (HV)",
  b: "Side 2 (LV)",
  c: "Side 3 (TV)",
  d: "Side 4 (Aux)",
  e: "Side 5 (Aux)",
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
    uniqueSuffixes.length <= 5 &&
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

function channelRange(samples: number[]): [number, number] | undefined {
  if (!samples.length) return undefined;
  let min = Infinity;
  let max = -Infinity;
  for (const value of samples) {
    if (!Number.isFinite(value)) continue;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return undefined;
  if (min === max) {
    const pad = Math.max(Math.abs(max) * 0.12, 1);
    return [min - pad, max + pad];
  }
  const pad = Math.max((max - min) * 0.12, Math.max(Math.abs(min), Math.abs(max)) * 0.04, 1e-6);
  return [min - pad, max + pad];
}

function formatStat(value: number, unit: string) {
  const abs = Math.abs(value);
  const text = abs >= 1000 ? value.toFixed(0) : abs >= 10 ? value.toFixed(2) : value.toFixed(3);
  return `${text} ${unit}`;
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

function firstSustainedIndex(flags: boolean[], startIdx: number, samplesNeeded: number) {
  let run = 0;
  for (let idx = startIdx; idx < flags.length; idx += 1) {
    run = flags[idx] ? run + 1 : 0;
    if (run >= samplesNeeded) return idx - samplesNeeded + 1;
  }
  return null;
}

function detectAnalogInceptionMs(comtrade: ComtradeData, cycleN: number): number | null {
  const timeMs = comtrade.time.map((t) => t * 1000);
  if (timeMs.length < Math.max(20, cycleN * 2)) return null;

  const baselineEnd = Math.max(8, Math.min(cycleN * 3, Math.floor(timeMs.length * 0.12)));
  const startIdx = Math.min(Math.max(baselineEnd, cycleN), timeMs.length - 2);
  const candidates: number[] = [];

  const currentChannels = comtrade.analog_channels.filter((ch) => ch.measurement === "current");
  currentChannels.forEach((ch) => {
    const baseline = ch.samples.slice(0, baselineEnd);
    const preRms = Math.sqrt(baseline.reduce((sum, value) => sum + value * value, 0) / Math.max(baseline.length, 1));
    const prePeak = Math.max(...baseline.map((value) => Math.abs(value)), 0);
    const peak = Math.max(...ch.samples.map((value) => Math.abs(value)), 0);
    const threshold = Math.max(preRms * 3.0, prePeak * 1.8, peak * 0.12, 0.05);
    const idx = firstSustainedIndex(ch.samples.map((value) => Math.abs(value) > threshold), startIdx, 3);
    if (idx !== null) candidates.push(idx);
  });

  const voltageChannels = comtrade.analog_channels.filter((ch) => ch.measurement === "voltage");
  voltageChannels.forEach((ch) => {
    const rms = computeRms(ch.samples, cycleN);
    const baseline = rms.slice(0, baselineEnd).filter((value) => Number.isFinite(value) && value > 0);
    if (!baseline.length) return;
    const preRms = baseline.reduce((sum, value) => sum + value, 0) / baseline.length;
    const sagLimit = preRms * 0.72;
    const swellLimit = preRms * 1.35;
    const idx = firstSustainedIndex(
      rms.map((value) => value < sagLimit || value > swellLimit),
      startIdx,
      Math.max(3, Math.floor(cycleN / 5))
    );
    if (idx !== null) candidates.push(idx);
  });

  if (!candidates.length) return null;
  const idx = Math.min(...candidates);
  return Number.isFinite(timeMs[idx]) ? timeMs[idx] : null;
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

    // Include relay-computed differential/restraint channels so 87L/87T files show their
    // operate quantities without the user having to know vendor-specific names.
    const diffIds = visibleAnalog
      .filter((ch) => {
        const role = channelRole(ch);
        return role === "differential" || role === "restraint";
      })
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
  const [analogViewMode, setAnalogViewMode] = useState<AnalogViewMode>("stacked");
  const [digitalHoverMs, setDigitalHoverMs] = useState<number | null>(null);
  const [showDigitalSOE, setShowDigitalSOE] = useState(true);
  const [soePosition, setSoePosition] = useState<SOEPosition>("belowDigital");

  // One electrical cycle in samples — used for RMS window
  const cycleN = useMemo(() => {
    const fs = comtrade.sampling_rates[0]?.[0] ?? 1200;
    const freq = comtrade.frequency ?? 50;
    return Math.max(2, Math.round(fs / freq));
  }, [comtrade]);

  // Detect fault inception from the earliest sustained analog disturbance.
  // Digital pickup/trip signals can legitimately lead the analogue fault, so they
  // are not used for this marker.
  const inceptionTimeMs = useMemo((): number | null => {
    return detectAnalogInceptionMs(comtrade, cycleN);
  }, [comtrade, cycleN]);

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
    const start = center <= 1000 ? 0 : Math.max(0, center - 300);
    return [start, Math.min(dur, center + 700)];
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
  const displayRecordStartMs = displayTimeMs[0] ?? 0;

  function switchGroup(groupIdx: number) {
    setSelectedGroup(groupIdx);
    const nextChannels =
      groupIdx === -1 ? analogChannels : channelGroups?.[groupIdx]?.channels ?? analogChannels;
    const nextIds = new Set(
      [
        ...nextChannels
        .filter((ch) => ["VA", "VB", "VC", "IA", "IB", "IC", "IN", "I0"].includes(ch.canonical_name || ch.name))
        .slice(0, 8)
        .map((ch) => ch.id),
        ...nextChannels
          .filter((ch) => {
            const role = channelRole(ch);
            return role === "differential" || role === "restraint";
          })
          .map((ch) => ch.id),
      ]
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
  function isDiffChannel(ch: AnalogChannel): boolean {
    const role = channelRole(ch);
    return role === "differential" || role === "restraint";
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
  const selectedAnalogStrips = visibleAnalog.filter((ch) => selectedAnalog.has(ch.id));
  const selectedStatus = statusChannels.filter((ch) => selectedDigital.has(ch.id));

  // All current strips always share one symmetric y-range so phase amplitudes
  // stay honest relative to each other — a 0.4 kA phase renders ~1/5 the height
  // of a 2 kA faulted phase instead of both filling their strip. This is the
  // only sensible default for fault analysis, so it is not user-toggleable.
  const sharedCurrentYRange = useMemo((): [number, number] | undefined => {
    const currents = selectedAnalogStrips.filter((ch) => ch.measurement === "current");
    if (currents.length < 2) return undefined;
    let max = 0;
    for (const ch of currents) {
      const p = channelPeak(samplesForDisplay(ch));
      if (p > max) max = p;
    }
    if (max === 0) return undefined;
    const pad = max * 0.12;
    return [-(max + pad), max + pad];
  // samplesForDisplay closes over displayMode/cycleN; both are in deps via the array below
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAnalogStrips, displayMode, cycleN]);

  const xAxisLabel = normalizeToInception ? "Waktu relatif inception (ms)" : "Waktu (ms)";
  const inceptionDisplayMs = normalizeToInception ? 0 : (inceptionTimeMs ?? null);

  const voltageTraces = buildTraces(selectedVoltage, "voltage");

  function buildTraces(
    channels: typeof analogChannels,
    measurement: string,
  ): Plotly.Data[] {
    return channels.map((ch) => {
      const info = channelDisplay(ch);
      const lbl = info.title;
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
        hovertemplate: `<b>${lbl}</b><br>${info.detail}<br>${info.raw}<br>%{y:.3f} ${ch.unit}<br>%{x:.2f} ms<extra></extra>`,
      } as Plotly.Data;
    });
  }

  const currentTraces = buildTraces(selectedCurrent, "current");

  const diffTraces = buildTraces(selectedDiff, "current");

  function samplesForDisplay(ch: AnalogChannel) {
    return displayMode === "rms" && ch.measurement !== "voltage"
      ? computeRms(ch.samples, cycleN)
      : ch.samples;
  }

  function channelStats(ch: AnalogChannel) {
    const samples = samplesForDisplay(ch);
    const range = activeRange;
    const values = samples.filter((_, idx) => {
      if (!range) return true;
      const t = displayTimeMs[idx];
      return t >= range[0] && t <= range[1];
    });
    const source = values.length > 0 ? values : samples;
    const max = Math.max(...source);
    const min = Math.min(...source);
    const rms = Math.sqrt(source.reduce((sum, value) => sum + value * value, 0) / Math.max(source.length, 1));
    return { max, min, rms };
  }

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
  const digitalEdgeTrace = useMemo(
    () => buildDigitalEdgeTrace(selectedStatus, displayTimeMs),
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
    line: { color: "#dc2626", width: 1.6, dash: "dot" },
  } as Plotly.Shape : null;

  // Distinct timeline markers shared by every analog/digital plot.
  const recordStartShape: Plotly.Shape = {
    type: "line",
    x0: displayRecordStartMs,
    x1: displayRecordStartMs,
    yref: "paper",
    y0: 0,
    y1: 1,
    line: { color: "#64748b", width: 1.4, dash: "dash" },
  } as Plotly.Shape;

  const triggerShape: Plotly.Shape | null = displayTriggerMs !== null ? {
    type: "line",
    x0: displayTriggerMs, x1: displayTriggerMs,
    yref: "paper", y0: 0, y1: 1,
    line: { color: "#7c3aed", width: 1.5, dash: "longdash" },
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
  const digitalSOERows = useMemo(
    () => buildDigitalEventRows(selectedStatus, displayTimeMs, activeRange),
    [selectedStatus, displayTimeMs, activeRange]
  );
  const visibleDigitalSOERows = digitalSOERows.slice(0, 120);

  const channelStripLayout = (ch: AnalogChannel, yRange?: [number, number]): Partial<Plotly.Layout> => ({
    height: 170,
    margin: { l: PLOT_LEFT_MARGIN, r: 20, t: 8, b: 24 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    hovermode: "x unified",
    dragmode: "pan",
    showlegend: false,
    uirevision: "comtrade-sync",
    xaxis: {
      tickfont: { size: 9 },
      autorange: activeRange ? undefined : true,
      range: activeRange ?? undefined,
      showgrid: true,
      gridcolor: "#e2e8f0",
      zeroline: normalizeToInception,
      zerolinecolor: "#dc2626",
      zerolinewidth: 1.5,
    },
    yaxis: {
      title: { text: ch.unit },
      tickfont: { size: 9 },
      tickformat: "~s",
      exponentformat: "SI",
      separatethousands: true,
      fixedrange: true,
      ...(yRange ? { range: yRange, autorange: false } : { autorange: true }),
    },
    shapes: [
      recordStartShape,
      ...(inceptionShape && !normalizeToInception ? [inceptionShape] : []),
      ...(triggerShape ? [triggerShape] : []),
    ],
  });

  function buildChannelTrace(ch: AnalogChannel): Plotly.Data {
    const info = channelDisplay(ch);
    const lbl = info.title;
    const series = buildSampledSeries(displayTimeMs, samplesForDisplay(ch), MAX_PLOT_POINTS);
    return {
      x: series.x,
      y: series.y,
      type: "scatter",
      mode: "lines",
      name: lbl,
      line: { color: inferChannelColor(lbl, ch.measurement), width: 1.2 },
      hovertemplate: `<b>${lbl}</b><br>${info.detail}<br>${info.raw}<br>%{y:.3f} ${ch.unit}<br>%{x:.2f} ms<extra></extra>`,
    } as Plotly.Data;
  }

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
      zerolinecolor: "#dc2626",
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
      recordStartShape,
      ...(inceptionShape && !normalizeToInception ? [inceptionShape] : []),
      ...(triggerShape ? [triggerShape] : []),
    ],
  });

  const digitalBarLayout: Partial<Plotly.Layout> = {
    height: Math.max(220, selectedStatus.length * 32 + 70),
    margin: { l: 160, r: 24, t: 12, b: 44 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    hovermode: "closest",
    dragmode: "pan",
    showlegend: false,
    uirevision: "comtrade-sync",
    xaxis: {
      title: { text: xAxisLabel },
      tickfont: { size: 9 },
      tickformat: ".2f",
      autorange: activeRange ? undefined : true,
      range: activeRange ?? undefined,
      showgrid: true,
      gridcolor: "#e2e8f0",
      zeroline: normalizeToInception,
      zerolinecolor: "#dc2626",
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
      recordStartShape,
      ...(inceptionShape && !normalizeToInception ? [inceptionShape] : []),
      ...(triggerShape ? [triggerShape] : []),
      ...(digitalHoverShape ? [digitalHoverShape] : []),
    ],
  };

  function renderDigitalSOE() {
    return (
      <div className={styles.digitalEventPanel}>
        <button
          type="button"
          className={styles.digitalEventToggle}
          onClick={() => setShowDigitalSOE((value) => !value)}
        >
          <div className={styles.digitalEventHeader}>
            <span>SOE Digital</span>
            <span>
              {digitalSOERows.length} events
              {digitalSOERows.length > visibleDigitalSOERows.length ? `, showing ${visibleDigitalSOERows.length}` : ""}
              {" | "}
              {showDigitalSOE ? "Hide" : "Show"}
            </span>
          </div>
        </button>
        <div className={`${styles.digitalEventBody} ${showDigitalSOE ? "" : styles.digitalEventBodyCollapsed}`}>
          {visibleDigitalSOERows.length > 0 ? (
            <div className={styles.digitalEventTable}>
              <div className={styles.digitalEventHead}>Time</div>
              <div className={styles.digitalEventHead}>State</div>
              <div className={styles.digitalEventHead}>Duration</div>
              <div className={styles.digitalEventHead}>Channel</div>
              {visibleDigitalSOERows.map((event, idx) => (
                <div className={styles.digitalEventRow} key={`${event.channel}-${event.timeMs}-${event.state}-${idx}`}>
                  <span className={styles.digitalEventTime}>{event.timeMs.toFixed(2)} ms</span>
                  <span className={event.state ? styles.digitalEventOn : styles.digitalEventOff}>
                    {event.state ? "ON" : "OFF"}
                  </span>
                  <span className={styles.digitalEventDuration}>
                    {event.durationMs !== null ? `${event.durationMs.toFixed(2)} ms` : "-"}
                  </span>
                  <span className={styles.digitalEventChannel} title={event.channel}>{event.channel}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className={styles.digitalEventEmpty}>Tidak ada perubahan status digital pada kanal/waktu yang sedang ditampilkan.</div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={styles.waveShell}>
      <div className={styles.waveHead}>
        <div>
          <div className={styles.waveTitle}>COMTRADE Explorer</div>
          <div className={styles.waveSub}>
            Semua plot tersinkron. Garis abu-abu (---) = awal rekaman. Garis ungu (-- --) = trigger CFG. Garis merah (dot) = inception terdeteksi.
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
          <button
            type="button"
            className={`${styles.waveGhostBtn} ${styles.waveModeBtn}`}
            onClick={() => setAnalogViewMode((mode) => mode === "stacked" ? "grouped" : "stacked")}
            style={analogViewMode === "stacked" ? { background: "#f0fdf4", borderColor: "#22c55e", color: "#15803d" } : undefined}
          >
            {analogViewMode === "stacked" ? "Stacked Channels" : "Grouped Overlay"}
          </button>
          {inceptionTimeMs !== null && (
            <button
              type="button"
              className={styles.waveGhostBtn}
              onClick={() => { setNormalizeToInception((v) => !v); setSharedRange(null); }}
              style={normalizeToInception ? { background: "#fef2f2", borderColor: "#dc2626", color: "#b91c1c" } : undefined}
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
            const info = channelDisplay(ch);
            const label = info.title;
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
                    {windingSide ?? info.detail}
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
          {analogViewMode === "stacked" && selectedAnalogStrips.map((ch) => {
            const info = channelDisplay(ch);
            const label = info.title;
            const stats = channelStats(ch);
            const yRange =
              ch.measurement === "current" && sharedCurrentYRange
                ? sharedCurrentYRange
                : channelRange(samplesForDisplay(ch));
            return (
              <div
                key={ch.id}
                className={styles.waveSubplot}
                data-pdf-chart-id={`waveform_analog_${ch.id}`}
                data-pdf-chart-title={`Waveform Analog - ${label}`}
              >
                <div className={styles.waveSubplotTitle} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                    <span className={styles.waveDot} style={{ background: inferChannelColor(label, ch.measurement) }} />
                    <span>{label}</span>
                    <span style={{ color: "#64748b", fontWeight: 600 }}>{info.detail}</span>
                    <span style={{ color: "#94a3b8", fontWeight: 500 }}>{ch.unit}</span>
                  </span>
                  <span style={{ fontSize: "0.68rem", color: "#64748b", fontWeight: 600, whiteSpace: "nowrap" }}>
                    {ch.measurement === "current" && sharedCurrentYRange && (
                      <span style={{ color: "#3b82f6", fontWeight: 700 }}>skala bersama · </span>
                    )}
                    Max: {formatStat(stats.max, ch.unit)} | Min: {formatStat(stats.min, ch.unit)} | RMS: {formatStat(stats.rms, ch.unit)}
                  </span>
                </div>
                <Plot
                  data={[buildChannelTrace(ch)]}
                  layout={channelStripLayout(ch, yRange)}
                  config={{ displayModeBar: false, responsive: true, scrollZoom: true, doubleClick: "reset" }}
                  style={{ width: "100%" }}
                  onRelayout={(event) => syncRange(event as Record<string, unknown>)}
                />
              </div>
            );
          })}

          {analogViewMode === "grouped" && (
            <>
          {voltageTraces.length > 0 && (
            <div
              className={styles.waveSubplot}
              data-pdf-chart-id="waveform_voltage"
              data-pdf-chart-title="Waveform Tegangan"
            >
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
              <div
                className={styles.waveSubplot}
                data-pdf-chart-id="waveform_current"
                data-pdf-chart-title="Waveform Arus"
              >
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

          {/* Differential / restraint subplot */}
          {diffTraces.length > 0 && (
            <div className={styles.waveSubplot}>
              <div className={styles.waveSubplotTitle}>Differential / Restraint</div>
              <Plot
                data={diffTraces}
                layout={analogLayout("p.u. / In", diffYRange)}
                config={{ displayModeBar: false, responsive: true, scrollZoom: true, doubleClick: "reset" }}
                style={{ width: "100%" }}
                onRelayout={(event) => syncRange(event as Record<string, unknown>)}
              />
            </div>
          )}
            </>
          )}

          {showDigital && selectedStatus.length > 0 && (
            <div
              className={styles.waveSubplot}
              data-pdf-chart-id="digital_status"
              data-pdf-chart-title="Sinyal Digital (Snapshot)"
            >
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
                <label className={styles.digitalEventPosition}>
                  SOE
                  <select
                    value={soePosition}
                    onChange={(event) => setSoePosition(event.target.value as SOEPosition)}
                  >
                    <option value="belowDigital">Below digital</option>
                    <option value="bottom">Bottom</option>
                  </select>
                </label>
              </div>
              <Plot
                data={[...digitalBarTraces, ...(digitalEdgeTrace ? [digitalEdgeTrace] : [])]}
                layout={digitalBarLayout}
                config={{ displayModeBar: false, responsive: true, scrollZoom: true, doubleClick: "reset" }}
                style={{ width: "100%" }}
                onRelayout={(event) => syncRange(event as Record<string, unknown>)}
                onHover={(event) => updateDigitalHover(event as Readonly<Plotly.PlotMouseEvent>)}
              />
              {soePosition === "belowDigital" && renderDigitalSOE()}
            </div>
          )}
        </div>
      </div>

      {showDigital && selectedStatus.length > 0 && soePosition === "bottom" && (
        <div className={styles.digitalEventBottom}>
          {renderDigitalSOE()}
        </div>
      )}

      {comtrade.warnings.length > 0 && (
        <div className={styles.waveHint}>
          {comtrade.warnings.join(" | ")}
        </div>
      )}
    </div>
  );
}

