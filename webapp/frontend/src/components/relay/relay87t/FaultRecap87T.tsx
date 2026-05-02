import { useMemo } from "react";
import type { AnalogChannel, ComtradeData } from "../../../context/AnalysisContext";
import styles from "../../panels/Panel.module.css";

interface Props {
  comtrade: ComtradeData;
  relayLabel?: string;
}

// ─── channel helpers ────────────────────────────────────────────────────────

function dotPrefix(name: string): string {
  const idx = name.indexOf(".");
  return idx > 0 ? name.slice(0, idx).toUpperCase() : "";
}

function windingLabel(prefix: string): string {
  const u = prefix.toUpperCase();
  if (u === "HVS" || u === "HV") return "HV Side";
  if (u === "LVS" || u === "LV") return "LV Side";
  if (u === "MVS" || u === "MV") return "MV Side";
  return prefix || "Main";
}

function phaseOf(ch: AnalogChannel): "A" | "B" | "C" | null {
  const canon = (ch.canonical_name ?? "").toUpperCase();
  if (canon === "IA" || canon === "IL1") return "A";
  if (canon === "IB" || canon === "IL2") return "B";
  if (canon === "IC" || canon === "IL3") return "C";
  const u = ch.name.toUpperCase();
  if (/\.IA$|_IA$/.test(u) || u.endsWith("IA")) return "A";
  if (/\.IB$|_IB$/.test(u) || u.endsWith("IB")) return "B";
  if (/\.IC$|_IC$/.test(u) || u.endsWith("IC")) return "C";
  return null;
}

function isRelayDiff(ch: AnalogChannel): boolean {
  const u = ch.unit.toLowerCase();
  const name = `${ch.name} ${ch.canonical_name}`.toUpperCase();
  return (
    u === "pu" ||
    u === "in" ||
    /\b(?:IDIFF|IDIF|IDL[123]?|IBIAS|IREST|IRESTR|BIAS|RESTRAINT|LDL)\b/.test(name) ||
    /\bIL[123]\s*D\b/.test(name) ||
    /\bIL[123]D\b/.test(name)
  );
}

type DigType = "trip" | "ref" | "diff" | "trigger" | "alarm" | "other";

function classifyDig(name: string): DigType {
  const u = name.toUpperCase();
  if (/TRPOUT|TRP_OUT|TRP\d|TRIPOUT/.test(u)) return "trip";
  if (/64REF|REF\.OP|REF_OP|SBEF|GFR\.OP/.test(u)) return "ref";
  if (/87[TL]\.|L3D|LT3D|LDL|DIFL|DIFF/.test(u)) return "diff";
  if (/TRIGDFR|TRIG\.DFR|DFR\.TRIG|TRIGDFR/.test(u)) return "trigger";
  if (/ALM|ALARM/.test(u)) return "alarm";
  return "other";
}

function rmsOf(samples: number[]): number {
  if (!samples.length) return 0;
  return Math.sqrt(samples.reduce((a, v) => a + v * v, 0) / samples.length);
}

function peakOf(samples: number[]): number {
  return samples.reduce((a, v) => Math.max(a, Math.abs(v)), 0);
}

// ─── analysis ───────────────────────────────────────────────────────────────

interface PhaseMetric {
  preFaultRms: number;
  faultPeak: number;
  ratio: number;
  elevated: boolean;
}

interface WindingStat {
  prefix: string;
  label: string;
  phases: Partial<Record<"A" | "B" | "C", { preFaultRms: number; faultPeak: number }>>;
}

interface DigEvent {
  name: string;
  type: DigType;
  activateMs: number | null;
  durationMs: number | null;
}

interface FaultAnalysis {
  windingStats: WindingStat[];
  phaseMetrics: Record<"A" | "B" | "C", PhaseMetric>;
  elevatedPhases: ("A" | "B" | "C")[];
  faultCode: string;
  faultLabel: string;
  faultDescription: string;
  digEvents: DigEvent[];
  tripTimeMs: number | null;
  tripDurationMs: number | null;
  refOperated: boolean;
  refWindings: string[];
  inceptionMs: number | null;
}

interface LineDiffAnalog {
  channel: string;
  role: "Differential" | "Restraint";
  phase: "A" | "B" | "C" | "N" | "-";
  peak: number;
  rms: number;
  firstRiseMs: number | null;
}

interface LineDiffAnalysis {
  analogs: LineDiffAnalog[];
  diffAnalogs: LineDiffAnalog[];
  restraintAnalogs: LineDiffAnalog[];
  activePhases: ("A" | "B" | "C" | "N")[];
  digEvents: DigEvent[];
  diffEvents: DigEvent[];
  tripEvents: DigEvent[];
  localTripMs: number | null;
  remoteTripMs: number | null;
  tripOutputMs: number | null;
  inceptionMs: number | null;
}

function analyze(comtrade: ComtradeData): FaultAnalysis {
  const timeMs = comtrade.time.map((t) => t * 1000);
  const triggerMs = comtrade.trigger_time * 1000;

  // Pre-fault window: up to CFG trigger or 20% of recording
  const preFaultEndIdx =
    triggerMs > 20
      ? Math.max(1, timeMs.findIndex((t) => t >= triggerMs))
      : Math.max(1, Math.floor(timeMs.length * 0.2));

  // --- Group current channels by winding prefix ---
  const currentChs = comtrade.analog_channels.filter(
    (ch) => ch.measurement === "current" && !isRelayDiff(ch)
  );

  const windingMap = new Map<string, AnalogChannel[]>();
  for (const ch of currentChs) {
    const prefix = dotPrefix(ch.name) || "MAIN";
    if (!windingMap.has(prefix)) windingMap.set(prefix, []);
    windingMap.get(prefix)!.push(ch);
  }

  const phaseMetrics: Record<"A" | "B" | "C", PhaseMetric> = {
    A: { preFaultRms: 0, faultPeak: 0, ratio: 1, elevated: false },
    B: { preFaultRms: 0, faultPeak: 0, ratio: 1, elevated: false },
    C: { preFaultRms: 0, faultPeak: 0, ratio: 1, elevated: false },
  };

  const windingStats: WindingStat[] = [];

  for (const [prefix, channels] of windingMap) {
    const phases: WindingStat["phases"] = {};
    for (const ch of channels) {
      const ph = phaseOf(ch);
      if (!ph || !ch.samples.length) continue;
      const pre = ch.samples.slice(0, preFaultEndIdx);
      const post = ch.samples.slice(preFaultEndIdx);
      const preRms = rmsOf(pre);
      const postPeak = peakOf(post);
      phases[ph] = { preFaultRms: preRms, faultPeak: postPeak };
      if (postPeak > phaseMetrics[ph].faultPeak) phaseMetrics[ph].faultPeak = postPeak;
      if (preRms > phaseMetrics[ph].preFaultRms) phaseMetrics[ph].preFaultRms = preRms;
    }
    windingStats.push({ prefix, label: windingLabel(prefix), phases });
  }

  // Classify elevation: a phase is elevated if its peak is ≥ 25% of the max peak
  // AND its fault/prefault ratio is at least 1.8× (or significantly above noise floor)
  const maxPeak = Math.max(...(["A", "B", "C"] as const).map((ph) => phaseMetrics[ph].faultPeak));
  for (const ph of ["A", "B", "C"] as const) {
    const m = phaseMetrics[ph];
    m.ratio = m.preFaultRms > 1 ? m.faultPeak / m.preFaultRms : m.faultPeak;
    m.elevated = maxPeak > 0 && m.faultPeak >= maxPeak * 0.25 && (m.ratio >= 1.8 || m.faultPeak >= maxPeak * 0.5);
  }

  // Fault inception: first sample crossing threshold in any current channel
  let inceptionMs: number | null = null;
  for (const ch of currentChs) {
    const s = ch.samples;
    const pre = s.slice(0, preFaultEndIdx);
    const preRms = rmsOf(pre);
    const thr = Math.max(preRms * 1.5, maxPeak * 0.1, 0.001);
    for (let i = preFaultEndIdx; i < s.length; i++) {
      if (Math.abs(s[i]) > thr) {
        const t = timeMs[i];
        if (inceptionMs === null || t < inceptionMs) inceptionMs = t;
        break;
      }
    }
  }

  const elevatedPhases = (["A", "B", "C"] as const).filter((ph) => phaseMetrics[ph].elevated);

  // --- Digital channel events ---
  const digEvents: DigEvent[] = comtrade.status_channels
    .filter((ch) => ch.samples.some((s) => s === 1))
    .map((ch) => {
      const type = classifyDig(ch.name);
      let activateMs: number | null = null;
      let deactivateMs: number | null = null;
      for (let i = 0; i < ch.samples.length; i++) {
        if (ch.samples[i] === 1 && activateMs === null) activateMs = timeMs[i] ?? null;
        if (ch.samples[i] === 0 && activateMs !== null && deactivateMs === null)
          deactivateMs = timeMs[i] ?? null;
      }
      return {
        name: ch.name,
        type,
        activateMs,
        durationMs:
          activateMs !== null && deactivateMs !== null ? deactivateMs - activateMs : null,
      };
    });

  const tripEvents = digEvents.filter((e) => e.type === "trip" && e.activateMs !== null);
  const tripTimeMs = tripEvents.length > 0 ? Math.min(...tripEvents.map((e) => e.activateMs!)) : null;
  const tripDurationMs =
    tripEvents.find((e) => e.durationMs !== null)?.durationMs ?? null;

  const refOperated = digEvents.some((e) => e.type === "ref");
  const refWindings = [
    ...new Set(
      digEvents
        .filter((e) => e.type === "ref")
        .map((e) => dotPrefix(e.name))
        .filter(Boolean)
    ),
  ];

  // --- Fault classification ---
  const n = elevatedPhases.length;
  let faultCode: string;
  let faultLabel: string;
  let faultDescription: string;

  if (n === 0) {
    faultCode = "?";
    faultLabel = "Tidak Terdeteksi";
    faultDescription = "Tidak ada fase yang menunjukkan peningkatan signifikan setelah gangguan.";
  } else if (n === 1) {
    const ph = elevatedPhases[0];
    if (refOperated) {
      const side = refWindings.length > 0 ? ` sisi ${refWindings.map(windingLabel).join("/")}` : "";
      faultCode = `REF-${ph}`;
      faultLabel = `REF / GFR — Fase ${ph}`;
      faultDescription = `Gangguan tanah terbatas (Restricted Earth Fault) Fase ${ph}${side} — elemen 64REF/GFR beroperasi.`;
    } else {
      faultCode = `SLG-${ph}`;
      faultLabel = `1 Fasa ke Tanah — Fase ${ph}`;
      faultDescription = `Gangguan satu fasa ke tanah terdeteksi pada Fase ${ph}.`;
    }
  } else if (n === 2) {
    const phs = elevatedPhases.join("");
    faultCode = `LL-${phs}`;
    faultLabel = `2 Fasa — Fase ${elevatedPhases.join("-")}`;
    faultDescription = `Gangguan dua fasa terdeteksi pada Fase ${elevatedPhases.join(" dan ")}.`;
  } else {
    faultCode = "3Ph";
    faultLabel = "3 Fasa";
    faultDescription = "Gangguan tiga fasa — semua fase terdampak.";
  }

  return {
    windingStats,
    phaseMetrics,
    elevatedPhases,
    faultCode,
    faultLabel,
    faultDescription,
    digEvents,
    tripTimeMs,
    tripDurationMs,
    refOperated,
    refWindings,
    inceptionMs,
  };
}

// ─── sub-components ─────────────────────────────────────────────────────────

function phaseOfLineDiff(ch: AnalogChannel): "A" | "B" | "C" | "N" | "-" {
  const text = `${ch.name} ${ch.canonical_name}`.toUpperCase();
  if (/\b(?:IDIFF_A|IDL1|IL1D|IL1\s+D)\b/.test(text)) return "A";
  if (/\b(?:IDIFF_B|IDL2|IL2D|IL2\s+D)\b/.test(text)) return "B";
  if (/\b(?:IDIFF_C|IDL3|IL3D|IL3\s+D)\b/.test(text)) return "C";
  if (/\b(?:IDIFF_N|IDNS|IN\s+D)\b/.test(text)) return "N";
  return "-";
}

function isLineDiffAnalog(ch: AnalogChannel): boolean {
  const text = `${ch.name} ${ch.canonical_name}`.toUpperCase();
  return (
    /\b(?:IDIFF|IDIF|IDL[123]?|LDL)\b/.test(text) ||
    /\bIL[123]\s*D\b/.test(text) ||
    /\bIL[123]D\b/.test(text) ||
    /\bIDNS(?:MAG)?\b/.test(text)
  );
}

function isRestraintAnalog(ch: AnalogChannel): boolean {
  const text = `${ch.name} ${ch.canonical_name}`.toUpperCase();
  return /\b(?:IREST|IRESTR|IBIAS|BIAS|RESTRAINT|RSTR)\b/.test(text);
}

function firstRiseMs(samples: number[], timeMs: number[]): number | null {
  if (!samples.length || !timeMs.length) return null;
  const preEnd = Math.max(4, Math.floor(samples.length * 0.2));
  const pre = samples.slice(0, preEnd);
  const preRms = rmsOf(pre);
  const peak = peakOf(samples);
  const threshold = Math.max(preRms * 2, peak * 0.15, 0.001);
  for (let i = preEnd; i < samples.length; i++) {
    if (Math.abs(samples[i]) >= threshold) return timeMs[i] ?? null;
  }
  return null;
}

function lineDiffEventTag(name: string) {
  const upper = name.toUpperCase();
  if (/TRLOCAL|LOCAL/.test(upper)) return "LOCAL";
  if (/TRREMOTE|REMOTE/.test(upper)) return "REMOTE";
  if (/TRL[123]|LT3D|LDL|DIFF|DIFL/.test(upper)) return "DIFF";
  if (/TRIP|TRP/.test(upper)) return "TRIP";
  if (/AR|RECLOS|RECLOSE/.test(upper)) return "AR";
  return "INFO";
}

function analyzeLineDiff(comtrade: ComtradeData): LineDiffAnalysis {
  const timeMs = comtrade.time.map((t) => t * 1000);
  const analogs: LineDiffAnalog[] = comtrade.analog_channels
    .filter((ch) => isLineDiffAnalog(ch) || isRestraintAnalog(ch))
    .map((ch) => {
      const role: LineDiffAnalog["role"] = isRestraintAnalog(ch) ? "Restraint" : "Differential";
      return {
        channel: ch.name,
        role,
        phase: phaseOfLineDiff(ch),
        peak: peakOf(ch.samples),
        rms: rmsOf(ch.samples),
        firstRiseMs: firstRiseMs(ch.samples, timeMs),
      };
    })
    .sort((a, b) => {
      const roleRank = a.role === b.role ? 0 : a.role === "Differential" ? -1 : 1;
      if (roleRank !== 0) return roleRank;
      return a.channel.localeCompare(b.channel);
    });

  const diffAnalogs = analogs.filter((item) => item.role === "Differential");
  const restraintAnalogs = analogs.filter((item) => item.role === "Restraint");
  const maxDiffPeak = Math.max(0, ...diffAnalogs.map((item) => item.peak));
  const activePhases = [
    ...new Set(
      diffAnalogs
        .filter((item) => item.phase !== "-" && item.peak >= Math.max(maxDiffPeak * 0.2, 0.001))
        .map((item) => item.phase as "A" | "B" | "C" | "N")
    ),
  ];

  const digEvents: DigEvent[] = comtrade.status_channels
    .filter((ch) => ch.samples.some((s) => s === 1))
    .map((ch) => {
      let activateMs: number | null = null;
      let deactivateMs: number | null = null;
      for (let i = 0; i < ch.samples.length; i++) {
        if (ch.samples[i] === 1 && activateMs === null) activateMs = timeMs[i] ?? null;
        if (ch.samples[i] === 0 && activateMs !== null && deactivateMs === null) {
          deactivateMs = timeMs[i] ?? null;
        }
      }
      return {
        name: ch.name,
        type: classifyDig(ch.name),
        activateMs,
        durationMs: activateMs !== null && deactivateMs !== null ? deactivateMs - activateMs : null,
      };
    });

  const diffEvents = digEvents
    .filter((event) => /L3D|LT3D|LDL|DIFL|DIFF|87L/i.test(event.name))
    .sort((a, b) => (a.activateMs ?? Infinity) - (b.activateMs ?? Infinity));
  const tripEvents = digEvents
    .filter((event) => /TRIP|TRP|TRL|TRLOCAL|TRREMOTE/i.test(event.name))
    .sort((a, b) => (a.activateMs ?? Infinity) - (b.activateMs ?? Infinity));

  const localTripMs = diffEvents.find((event) => /TRLOCAL|LOCAL/i.test(event.name))?.activateMs ?? null;
  const remoteTripMs = diffEvents.find((event) => /TRREMOTE|REMOTE/i.test(event.name))?.activateMs ?? null;
  const tripOutputMs = tripEvents.find((event) => /^TRIP|TRIP\s|TRIP[123LP]|CB\s/i.test(event.name))?.activateMs
    ?? tripEvents[0]?.activateMs
    ?? null;
  const inceptionMs = diffAnalogs
    .map((item) => item.firstRiseMs)
    .filter((value): value is number => value !== null)
    .sort((a, b) => a - b)[0] ?? null;

  return {
    analogs,
    diffAnalogs,
    restraintAnalogs,
    activePhases,
    digEvents,
    diffEvents,
    tripEvents,
    localTripMs,
    remoteTripMs,
    tripOutputMs,
    inceptionMs,
  };
}

function MetricCard({
  label,
  value,
  unit,
  highlight = false,
}: {
  label: string;
  value: string;
  unit: string;
  highlight?: boolean;
}) {
  return (
    <div
      style={{
        padding: "10px 14px",
        background: highlight ? "#fef2f2" : "#f8fafc",
        border: `1.5px solid ${highlight ? "#fca5a5" : "#e2e8f0"}`,
        borderRadius: 10,
      }}
    >
      <div
        style={{
          fontSize: "0.7rem",
          color: "#64748b",
          fontWeight: 600,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          marginBottom: 2,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: "1.05rem",
          fontWeight: 700,
          color: highlight ? "#dc2626" : "#1e293b",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
        <span style={{ fontSize: "0.75rem", fontWeight: 400, color: "#64748b", marginLeft: 4 }}>
          {unit}
        </span>
      </div>
    </div>
  );
}

const TH: React.CSSProperties = {
  padding: "6px 10px",
  textAlign: "left",
  borderBottom: "1.5px solid #e2e8f0",
  fontWeight: 700,
  color: "#475569",
  fontSize: "0.78rem",
  whiteSpace: "nowrap",
};

const TD: React.CSSProperties = {
  padding: "6px 10px",
  borderBottom: "1px solid #f1f5f9",
  fontSize: "0.82rem",
  fontVariantNumeric: "tabular-nums",
};

// ─── main component ─────────────────────────────────────────────────────────

function LineDifferentialRecap({ comtrade }: { comtrade: ComtradeData }) {
  const r = useMemo(() => analyzeLineDiff(comtrade), [comtrade]);
  const operated = r.diffEvents.length > 0 || r.tripEvents.length > 0;
  const firstOperateMs = r.diffEvents[0]?.activateMs ?? r.tripOutputMs;
  const inceptionToTrip =
    r.inceptionMs !== null && firstOperateMs !== null ? firstOperateMs - r.inceptionMs : null;
  const sequenceEvents = [...r.diffEvents, ...r.tripEvents]
    .filter((event, index, arr) => arr.findIndex((item) => item.name === event.name) === index)
    .sort((a, b) => (a.activateMs ?? Infinity) - (b.activateMs ?? Infinity))
    .slice(0, 12);

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>87L Operation Recap</h2>
      </div>

      <div className={styles.row} style={{ alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <span
          className={styles.statusBadge}
          style={
            operated
              ? { background: "#eff6ff", color: "#1d4ed8", border: "1.5px solid #93c5fd" }
              : { background: "#f8fafc", color: "#64748b", border: "1.5px solid #cbd5e1" }
          }
        >
          {operated ? "Differential trip observed" : "No differential trip signal found"}
        </span>
        {(["A", "B", "C", "N"] as const).map((ph) => {
          const active = r.activePhases.includes(ph);
          return (
            <span
              key={ph}
              title={`Differential phase ${ph}`}
              style={{
                width: 32,
                height: 32,
                borderRadius: "50%",
                border: "2px solid",
                borderColor: active ? "#2563eb" : "#e2e8f0",
                background: active ? "#eff6ff" : "#f8fafc",
                color: active ? "#1d4ed8" : "#94a3b8",
                fontWeight: 800,
                fontSize: "0.78rem",
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {ph}
            </span>
          );
        })}
      </div>

      <p style={{ fontSize: "0.82rem", color: "#475569", margin: "8px 0 16px", lineHeight: 1.5 }}>
        Ringkasan ini membaca sinyal 87L sebagai operasi diferensial: arus differential/operate,
        arus restraint/bias, serta urutan trip lokal dan remote. Ini bukan klasifikasi OCR berbasis
        puncak arus fasa biasa.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 8, marginBottom: 16 }}>
        {r.inceptionMs !== null && <MetricCard label="Diff Inception" value={r.inceptionMs.toFixed(1)} unit="ms" />}
        {r.localTripMs !== null && <MetricCard label="Local Trip" value={r.localTripMs.toFixed(1)} unit="ms" highlight />}
        {r.remoteTripMs !== null && <MetricCard label="Remote Trip" value={r.remoteTripMs.toFixed(1)} unit="ms" />}
        {r.tripOutputMs !== null && <MetricCard label="Trip Output" value={r.tripOutputMs.toFixed(1)} unit="ms" highlight />}
        {inceptionToTrip !== null && <MetricCard label="Operate Time" value={inceptionToTrip.toFixed(1)} unit="ms" />}
      </div>

      {r.analogs.length > 0 && (
        <>
          <h3 style={{ fontSize: "0.75rem", color: "#475569", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", margin: "0 0 8px" }}>
            Differential / Restraint Analog Channels
          </h3>
          <div style={{ overflowX: "auto", marginBottom: 16 }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={TH}>Channel</th>
                  <th style={TH}>Role</th>
                  <th style={TH}>Phase</th>
                  <th style={TH}>Peak</th>
                  <th style={TH}>RMS</th>
                  <th style={TH}>First Rise</th>
                </tr>
              </thead>
              <tbody>
                {r.analogs.map((item) => (
                  <tr key={`${item.role}-${item.channel}`}>
                    <td style={{ ...TD, fontWeight: 700, color: "#1e293b" }}>{item.channel}</td>
                    <td style={TD}>
                      <span style={{ color: item.role === "Differential" ? "#1d4ed8" : "#7c3aed", fontWeight: 700 }}>
                        {item.role}
                      </span>
                    </td>
                    <td style={TD}>{item.phase}</td>
                    <td style={TD}><strong>{item.peak.toFixed(item.peak >= 100 ? 1 : 2)}</strong></td>
                    <td style={TD}>{item.rms.toFixed(item.rms >= 100 ? 1 : 2)}</td>
                    <td style={TD}>{item.firstRiseMs !== null ? `${item.firstRiseMs.toFixed(1)} ms` : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {sequenceEvents.length > 0 && (
        <>
          <h3 style={{ fontSize: "0.75rem", color: "#475569", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", margin: "0 0 8px" }}>
            Protection Sequence
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            {sequenceEvents.map((event) => {
              const tag = lineDiffEventTag(event.name);
              return (
                <div
                  key={event.name}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "72px minmax(0, 1fr) auto auto",
                    gap: 8,
                    alignItems: "center",
                    padding: "7px 10px",
                    background: tag === "DIFF" || tag === "LOCAL" ? "#eff6ff" : "#f8fafc",
                    border: "1px solid #e2e8f0",
                    borderRadius: 7,
                  }}
                >
                  <span style={{ color: "#1d4ed8", fontSize: "0.68rem", fontWeight: 800, letterSpacing: "0.08em" }}>{tag}</span>
                  <span style={{ color: "#1e293b", fontSize: "0.82rem", fontWeight: 650, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {event.name}
                  </span>
                  <span style={{ color: "#64748b", fontSize: "0.75rem", fontVariantNumeric: "tabular-nums" }}>
                    {event.activateMs !== null ? `${event.activateMs.toFixed(1)} ms` : "-"}
                  </span>
                  <span style={{ color: "#94a3b8", fontSize: "0.72rem", fontVariantNumeric: "tabular-nums" }}>
                    {event.durationMs !== null ? `${event.durationMs.toFixed(0)} ms` : ""}
                  </span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

export default function FaultRecap87T({ comtrade, relayLabel = "Transformer 87T / REF" }: Props) {
  const r = useMemo(() => analyze(comtrade), [comtrade]);
  if (/87L|LINE DIFFERENTIAL|CCP|STUB DIFFERENTIAL/i.test(relayLabel)) {
    return <LineDifferentialRecap comtrade={comtrade} />;
  }

  const faultColors: Record<string, { bg: string; border: string; text: string }> = {
    "?": { bg: "#f8fafc", border: "#e2e8f0", text: "#64748b" },
    REF: { bg: "#fffbeb", border: "#fbbf24", text: "#b45309" },
    SLG: { bg: "#fff7ed", border: "#fdba74", text: "#c2410c" },
    LL: { bg: "#fef2f2", border: "#fca5a5", text: "#b91c1c" },
    "3Ph": { bg: "#fef2f2", border: "#f87171", text: "#991b1b" },
  };
  const fcKey = r.faultCode.startsWith("REF")
    ? "REF"
    : r.faultCode.startsWith("SLG")
    ? "SLG"
    : r.faultCode.startsWith("LL")
    ? "LL"
    : r.faultCode === "3Ph"
    ? "3Ph"
    : "?";
  const fc = faultColors[fcKey];

  const keyDigEvents = r.digEvents
    .filter((e) => e.type === "trip" || e.type === "ref" || e.type === "diff")
    .sort((a, b) => (a.activateMs ?? Infinity) - (b.activateMs ?? Infinity));

  const alarmEvents = r.digEvents.filter((e) => e.type === "alarm");

  const inceptionToTrip =
    r.inceptionMs !== null && r.tripTimeMs !== null
      ? r.tripTimeMs - r.inceptionMs
      : null;

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Fault Recap — {relayLabel}</h2>
      </div>

      {/* ── Fault type + phase indicators ── */}
      <div className={styles.row} style={{ alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <div
          style={{
            padding: "8px 18px",
            background: fc.bg,
            border: `1.5px solid ${fc.border}`,
            borderRadius: 8,
            color: fc.text,
            fontWeight: 700,
            fontSize: "1rem",
          }}
        >
          {r.faultLabel}
        </div>

        {(["A", "B", "C"] as const).map((ph) => (
          <div
            key={ph}
            title={`Fase ${ph}: peak ${r.phaseMetrics[ph].faultPeak.toFixed(1)} A`}
            style={{
              width: 36,
              height: 36,
              borderRadius: "50%",
              border: "2px solid",
              borderColor: r.phaseMetrics[ph].elevated ? "#ef4444" : "#e2e8f0",
              background: r.phaseMetrics[ph].elevated ? "#fef2f2" : "#f8fafc",
              color: r.phaseMetrics[ph].elevated ? "#dc2626" : "#94a3b8",
              fontWeight: 700,
              fontSize: "0.85rem",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "default",
            }}
          >
            {ph}
          </div>
        ))}

        {r.refOperated && (
          <span
            className={styles.statusBadge}
            style={{ background: "#fffbeb", color: "#b45309", border: "1.5px solid #fbbf24" }}
          >
            REF / 64
          </span>
        )}
      </div>

      <p style={{ fontSize: "0.8rem", color: "#64748b", margin: "8px 0 16px", lineHeight: 1.5 }}>
        {r.faultDescription}
      </p>

      {/* ── Timing metrics ── */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
          gap: 8,
          marginBottom: 16,
        }}
      >
        {r.inceptionMs !== null && (
          <MetricCard label="Inception" value={r.inceptionMs.toFixed(1)} unit="ms" />
        )}
        {r.tripTimeMs !== null && (
          <MetricCard label="Trip" value={r.tripTimeMs.toFixed(1)} unit="ms" highlight />
        )}
        {inceptionToTrip !== null && (
          <MetricCard
            label="Durasi Gangguan"
            value={inceptionToTrip.toFixed(1)}
            unit="ms"
          />
        )}
        {r.tripDurationMs !== null && (
          <MetricCard label="Kontak Trip" value={r.tripDurationMs.toFixed(0)} unit="ms" />
        )}
      </div>

      {/* ── Winding current table ── */}
      {r.windingStats.length > 0 && (
        <>
          <h3
            style={{
              fontSize: "0.75rem",
              color: "#475569",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              margin: "0 0 8px",
            }}
          >
            Arus per Kelompok — Pragangguan (A rms) / Puncak Gangguan (A)
          </h3>
          <div style={{ overflowX: "auto", marginBottom: 16 }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={TH}>Kelompok</th>
                  {(["A", "B", "C"] as const).map((ph) => (
                    <th
                      key={ph}
                      style={{
                        ...TH,
                        color: r.phaseMetrics[ph].elevated ? "#dc2626" : "#475569",
                      }}
                    >
                      Fase {ph}
                      {r.phaseMetrics[ph].elevated && (
                        <span style={{ marginLeft: 4, fontSize: "0.65rem" }}>▲</span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {r.windingStats.map((ws) => (
                  <tr key={ws.prefix}>
                    <td style={{ ...TD, fontWeight: 600, color: "#334155" }}>{ws.label}</td>
                    {(["A", "B", "C"] as const).map((ph) => {
                      const pd = ws.phases[ph];
                      const elevated = r.phaseMetrics[ph].elevated;
                      return (
                        <td
                          key={ph}
                          style={{
                            ...TD,
                            background: elevated ? "#fef2f2" : "transparent",
                            color: elevated ? "#b91c1c" : "#1e293b",
                          }}
                        >
                          {pd ? (
                            <>
                              <span style={{ color: "#94a3b8" }}>
                                {pd.preFaultRms.toFixed(1)}
                              </span>
                              {" / "}
                              <strong>{pd.faultPeak.toFixed(1)}</strong>
                            </>
                          ) : (
                            <span style={{ color: "#cbd5e1" }}>—</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── Protection operations ── */}
      {keyDigEvents.length > 0 && (
        <>
          <h3
            style={{
              fontSize: "0.75rem",
              color: "#475569",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              margin: "0 0 8px",
            }}
          >
            Operasi Proteksi
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 16 }}>
            {keyDigEvents.map((ev) => {
              const colMap = {
                trip: { bg: "#fef2f2", border: "#fca5a5", text: "#b91c1c", tag: "TRIP" },
                ref: { bg: "#fffbeb", border: "#fbbf24", text: "#b45309", tag: "REF" },
                diff: { bg: "#f0f9ff", border: "#bae6fd", text: "#0369a1", tag: "DIFF" },
              } as const;
              const col = colMap[ev.type as keyof typeof colMap] ?? colMap.diff;
              return (
                <div
                  key={ev.name}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "6px 10px",
                    borderRadius: 6,
                    background: col.bg,
                    border: `1px solid ${col.border}`,
                  }}
                >
                  <span
                    style={{
                      fontSize: "0.65rem",
                      fontWeight: 700,
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                      color: col.text,
                      minWidth: 36,
                    }}
                  >
                    {col.tag}
                  </span>
                  <span
                    style={{ fontWeight: 600, fontSize: "0.82rem", color: "#1e293b", flex: 1 }}
                  >
                    {ev.name}
                  </span>
                  {ev.activateMs !== null && (
                    <span
                      style={{
                        fontSize: "0.75rem",
                        color: "#64748b",
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      t = {ev.activateMs.toFixed(1)} ms
                    </span>
                  )}
                  {ev.durationMs !== null && (
                    <span
                      style={{
                        fontSize: "0.72rem",
                        color: "#94a3b8",
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      ({ev.durationMs.toFixed(0)} ms)
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* ── Alarms (collapsed list) ── */}
      {alarmEvents.length > 0 && (
        <div
          style={{
            fontSize: "0.75rem",
            color: "#94a3b8",
            borderTop: "1px solid #f1f5f9",
            paddingTop: 8,
            marginTop: 4,
          }}
        >
          <span style={{ fontWeight: 600, color: "#64748b" }}>Alarm: </span>
          {alarmEvents.map((e) => e.name).join(" · ")}
        </div>
      )}
    </div>
  );
}
