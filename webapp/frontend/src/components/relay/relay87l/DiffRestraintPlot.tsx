import { useState } from "react";
import { diffRestraint87L, diffRestraint87T } from "../../../api/client";
import Plot from "../../plot/PlotlyChart";
import styles from "../../panels/Panel.module.css";

interface Props {
  analysisId: string;
  relayType: "87L" | "87T";
}

interface DiffParams {
  device_type: string;
  idiff_pickup: number;
  slope1: number;
  intersection1: number;
  slope2: number;
  intersection2: number;
  idiff_fast: number;
  in_base_a: number;   // base current In (A); 0 = auto-estimate
}

interface Preset {
  key: string;
  label: string;
  hint: string;
  params: DiffParams;
}

const PRESETS: Preset[] = [
  {
    key: "std-30-70",
    label: "Standar — s₁=30%, s₂=70%",
    hint: "Dual-slope dengan zona pickup flat 0→0.30 pu. Umum pada relay modern (Siemens 7UT8x, ABB RET670, GE T60).",
    params: { device_type: "std-30-70", idiff_pickup: 0.20, slope1: 0.30, intersection1: 0.30, slope2: 0.70, intersection2: 2.50, idiff_fast: 7.50, in_base_a: 0 },
  },
  {
    key: "cons-25-50",
    label: "Konservatif — s₁=25%, s₂=50%",
    hint: "Slope lebih rendah; tidak ada zona flat (langsung slope dari 0). Cocok untuk relay generasi lama (Siemens 7UT6x, GE T35).",
    params: { device_type: "cons-25-50", idiff_pickup: 0.20, slope1: 0.25, intersection1: 0.0, slope2: 0.50, intersection2: 2.50, idiff_fast: 7.50, in_base_a: 0 },
  },
  {
    key: "agr-20-80",
    label: "Agresif — s₁=20%, s₂=80%",
    hint: "Slope rendah di tahap 1 (sensitif terhadap gangguan kecil), slope tinggi di tahap 2 (stabil saat arus inrush). Umum pada ABB RET650 / SEL-387.",
    params: { device_type: "agr-20-80", idiff_pickup: 0.20, slope1: 0.20, intersection1: 0.0, slope2: 0.80, intersection2: 2.00, idiff_fast: 7.50, in_base_a: 0 },
  },
  {
    key: "custom",
    label: "Custom",
    hint: "Atur semua parameter secara manual sesuai setting aktual relay di lapangan.",
    params: { device_type: "custom", idiff_pickup: 0.20, slope1: 0.30, intersection1: 0.30, slope2: 0.70, intersection2: 2.50, idiff_fast: 7.50, in_base_a: 0 },
  },
];

interface Sample { t: number; i_diff: number; i_rest: number; phase: string; }
interface TripMarker {
  kind: "RELAY_TRIP" | "DIFF" | "DIFF_FAST";
  channel_name: string;
  t: number;
  phase: string | null;
  i_diff: number;
  i_rest: number;
}
interface PhaseClassification {
  phase: string;
  verdict: string;
  confidence: string;
  max_idiff: number;
  max_irest: number;
  max_ratio: number;
}

const PHASE_COLORS: Record<string, string> = {
  L1: "#f59e0b",
  L2: "#22c55e",
  L3: "#3b82f6",
};

const TRIP_STYLE: Record<string, { color: string; label: string }> = {
  RELAY_TRIP: { color: "#f97316", label: "Relay TRIP" },
  DIFF: { color: "#6366f1", label: "Diff> TRIP" },
  DIFF_FAST: { color: "#db2777", label: "Diff>> TRIP" },
};

const VERDICT_COLOR: Record<string, string> = {
  "Internal Fault": "#dc2626",
  "Through Fault": "#d97706",
  "Inrush?": "#7c3aed",
  "Not Operated": "#64748b",
};

function buildCharacteristic(p: DiffParams) {
  const maxRest = 10;
  const points: { x: number; y: number }[] = [];

  // Pickup line (flat from 0 to intersection1)
  points.push({ x: 0, y: p.idiff_pickup });
  points.push({ x: p.intersection1, y: p.idiff_pickup });

  // Slope 1
  const y_at_int2 = p.idiff_pickup + p.slope1 * (p.intersection2 - p.intersection1);
  points.push({ x: p.intersection2, y: y_at_int2 });

  // Slope 2
  const y_end = y_at_int2 + p.slope2 * (maxRest - p.intersection2);
  points.push({ x: maxRest, y: y_end });

  return points;
}

export default function DiffRestraintPlot({ analysisId, relayType }: Props) {
  const [selectedPreset, setSelectedPreset] = useState<string>(PRESETS[0].key);
  const [params, setParams] = useState<DiffParams>(PRESETS[0].params);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [operatedPhases, setOperatedPhases] = useState<string[]>([]);
  const [tripMarkers, setTripMarkers] = useState<TripMarker[]>([]);
  const [phaseClass, setPhaseClass] = useState<PhaseClassification[]>([]);
  const [diffMode, setDiffMode] = useState<string>("TWO_TERMINAL");
  const [relayDiffPhases, setRelayDiffPhases] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const activePreset = PRESETS.find((p) => p.key === selectedPreset) ?? PRESETS[0];

  function updateParam(field: keyof DiffParams, val: number | string) {
    setParams((prev) => ({ ...prev, [field]: val, device_type: "custom" }));
    setSelectedPreset("custom");
  }

  function applyPreset(key: string) {
    const preset = PRESETS.find((p) => p.key === key);
    if (!preset) return;
    setSelectedPreset(key);
    if (key !== "custom") setParams(preset.params);
  }

  async function fetchPlot() {
    setLoading(true);
    try {
      const fn = relayType === "87T" ? diffRestraint87T : diffRestraint87L;
      const res = await fn(analysisId, params);
      setSamples(res.samples ?? []);
      setStatus(res.operated_status);
      setOperatedPhases(res.operated_phases ?? []);
      setTripMarkers(res.trip_markers ?? []);
      setPhaseClass(res.phase_classification ?? []);
      setDiffMode(res.diff_data_mode ?? "TWO_TERMINAL");
      setRelayDiffPhases(res.relay_diff_phases ?? []);
    } finally {
      setLoading(false);
    }
  }

  const charPts = buildCharacteristic(params);

  // Operate region = polygon bounded below by the dual-slope characteristic
  // and above by the I-DIFF>> fast line. We draw it as a closed scattergl
  // trace with `fill: "toself"` so it follows the curve instead of sitting
  // as a rectangle behind both operate and restrain zones.
  const operateRegion: Partial<Plotly.ScatterData> = {
    x: [...charPts.map((p) => p.x), 10, 0, charPts[0].x],
    y: [...charPts.map((p) => p.y), params.idiff_fast, params.idiff_fast, charPts[0].y],
    type: "scatter",
    mode: "lines",
    fill: "toself",
    fillcolor: "rgba(254, 226, 226, 0.55)",
    line: { width: 0 },
    name: "Operate region",
    hoverinfo: "skip",
    showlegend: false,
  };

  const charTrace: Partial<Plotly.ScatterData> = {
    x: charPts.map((p) => p.x),
    y: charPts.map((p) => p.y),
    type: "scatter",
    mode: "lines",
    name: "I-DIFF> Characteristic",
    line: { color: "#f59e0b", width: 2 },
  };

  const fastLine: Partial<Plotly.ScatterData> = {
    x: [0, 10],
    y: [params.idiff_fast, params.idiff_fast],
    type: "scatter",
    mode: "lines",
    name: "I-DIFF>>",
    line: { color: "#ef4444", width: 1.5, dash: "dash" },
  };

  const phases = ["L1", "L2", "L3"];
  const phaseTraces: Partial<Plotly.ScatterData>[] = phases.map((ph) => ({
    x: samples.filter((s) => s.phase === ph).map((s) => s.i_rest),
    y: samples.filter((s) => s.phase === ph).map((s) => s.i_diff),
    type: "scatter",
    mode: "markers",
    name: ph,
    marker: { color: PHASE_COLORS[ph], size: 4, opacity: 0.8 },
  }));

  // TRIP markers — one trace per trip kind, placed at the operating point at
  // the trip instant. Square 'T' so it stands out over the sample cloud.
  const tripTraces: Partial<Plotly.ScatterData>[] = Object.entries(TRIP_STYLE)
    .map(([kind, style]) => {
      const ms = tripMarkers.filter((m) => m.kind === kind);
      if (ms.length === 0) return null;
      return {
        x: ms.map((m) => m.i_rest),
        y: ms.map((m) => m.i_diff),
        type: "scatter",
        mode: "text+markers",
        name: style.label,
        text: ms.map(() => "T"),
        textposition: "middle center",
        textfont: { color: "#ffffff", size: 9 },
        hovertext: ms.map((m) => `${style.label}${m.phase ? ` (${m.phase})` : ""} @ ${(m.t * 1000).toFixed(0)} ms — ${m.channel_name}`),
        hoverinfo: "text",
        marker: { color: style.color, size: 16, symbol: "square", line: { color: "#111827", width: 1 } },
      } as Partial<Plotly.ScatterData>;
    })
    .filter((t): t is Partial<Plotly.ScatterData> => t !== null);

  const localOnly = diffMode === "LOCAL_ONLY";

  // In LOCAL_ONLY mode the samples are local-terminal current in primary Amperes,
  // not a p.u. differential — so the p.u. characteristic does not apply and the
  // axes must auto-range to the actual current, otherwise points fall off-screen.
  const maxSampleY = samples.length ? Math.max(...samples.map((s) => s.i_diff)) : 1;
  const maxSampleX = samples.length ? Math.max(...samples.map((s) => s.i_rest)) : 1;

  const layout: Partial<Plotly.Layout> = {
    height: 400,
    margin: { t: 20, b: 50, l: 60, r: 20 },
    xaxis: {
      title: { text: localOnly ? "Arus terminal lokal (A)" : "I Restraint (p.u.)" },
      tickfont: { size: 10 },
      range: localOnly ? [0, maxSampleX * 1.1] : [0, 10],
    },
    yaxis: {
      title: { text: localOnly ? "Arus terminal lokal (A)" : "I Differential (p.u.)" },
      tickfont: { size: 10 },
      range: localOnly ? [0, maxSampleY * 1.1] : [0, params.idiff_fast * 1.1],
    },
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
    legend: { orientation: "h", y: -0.15 },
  };

  const statusClass = status === "NOT_OPERATED" ? styles.statusNot : status === "IDIFF_FAST_OPERATED" ? styles.statusFast : styles.statusOperated;
  // In LOCAL_ONLY the operate verdict cannot come from the reconstructed scatter
  // (it isn't a true differential) — it comes from the relay's own diff trip.
  const statusLabel = localOnly
    ? (relayDiffPhases.length
        ? `RELAY DIFF OPERATED — ${relayDiffPhases.join(", ")}`
        : "RELAY DIFF — TIDAK ADA SINYAL OPERATE")
    : status === "NOT_OPERATED" ? "NOT OPERATED"
    : status === "IDIFF_FAST_OPERATED" ? "I-DIFF FAST OPERATED"
    : "IDIFF OPERATED";
  const localStatusClass = relayDiffPhases.length ? styles.statusFast : styles.statusNot;

  const plotExplanation = samples.length === 0
    ? "Tekan Compute untuk memproses rekaman."
    : localOnly
      ? "Mode LOCAL_ONLY: titik adalah arus terminal LOKAL per fasa (Ampere), bukan differential dua-terminal. Tidak ada arus sisi remote di rekaman ini, jadi kurva operate/restraint p.u. tidak diterapkan."
      : "Setiap titik adalah posisi operasi sesaat dari window RMS pada rekaman: I-diff terhadap I-restraint per fasa. Banyak titik berarti banyak sampel waktu yang diplot, bukan jumlah spike arus.";

  const assessmentText = !status
    ? "Assessment belum dihitung."
    : localOnly
      ? (relayDiffPhases.length
          ? `Assessment: differential dua-terminal tidak dapat direkonstruksi (arus sisi remote tidak direkam). Namun relay 87L sendiri melaporkan elemen differential OPERATE pada fasa ${relayDiffPhases.join(", ")} (dari sinyal DIF_TRIP). Verdict diambil dari keputusan relay, bukan dari waveform lokal. Arus lokal hanya konteks besaran gangguan.`
          : "Assessment: differential dua-terminal tidak dapat direkonstruksi (arus sisi remote tidak direkam), dan tidak ditemukan sinyal DIF_TRIP per fasa dari relay. Tidak ada bukti operate elemen differential pada rekaman ini — periksa sinyal trip lain (Relay TRIP, OC/EF).")
    : status === "NOT_OPERATED"
      ? "Assessment: berdasarkan setting karakteristik yang dipilih, titik operasi masih berada di area restraint/non-operate. Relay diasumsikan tidak seharusnya trip untuk rekaman ini, kecuali ada setting aktual lain yang belum dimasukkan."
      : status === "IDIFF_FAST_OPERATED"
        ? `Assessment: titik operasi menembus elemen I-DIFF>> fast${operatedPhases.length ? ` pada fasa ${operatedPhases.join(", ")}` : ""}. Dengan setting yang diberikan, operasi relay dapat dianggap sesuai karakteristik fast differential.`
        : `Assessment: titik operasi melewati kurva operate I-DIFF>${operatedPhases.length ? ` pada fasa ${operatedPhases.join(", ")}` : ""}. Dengan setting yang diberikan, relay diasumsikan bekerja sesuai karakteristik differential/restraint.`;

  // Hide the p.u. characteristic/region/fast line when they don't apply.
  const baseTraces = localOnly ? [] : [operateRegion, charTrace, fastLine];

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Differential / Restraint Characteristic</h2>
        <div className={styles.controls}>
          <select
            className={styles.selectField}
            value={selectedPreset}
            onChange={(e) => applyPreset(e.target.value)}
            style={{ minWidth: 220 }}
          >
            {PRESETS.map((p) => (
              <option key={p.key} value={p.key}>{p.label}</option>
            ))}
          </select>
          <button className={styles.applyBtn} onClick={fetchPlot} disabled={loading}>
            {loading ? "Computing…" : "Compute"}
          </button>
        </div>
      </div>
      <div style={{ fontSize: "0.72rem", color: "#94a3b8", marginBottom: 10, lineHeight: 1.5 }}>
        {activePreset.hint}
      </div>
      <div className={styles.row} style={{ marginBottom: 12 }}>
        <span className={styles.badge}>{plotExplanation}</span>
      </div>

      {localOnly && samples.length > 0 && (
        <div
          className={styles.row}
          style={{ marginBottom: 12, padding: "8px 12px", background: "#fffbeb", border: "1px solid #fbbf24", borderRadius: 6 }}
        >
          <span style={{ fontSize: "0.74rem", color: "#92400e", lineHeight: 1.5 }}>
            ⚠️ <strong>Differential dua-terminal tidak tersedia.</strong> Rekaman ini hanya berisi arus
            terminal lokal (tidak ada arus sisi remote dan tidak ada channel differential terhitung dari
            relay). Plot di bawah adalah <strong>arus lokal per fasa</strong>, bukan diff/restraint sejati —
            karena itu kurva operate p.u. tidak ditampilkan. Verdict diff diambil dari sinyal trip relay.
          </span>
        </div>
      )}

      {diffMode === "TWO_TERMINAL_RAW" && samples.length > 0 && (
        <div
          className={styles.row}
          style={{ marginBottom: 12, padding: "8px 12px", background: "#ecfdf5", border: "1px solid #34d399", borderRadius: 6 }}
        >
          <span style={{ fontSize: "0.74rem", color: "#065f46", lineHeight: 1.5 }}>
            ✓ <strong>Differential dua-terminal sejati.</strong> Kedua sisi arus terdeteksi otomatis
            (uji fisika: fasa sehat melalui-arus saling meniadakan). Idiff = |I<sub>lokal</sub> + I<sub>remote</sub>|,
            dinormalisasi ke In{params.in_base_a > 0 ? ` = ${params.in_base_a} A (input)` : " (estimasi otomatis — isi In di bawah untuk akurasi)"}.
            Sumbu dalam p.u., kurva operate/restraint berlaku.
          </span>
        </div>
      )}

      {status && (
        <div className={`${styles.statusBadge} ${localOnly ? localStatusClass : statusClass}`} style={{ marginBottom: 12 }}>
          {statusLabel}
          {!localOnly && operatedPhases.length > 0 && ` — Phase ${operatedPhases.join(", ")}`}
          {!localOnly && relayDiffPhases.length > 0 && ` · Relay diff: ${relayDiffPhases.join(", ")}`}
        </div>
      )}

      <div data-pdf-chart-id="diff_restraint" data-pdf-chart-title="Diff / Restraint Plot">
        <Plot
          data={[...baseTraces, ...phaseTraces, ...tripTraces] as Plotly.Data[]}
          layout={layout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
        />
      </div>
      <div className={styles.row} style={{ marginTop: 12 }}>
        <span className={styles.badge}>{assessmentText}</span>
      </div>

      {/* Per-phase classification. In LOCAL_ONLY the ratio-to-threshold is
          meaningless (current in A vs a p.u. characteristic), so we show the
          local fault current as context + the relay's own diff verdict instead. */}
      {phaseClass.length > 0 && (
        <>
          <h3 style={{ fontSize: "0.85rem", color: "#475569", margin: "16px 0 10px" }}>
            {localOnly ? "Ringkasan per Fasa (arus lokal + verdict relay)" : "Klasifikasi per Fasa"}
          </h3>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            {phaseClass.map((c) => {
              const relayOperated = relayDiffPhases.includes(c.phase);
              const verdict = localOnly
                ? (relayOperated ? "Relay Diff OPERATE" : "Relay Diff diam")
                : c.verdict;
              const color = localOnly
                ? (relayOperated ? "#dc2626" : "#64748b")
                : (VERDICT_COLOR[c.verdict] ?? "#334155");
              return (
                <div
                  key={c.phase}
                  style={{
                    flex: "1 1 180px",
                    border: `1px solid ${color}`,
                    borderRadius: 8,
                    padding: "10px 12px",
                    background: "#ffffff",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <strong style={{ color, fontSize: "0.82rem" }}>
                      {c.phase}: {verdict}
                    </strong>
                    {!localOnly && <span className={styles.badge} style={{ fontSize: "0.66rem" }}>{c.confidence}</span>}
                  </div>
                  <div style={{ fontSize: "0.72rem", color: "#475569", lineHeight: 1.6 }}>
                    {localOnly ? (
                      <>
                        Arus lokal puncak: <strong>{c.max_idiff.toFixed(0)}</strong> A<br />
                        Sumber verdict: <strong>{relayOperated ? "sinyal DIF_TRIP relay" : "tidak ada DIF_TRIP"}</strong>
                      </>
                    ) : (
                      <>
                        Max I-diff: <strong>{c.max_idiff.toFixed(2)}</strong> pu<br />
                        Max I-rest: <strong>{c.max_irest.toFixed(2)}</strong> pu<br />
                        Max ratio: <strong>{(c.max_ratio * 100).toFixed(0)}%</strong> dari threshold
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* Parameter editor */}
      <h3 style={{ fontSize: "0.85rem", color: "#475569", margin: "16px 0 10px" }}>Parameters</h3>
      <div className={styles.zoneEditorRow}>
        <label className={styles.zoneLabel}>
          I-DIFF&gt; Pickup (p.u.)
          <input className={styles.inputField} type="number" step={0.01} value={params.idiff_pickup} onChange={(e) => updateParam("idiff_pickup", parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          Slope 1
          <input className={styles.inputField} type="number" step={0.01} value={params.slope1} onChange={(e) => updateParam("slope1", parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          Intersection 1 (p.u.)
          <input className={styles.inputField} type="number" step={0.01} value={params.intersection1} onChange={(e) => updateParam("intersection1", parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          Slope 2
          <input className={styles.inputField} type="number" step={0.01} value={params.slope2} onChange={(e) => updateParam("slope2", parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          Intersection 2 (p.u.)
          <input className={styles.inputField} type="number" step={0.01} value={params.intersection2} onChange={(e) => updateParam("intersection2", parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          I-DIFF&gt;&gt; Fast (p.u.)
          <input className={styles.inputField} type="number" step={0.1} value={params.idiff_fast} onChange={(e) => updateParam("idiff_fast", parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          In Base (A) — 0 = auto
          <input className={styles.inputField} type="number" step={1} value={params.in_base_a} onChange={(e) => updateParam("in_base_a", parseFloat(e.target.value) || 0)} />
        </label>
      </div>
      <div style={{ fontSize: "0.68rem", color: "#94a3b8", marginTop: 6, lineHeight: 1.5 }}>
        In Base = arus nominal (CT primary / rated) untuk normalisasi p.u. pada differential dua-terminal.
        Isi sesuai setting relay aktual agar Idiff/Irest p.u. akurat; biarkan 0 untuk estimasi otomatis dari arus beban prefault.
      </div>
    </div>
  );
}
