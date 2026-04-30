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
    params: { device_type: "std-30-70", idiff_pickup: 0.20, slope1: 0.30, intersection1: 0.30, slope2: 0.70, intersection2: 2.50, idiff_fast: 7.50 },
  },
  {
    key: "cons-25-50",
    label: "Konservatif — s₁=25%, s₂=50%",
    hint: "Slope lebih rendah; tidak ada zona flat (langsung slope dari 0). Cocok untuk relay generasi lama (Siemens 7UT6x, GE T35).",
    params: { device_type: "cons-25-50", idiff_pickup: 0.20, slope1: 0.25, intersection1: 0.0, slope2: 0.50, intersection2: 2.50, idiff_fast: 7.50 },
  },
  {
    key: "agr-20-80",
    label: "Agresif — s₁=20%, s₂=80%",
    hint: "Slope rendah di tahap 1 (sensitif terhadap gangguan kecil), slope tinggi di tahap 2 (stabil saat arus inrush). Umum pada ABB RET650 / SEL-387.",
    params: { device_type: "agr-20-80", idiff_pickup: 0.20, slope1: 0.20, intersection1: 0.0, slope2: 0.80, intersection2: 2.00, idiff_fast: 7.50 },
  },
  {
    key: "custom",
    label: "Custom",
    hint: "Atur semua parameter secara manual sesuai setting aktual relay di lapangan.",
    params: { device_type: "custom", idiff_pickup: 0.20, slope1: 0.30, intersection1: 0.30, slope2: 0.70, intersection2: 2.50, idiff_fast: 7.50 },
  },
];

interface Sample { t: number; i_diff: number; i_rest: number; phase: string; }

const PHASE_COLORS: Record<string, string> = {
  L1: "#f59e0b",
  L2: "#22c55e",
  L3: "#3b82f6",
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
    } finally {
      setLoading(false);
    }
  }

  const charPts = buildCharacteristic(params);

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

  const layout: Partial<Plotly.Layout> = {
    height: 400,
    margin: { t: 20, b: 50, l: 60, r: 20 },
    xaxis: { title: { text: "I Restraint (p.u.)" }, tickfont: { size: 10 }, range: [0, 10] },
    yaxis: { title: { text: "I Differential (p.u.)" }, tickfont: { size: 10 }, range: [0, params.idiff_fast * 1.1] },
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
    legend: { orientation: "h", y: -0.15 },
    shapes: [
      {
        type: "rect",
        x0: 0, x1: 10,
        y0: charPts[0].y, y1: params.idiff_fast,
        fillcolor: "#fef2f2",
        opacity: 0.3,
        line: { width: 0 },
        layer: "below",
      } as Plotly.Shape,
    ],
  };

  const statusClass = status === "NOT_OPERATED" ? styles.statusNot : status === "IDIFF_FAST_OPERATED" ? styles.statusFast : styles.statusOperated;
  const statusLabel =
    status === "NOT_OPERATED" ? "NOT OPERATED"
    : status === "IDIFF_FAST_OPERATED" ? "I-DIFF FAST OPERATED"
    : "IDIFF OPERATED";

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

      {status && (
        <div className={`${styles.statusBadge} ${statusClass}`} style={{ marginBottom: 12 }}>
          {statusLabel}
          {operatedPhases.length > 0 && ` — Phase ${operatedPhases.join(", ")}`}
        </div>
      )}

      <Plot
        data={[charTrace, fastLine, ...phaseTraces] as Plotly.Data[]}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />

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
      </div>
    </div>
  );
}
