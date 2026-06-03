import { useState } from "react";
import { tccMultiStage } from "../../../api/client";
import type { TccStage, TccResult } from "../../../api/client";
import Plot from "../../plot/PlotlyChart";
import styles from "../../panels/Panel.module.css";

interface Props {
  analysisId: string;
  relayType?: "OCR" | "SBEF";
  onReportSettingsChange?: (settings: OCRReportSettings | null) => void;
}

// Kept backward-compatible with the PDF report consumer (Workspace -> generateReport).
export interface OCRReportSettings {
  curve_type: string;
  is_pickup_a: number;
  tms: number;
  measured_current_a: number;
  measured_trip_time_s: number | null;
  intersection_ratio: number | null;
}

const CURVE_LABELS: Record<string, string> = {
  NI: "IEC Normal Inverse",
  VI: "IEC Very Inverse",
  EI: "IEC Extremely Inverse",
  LTI: "IEC Long Time Inverse",
  DT: "Definite Time / INST (moment)",
};

const STAGE_COLORS = ["#16a34a", "#2563eb", "#d97706", "#7c3aed"];
const POINT_COLORS: Record<string, string> = { A: "#ef4444", B: "#2563eb", C: "#16a34a", "N/EF": "#9333ea" };

function defaultStages(mode: "phase" | "ef", domain: "line" | "trafo"): TccStage[] {
  const momentTime = domain === "trafo" ? 0.7 : 0.3;
  if (mode === "ef") {
    return [
      { label: "S1", curve_type: "NI", is_pickup_a: 40, tms: 0.1, definite_time_s: 0 },
      { label: "I>>", curve_type: "DT", is_pickup_a: 400, tms: 0, definite_time_s: momentTime },
    ];
  }
  return [
    { label: "S1", curve_type: "NI", is_pickup_a: 100, tms: 0.1, definite_time_s: 0 },
    { label: "I>>", curve_type: "DT", is_pickup_a: 700, tms: 0, definite_time_s: momentTime },
  ];
}

export default function OvercurrentOverlay({ analysisId, relayType = "OCR", onReportSettingsChange }: Props) {
  const [mode, setMode] = useState<"phase" | "ef">("phase");
  const [domain, setDomain] = useState<"line" | "trafo">("line");
  const [stages, setStages] = useState<TccStage[]>(() => defaultStages("phase", "line"));
  const [result, setResult] = useState<TccResult | null>(null);
  const [loading, setLoading] = useState(false);

  function clearComputedResult() {
    setResult(null);
    onReportSettingsChange?.(null);
  }

  function updateStage(idx: number, patch: Partial<TccStage>) {
    setStages((prev) => prev.map((s, i) => (i === idx ? { ...s, ...patch } : s)));
    clearComputedResult();
  }
  function addStage() {
    setStages((prev) => [
      ...prev,
      { label: `S${prev.length + 1}`, curve_type: "DT", is_pickup_a: 1000, tms: 0, definite_time_s: domain === "trafo" ? 0.7 : 0.3 },
    ]);
    clearComputedResult();
  }
  function removeStage(idx: number) {
    setStages((prev) => prev.filter((_, i) => i !== idx));
    clearComputedResult();
  }
  function switchMode(nextMode: "phase" | "ef") {
    setMode(nextMode);
    setStages(defaultStages(nextMode, domain));
    clearComputedResult();
  }
  function switchDomain(nextDomain: "line" | "trafo") {
    setDomain(nextDomain);
    setStages(defaultStages(mode, nextDomain));
    clearComputedResult();
  }

  async function fetchCurve() {
    setLoading(true);
    try {
      const res = await tccMultiStage(analysisId, mode, domain, stages);
      setResult(res);
      // Feed the PDF report with the highest-current operating phase (back-compat shape).
      const operating = res.fault_points.filter((p) => p.winning_stage_label != null);
      const top = (operating.length ? operating : res.fault_points)
        .reduce((a, b) => (b.current_a > a.current_a ? b : a), res.fault_points[0]);
      if (top) {
        onReportSettingsChange?.({
          curve_type: top.winning_curve_type ?? stages[0]?.curve_type ?? "NI",
          is_pickup_a: stages[0]?.is_pickup_a ?? 0,
          tms: stages[0]?.tms ?? 0,
          measured_current_a: top.current_a,
          measured_trip_time_s: top.trip_time_s,
          intersection_ratio: top.multiple_of_pickup,
        });
      }
    } finally {
      setLoading(false);
    }
  }

  const curveTraces: Partial<Plotly.ScatterData>[] = (result?.curves ?? []).map((c, i) => ({
    x: c.currents_a,
    y: c.trip_times_s,
    type: "scatter",
    mode: "lines",
    name: `${c.label} · ${CURVE_LABELS[c.curve_type] ?? c.curve_type} (Is=${c.is_pickup_a} A)`,
    line: { color: STAGE_COLORS[i % STAGE_COLORS.length], width: 2 },
  }));

  const pointTraces: Partial<Plotly.ScatterData>[] = (result?.fault_points ?? [])
    .filter((p) => p.winning_stage_label != null && p.trip_time_s != null)
    .map((p) => ({
      x: [p.current_a],
      y: [p.trip_time_s ?? 0],
      type: "scatter",
      mode: "text+markers",
      name: `${p.channel_label}: ${p.current_a.toFixed(0)} A → ${p.winning_stage_label} ${p.is_moment ? "INST" : ""}`,
      text: [p.channel_label],
      textposition: "top center",
      marker: {
        color: POINT_COLORS[p.channel_label] ?? "#ef4444",
        size: 12,
        symbol: p.is_moment ? "x" : "circle",
        line: { color: "#111827", width: 1 },
      },
    }));

  const allCurrents = (result?.curves ?? []).flatMap((c) => c.currents_a);
  const xMin = allCurrents.length ? Math.max(1, Math.min(...allCurrents) * 0.8) : 10;
  const xMax = allCurrents.length ? Math.max(...allCurrents) * 1.05 : 10000;

  const layout: Partial<Plotly.Layout> = {
    height: 420,
    margin: { t: 20, b: 50, l: 60, r: 20 },
    xaxis: {
      title: { text: "Arus gangguan (A)" },
      type: "log",
      range: [Math.log10(xMin), Math.log10(xMax)],
      tickfont: { size: 10 },
    },
    yaxis: { title: { text: "Waktu kerja (s)" }, type: "log", tickfont: { size: 10 } },
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
    legend: { orientation: "h", y: -0.18, font: { size: 10 } },
  };

  const panelTitle = relayType === "SBEF" ? "SBEF Timing Characteristic" : "TCC — Time-Current Characteristic";

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>{panelTitle}</h2>
        <div className={styles.controls}>
          <button
            className={mode === "phase" ? styles.applyBtn : styles.selectField}
            onClick={() => switchMode("phase")}
          >
            Phase (OCR)
          </button>
          <button
            className={mode === "ef" ? styles.applyBtn : styles.selectField}
            onClick={() => switchMode("ef")}
          >
            EF / Ground (GFR)
          </button>
        </div>
      </div>

      <div className={styles.row} style={{ marginBottom: 12 }}>
        <label className={styles.label}>Objek</label>
        <select
          className={styles.selectField}
          value={domain}
          onChange={(e) => switchDomain(e.target.value as "line" | "trafo")}
        >
          <option value="line">Penghantar / Line</option>
          <option value="trafo">Trafo</option>
        </select>
        <span className={styles.badge}>
          Sumbu X = arus gangguan absolut (A); Y = waktu kerja (s). Stage tercepat yang menang menentukan trip.
        </span>
      </div>

      {/* Stage editor — mirrors a physical relay setting sheet (elemen waktu + I>> moment). */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 12 }}>
        {stages.map((s, i) => (
          <div key={i} className={styles.row} style={{ alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <input
              className={styles.inputField}
              style={{ width: 56 }}
              value={s.label}
              onChange={(e) => updateStage(i, { label: e.target.value })}
            />
            <select
              className={styles.selectField}
              value={s.curve_type}
              onChange={(e) => updateStage(i, { curve_type: e.target.value })}
            >
              {Object.entries(CURVE_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
            <label className={styles.label}>Is (A)</label>
            <input
              className={styles.inputField}
              style={{ width: 80 }}
              type="number"
              step={1}
              value={s.is_pickup_a}
              onChange={(e) => updateStage(i, { is_pickup_a: parseFloat(e.target.value) })}
            />
            {s.curve_type === "DT" ? (
              <>
                <label className={styles.label}>t (s)</label>
                <input
                  className={styles.inputField}
                  style={{ width: 70 }}
                  type="number"
                  step={0.01}
                  value={s.definite_time_s}
                  onChange={(e) => updateStage(i, { definite_time_s: parseFloat(e.target.value) })}
                />
              </>
            ) : (
              <>
                <label className={styles.label}>TMS</label>
                <input
                  className={styles.inputField}
                  style={{ width: 70 }}
                  type="number"
                  step={0.01}
                  value={s.tms}
                  onChange={(e) => updateStage(i, { tms: parseFloat(e.target.value) })}
                />
              </>
            )}
            <button className={styles.selectField} onClick={() => removeStage(i)} disabled={stages.length <= 1}>
              ✕
            </button>
          </div>
        ))}
        <div className={styles.row}>
          <button className={styles.selectField} onClick={addStage}>+ Add Stage</button>
          <button className={styles.applyBtn} onClick={fetchCurve} disabled={loading || stages.length === 0}>
            {loading ? "Computing…" : "Plot TCC"}
          </button>
        </div>
      </div>

      <div
        data-pdf-chart-id="overcurrent_overlay"
        data-pdf-chart-title="TCC Overlay (multi-stage)"
        data-pdf-ready={result ? "true" : "false"}
      >
        <Plot
          data={[...curveTraces, ...pointTraces] as Plotly.Data[]}
          layout={layout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
        />
      </div>

      {result && (
        <>
          <div className={styles.row} style={{ marginTop: 12, flexWrap: "wrap", gap: 6 }}>
            {result.fault_points.map((p) => (
              <span
                key={p.channel_label}
                className={styles.badge}
                style={p.is_moment ? { borderColor: "#ef4444", color: "#b91c1c" } : undefined}
              >
                {p.channel_label}: {p.current_a.toFixed(0)} A
                {p.winning_stage_label
                  ? ` → ${p.winning_stage_label}${p.is_moment ? " INST" : ""}` +
                    (p.multiple_of_pickup ? ` (${p.multiple_of_pickup.toFixed(1)}× Is` : "") +
                    (p.trip_time_s != null ? `, ${p.is_moment ? `${(p.trip_time_s * 1000).toFixed(0)} ms` : `${p.trip_time_s.toFixed(2)} s`})` : ")")
                  : " → di bawah pickup"}
              </span>
            ))}
          </div>
          <div className={styles.row} style={{ marginTop: 10 }}>
            <span className={styles.badge} style={{ whiteSpace: "normal", lineHeight: 1.5 }}>
              {result.assessment}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
