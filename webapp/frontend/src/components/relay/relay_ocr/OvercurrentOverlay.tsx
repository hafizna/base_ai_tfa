import { useState } from "react";
import { overCurrentCharacteristic } from "../../../api/client";
import Plot from "../../plot/PlotlyChart";
import styles from "../../panels/Panel.module.css";

interface Props {
  analysisId: string;
  relayType?: "OCR" | "SBEF";
}
interface CurvePoint { current_ratio: number; trip_time_s: number; }
interface OCRResult {
  curve_points: CurvePoint[];
  measured_current_a: number;
  measured_trip_time_s: number | null;
  intersection_ratio: number | null;
}

const CURVE_LABELS: Record<string, string> = {
  NI: "Normal Inverse (IEC)",
  VI: "Very Inverse (IEC)",
  EI: "Extremely Inverse (IEC)",
  LTI: "Long Time Inverse",
};

export default function OvercurrentOverlay({ analysisId, relayType = "OCR" }: Props) {
  const [curveType, setCurveType] = useState("NI");
  const [pickup, setPickup] = useState(1.0);
  const [tms, setTms] = useState(0.1);
  const [result, setResult] = useState<OCRResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function fetchCurve() {
    setLoading(true);
    try {
      const res = await overCurrentCharacteristic(analysisId, curveType, pickup, tms);
      setResult(res);
    } finally {
      setLoading(false);
    }
  }

  const curveTrace: Partial<Plotly.ScatterData> = result ? {
    x: result.curve_points.map((p) => p.current_ratio * pickup),
    y: result.curve_points.map((p) => p.trip_time_s),
    type: "scatter",
    mode: "lines",
    name: CURVE_LABELS[curveType],
    line: { color: "#3b82f6", width: 2 },
  } : { x: [], y: [], type: "scatter", mode: "lines", name: "No data" };

  const measuredTrace: Partial<Plotly.ScatterData> = result?.measured_current_a ? {
    x: [result.measured_current_a],
    y: [result.measured_trip_time_s ?? 0],
    type: "scatter",
    mode: "markers",
    name: `Measured: ${result.measured_current_a.toFixed(1)} A`,
    marker: { color: "#ef4444", size: 12, symbol: "x" },
  } : { x: [], y: [], type: "scatter", mode: "markers", name: "" };

  const layout: Partial<Plotly.Layout> = {
    height: 380,
    margin: { t: 20, b: 50, l: 60, r: 20 },
    xaxis: { title: { text: "Current (A)" }, type: "log", tickfont: { size: 10 } },
    yaxis: { title: { text: "Trip Time (s)" }, type: "log", tickfont: { size: 10 } },
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
    legend: { orientation: "h", y: -0.15 },
  };

  const panelTitle = relayType === "SBEF"
    ? "SBEF Timing Characteristic"
    : "Overcurrent / GFR Characteristic";

  const panelNote = relayType === "SBEF"
    ? "SBEF is separated here so its timing review is not mixed with standard OCR/GFR assumptions."
    : "Use this view for OCR and GFR relays that operate through pickup and time-delay behavior.";

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>{panelTitle}</h2>
        <div className={styles.controls}>
          <select className={styles.selectField} value={curveType} onChange={(e) => setCurveType(e.target.value)}>
            {Object.entries(CURVE_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
        </div>
      </div>

      <div className={styles.row} style={{ marginBottom: 12 }}>
        <span className={styles.badge}>{panelNote}</span>
      </div>

      <div className={styles.row}>
        <label className={styles.label}>Is Pickup (A)</label>
        <input className={styles.inputField} type="number" step={0.1} value={pickup} onChange={(e) => setPickup(parseFloat(e.target.value))} />
        <label className={styles.label}>TMS</label>
        <input className={styles.inputField} type="number" step={0.01} value={tms} onChange={(e) => setTms(parseFloat(e.target.value))} />
        <button className={styles.applyBtn} onClick={fetchCurve} disabled={loading}>
          {loading ? "Computing…" : "Plot Curve"}
        </button>
      </div>

      <Plot
        data={[curveTrace, measuredTrace] as Plotly.Data[]}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />

      {result && result.intersection_ratio != null && (
        <div className={styles.row} style={{ marginTop: 12 }}>
          <span className={styles.badge}>
            Measured: {result.measured_current_a.toFixed(1)} A
            ({result.intersection_ratio.toFixed(2)}× Is)
          </span>
          {result.measured_trip_time_s != null && (
            <span className={styles.badge}>
              Trip time: {result.measured_trip_time_s.toFixed(3)} s
            </span>
          )}
        </div>
      )}
    </div>
  );
}
