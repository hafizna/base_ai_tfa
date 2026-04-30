import { useEffect, useState } from "react";

import { aiFaultAnalysis21, extractFeatures21 } from "../../../api/client";
import styles from "../../panels/Panel.module.css";

interface Props {
  analysisId: string;
  dataRevision?: number;
}

interface Features {
  fault_inception_angle_deg: number;
  fault_duration_ms: number;
  prefault_load_a: number;
  impedance_at_trip_ohm: number;
  waveform_asymmetry: number;
  dc_offset: number;
  ar_result: "successful" | "failed" | null | "";
}

interface AIResult {
  cause_ranking: { cause: string; label: string; confidence: number }[];
  fault_type: string;
  overall_confidence: number;
  evidence: string[];
}

const EMPTY_FEATURES: Features = {
  fault_inception_angle_deg: 0,
  fault_duration_ms: 0,
  prefault_load_a: 0,
  impedance_at_trip_ohm: 0,
  waveform_asymmetry: 0,
  dc_offset: 0,
  ar_result: null,
};

export default function AIFaultAnalysis21({ analysisId, dataRevision = 0 }: Props) {
  const [features, setFeatures] = useState<Features>(EMPTY_FEATURES);
  const [extracting, setExtracting] = useState(true);
  const [result, setResult] = useState<AIResult | null>(null);
  const [running, setRunning] = useState(false);
  const [autoRan, setAutoRan] = useState(false);

  useEffect(() => {
    setExtracting(true);
    setAutoRan(false);
    setResult(null);

    extractFeatures21(analysisId)
      .then((data) => setFeatures({ ...data, ar_result: data.ar_result ?? "" }))
      .catch(() => setFeatures(EMPTY_FEATURES))
      .finally(() => setExtracting(false));
  }, [analysisId, dataRevision]);

  useEffect(() => {
    if (extracting || autoRan) return;
    void run(features, true);
  }, [extracting, autoRan, features]);

  async function run(nextFeatures = features, silent = false) {
    setRunning(true);
    try {
      const response = await aiFaultAnalysis21(analysisId, {
        ...nextFeatures,
        ar_result: nextFeatures.ar_result || null,
      });
      setResult(response);
    } catch {
      if (!silent) {
        setResult({
          fault_type: "transient",
          cause_ranking: [],
          overall_confidence: 0,
          evidence: ["AI analysis request failed."],
        });
      }
    } finally {
      setRunning(false);
      if (silent) setAutoRan(true);
    }
  }

  const importFailed = result?.evidence.some((item) => item.toLowerCase().includes("model imports gagal")) ?? false;

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>AI Fault Analysis - Distance (21)</h2>
        {extracting && <span className={styles.badge}>Extracting features...</span>}
        {!extracting && (
          <span className={styles.badge} style={{ background: "#f0fdf4", color: "#16a34a" }}>
            Auto-extracted from COMTRADE
          </span>
        )}
        <button
          type="button"
          onClick={() => void run()}
          disabled={running || extracting}
          style={{
            marginLeft: "auto",
            background: "none",
            border: "none",
            color: running ? "#94a3b8" : "#3b82f6",
            fontSize: "0.78rem",
            cursor: running ? "default" : "pointer",
            padding: 0,
            textDecoration: "underline",
          }}
        >
          {running ? "Menganalisis..." : "Jalankan ulang analisis"}
        </button>
      </div>

      {importFailed && (
        <div className={styles.warning} style={{ marginBottom: 12 }}>
          Backend AI model failed to load. This usually means the API Python environment is missing
          `lightgbm`.
        </div>
      )}

      {result && (
        <>
          <div className={styles.row}>
            {result.cause_ranking.length > 0 && (
              <span
                className={styles.statusBadge}
                style={
                  result.fault_type === "permanent"
                    ? { background: "#fef2f2", color: "#b91c1c", border: "1.5px solid #fca5a5" }
                    : { background: "#f0fdf4", color: "#15803d", border: "1.5px solid #86efac" }
                }
              >
                {result.cause_ranking[0].label}
              </span>
            )}
            <span
              className={styles.badge}
              style={
                result.fault_type === "permanent"
                  ? { background: "#fef2f2", color: "#dc2626" }
                  : { background: "#f0fdf4", color: "#16a34a" }
              }
            >
              {result.fault_type === "permanent" ? "Permanen" : "Transien"}
            </span>
            <span className={styles.badge}>Keyakinan: {(result.overall_confidence * 100).toFixed(0)}%</span>
          </div>

          {result.cause_ranking.length > 0 && (
            <>
              <h3 style={{ fontSize: "0.85rem", color: "#475569", margin: "14px 0 8px" }}>Cause Ranking</h3>
              {result.cause_ranking.map((item) => (
                <div key={item.cause} className={styles.rankingBar}>
                  <span className={styles.rankLabel}>{item.label}</span>
                  <div style={{ flex: 1, background: "#f1f5f9", height: 10, borderRadius: 5, overflow: "hidden" }}>
                    <div className={styles.rankFill} style={{ width: `${item.confidence * 100}%` }} />
                  </div>
                  <span className={styles.rankPct}>{(item.confidence * 100).toFixed(0)}%</span>
                </div>
              ))}
            </>
          )}

          <h3 style={{ fontSize: "0.85rem", color: "#475569", margin: "16px 0 8px" }}>Evidence</h3>
          <ul className={styles.evidenceList}>
            {result.evidence.map((item, index) => (
              <li key={index} className={styles.evidenceItem}>
                {item}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
