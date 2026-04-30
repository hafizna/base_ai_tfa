import { useState } from "react";
import { aiFaultAnalysis87L } from "../../../api/client";
import styles from "../../panels/Panel.module.css";

interface Props { analysisId: string; }

interface AIResult {
  cause_ranking: { cause: string; label: string; confidence: number }[];
  fault_type: string;
  overall_confidence: number;
  evidence: string[];
}

const DEFAULT_PARAMS = {
  device_type: "SP5",
  idiff_pickup: 0.20,
  slope1: 0.30,
  intersection1: 0.30,
  slope2: 0.70,
  intersection2: 2.50,
  idiff_fast: 7.50,
};

export default function AIFaultAnalysis87L({ analysisId }: Props) {
  const [result, setResult] = useState<AIResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    try {
      const res = await aiFaultAnalysis87L(analysisId, DEFAULT_PARAMS);
      setResult(res);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>AI Fault Analysis — Line Differential (87L)</h2>
      </div>

      <p style={{ fontSize: "0.85rem", color: "#64748b", marginBottom: 16 }}>
        Analyses differential current characteristics to classify internal vs external fault and identify probable cause.
      </p>

      <button className={styles.applyBtn} onClick={run} disabled={loading} style={{ marginBottom: 20 }}>
        {loading ? "Analysing…" : "Run AI Analysis"}
      </button>

      {result && (
        <>
          <div className={styles.row}>
            <span className={`${styles.statusBadge} ${result.fault_type === "permanent" ? styles.statusOperated : styles.statusNot}`}>
              {result.fault_type === "permanent" ? "Permanent / Internal Fault" : "Transient / External Fault"}
            </span>
            <span className={styles.badge}>
              Confidence: {(result.overall_confidence * 100).toFixed(0)}%
            </span>
          </div>

          <h3 style={{ fontSize: "0.85rem", color: "#475569", margin: "12px 0 8px" }}>Classification</h3>
          {result.cause_ranking.map((r) => (
            <div key={r.cause} className={styles.rankingBar}>
              <span className={styles.rankLabel}>{r.label}</span>
              <div className={styles.rankTrack}>
                <div className={styles.rankFill} style={{ width: `${r.confidence * 100}%` }} />
              </div>
              <span className={styles.rankPct}>{(r.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}

          <h3 style={{ fontSize: "0.85rem", color: "#475569", margin: "16px 0 8px" }}>Evidence</h3>
          <ul className={styles.evidenceList}>
            {result.evidence.map((e, i) => (
              <li key={i} className={styles.evidenceItem}>{e}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
