import { useState } from "react";

import { aiFaultAnalysis87L } from "../../../api/client";
import styles from "../../panels/Panel.module.css";
import AIFaultResultView, { type AIFaultResult } from "../shared/AIFaultResultView";

interface Props {
  analysisId: string;
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
  const [result, setResult] = useState<AIFaultResult | null>(null);
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
        <h2 className={styles.panelTitle}>AI Fault Analysis - Line Differential (87L)</h2>
      </div>

      <p style={{ fontSize: "0.85rem", color: "#64748b", marginBottom: 16 }}>
        Analyses differential current characteristics to classify internal vs external fault and identify probable cause.
      </p>

      <button className={styles.applyBtn} onClick={run} disabled={loading} style={{ marginBottom: 20 }}>
        {loading ? "Analysing..." : "Run AI Analysis"}
      </button>

      {result && (
        <AIFaultResultView
          result={result}
          classificationTitle="Differential Classification"
          permanentLabel="Permanent / internal fault"
          transientLabel="Transient / external fault"
        />
      )}
    </div>
  );
}
