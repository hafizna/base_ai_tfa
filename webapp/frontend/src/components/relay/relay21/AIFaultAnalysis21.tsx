import { useEffect, useState } from "react";

import { aiFaultAnalysis21, extractFeatures21 } from "../../../api/client";
import styles from "../../panels/Panel.module.css";
import AIFaultResultView, { type AIApiTrace, type AIFaultResult } from "../shared/AIFaultResultView";

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
  const [result, setResult] = useState<AIFaultResult | null>(null);
  const [running, setRunning] = useState(false);
  const [autoRan, setAutoRan] = useState(false);
  const [apiTrace, setApiTrace] = useState<AIApiTrace | null>(null);

  useEffect(() => {
    setExtracting(true);
    setAutoRan(false);
    setResult(null);
    setApiTrace(null);

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
    const requestPayload = {
      analysis_id: analysisId,
      ...nextFeatures,
      ar_result: nextFeatures.ar_result || null,
    };
    const startedAt = new Date();
    const t0 = performance.now();
    try {
      const response = await aiFaultAnalysis21(analysisId, {
        ...nextFeatures,
        ar_result: nextFeatures.ar_result || null,
      });
      setResult(response);
      setApiTrace({
        method: "POST",
        endpoint: "/api/analyze/21/ai-analysis",
        requestPayload,
        responsePayload: response,
        startedAt: startedAt.toISOString(),
        durationMs: performance.now() - t0,
        status: 200,
      });
    } catch (err) {
      const fallback: AIFaultResult = {
        fault_type: "transient",
        cause_ranking: [],
        overall_confidence: 0,
        evidence: [{ text: "AI analysis request failed.", severity: "critical" }],
      };
      if (!silent) setResult(fallback);
      setApiTrace({
        method: "POST",
        endpoint: "/api/analyze/21/ai-analysis",
        requestPayload,
        responsePayload: { error: String(err) },
        startedAt: startedAt.toISOString(),
        durationMs: performance.now() - t0,
      });
    } finally {
      setRunning(false);
      if (silent) setAutoRan(true);
    }
  }

  const importFailed = result?.evidence.some((item) => {
    const text = typeof item === "string" ? item : item.text ?? "";
    return text.toLowerCase().includes("model imports gagal");
  }) ?? false;

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
        <AIFaultResultView
          result={result}
          permanentLabel="Permanen"
          transientLabel="Transien"
          apiTrace={apiTrace ?? undefined}
        />
      )}
    </div>
  );
}
