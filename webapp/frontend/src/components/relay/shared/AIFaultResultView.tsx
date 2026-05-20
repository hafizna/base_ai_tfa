import { useMemo, useState, type CSSProperties } from "react";

import styles from "../../panels/Panel.module.css";

export interface AICauseRank {
  cause: string;
  label: string;
  confidence: number;
}

export type AIEvidenceSeverity = "verdict" | "critical" | "warning" | "notable" | "info";

export interface AIEvidenceItem {
  text: string;
  severity?: AIEvidenceSeverity;
  weight?: number;
  kind?: string;
}

export interface AIAppliedCap {
  name: string;
  before: number;
  after: number;
  reason?: string;
}

export interface AITier1Info {
  fired: boolean;
  rule_name?: string;
  label?: string;
  confidence?: number;
  evidence?: string;
}

export interface AIFaultResult {
  cause_ranking: AICauseRank[];
  fault_type: string;
  overall_confidence: number;
  evidence: Array<string | AIEvidenceItem>;
  tier1?: AITier1Info | null;
  raw_probabilities?: Record<string, number> | null;
  calibrated_probabilities?: Record<string, number> | null;
  applied_caps?: AIAppliedCap[];
  feature_vector_used?: Record<string, number> | null;
  meta?: Record<string, unknown> | null;
}

export interface AIApiTrace {
  method?: string;
  endpoint: string;
  requestPayload: unknown;
  responsePayload: unknown;
  startedAt?: string;
  durationMs?: number;
  status?: number;
}

interface Props {
  result: AIFaultResult;
  classificationTitle?: string;
  evidenceTitle?: string;
  permanentLabel?: string;
  transientLabel?: string;
  apiTrace?: AIApiTrace;
}

const ACCENTS = ["#2563eb", "#0891b2", "#7c3aed", "#ea580c", "#16a34a", "#dc2626", "#475569"];

function clampPct(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value * 100));
}

function pct(value: number) {
  return `${clampPct(value).toFixed(0)}%`;
}

function formatJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

async function copyToClipboard(text: string) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    /* clipboard unavailable — ignore silently */
  }
}

export default function AIFaultResultView({
  result,
  classificationTitle = "Cause Fingerprint",
  evidenceTitle = "Diagnostic Notes",
  permanentLabel = "Permanent fault behavior",
  transientLabel = "Transient fault behavior",
  apiTrace,
}: Props) {
  const topCause = result.cause_ranking[0];
  const isPermanent = result.fault_type === "permanent";
  const confidence = clampPct(result.overall_confidence);
  const [inspectorOpen, setInspectorOpen] = useState(false);

  const requestJson = useMemo(() => formatJson(apiTrace?.requestPayload ?? null), [apiTrace?.requestPayload]);
  const responseJson = useMemo(() => formatJson(apiTrace?.responsePayload ?? result), [apiTrace?.responsePayload, result]);

  return (
    <div className={styles.aiResultLayout}>
      <section className={styles.aiVerdictStrip}>
        <div className={styles.aiVerdictMain}>
          <span className={styles.aiEyebrow}>Primary pattern</span>
          <div className={styles.aiVerdictTitle}>{topCause?.label ?? "No dominant cause"}</div>
          <div className={styles.aiVerdictMeta}>
            <span className={isPermanent ? styles.aiFaultPermanent : styles.aiFaultTransient}>
              {isPermanent ? permanentLabel : transientLabel}
            </span>
            <span>{result.cause_ranking.length} ranked candidates</span>
          </div>
        </div>

        <div
          className={styles.aiConfidenceDial}
          style={{ "--confidence": `${confidence}%` } as CSSProperties}
          aria-label={`Overall confidence ${confidence.toFixed(0)}%`}
        >
          <span>{confidence.toFixed(0)}%</span>
          <small>AI confidence</small>
        </div>
      </section>

      {result.cause_ranking.length > 0 && (
        <section>
          <h3 className={styles.aiSectionTitle}>{classificationTitle}</h3>
          <div className={styles.aiCauseGrid}>
            {result.cause_ranking.map((item, index) => {
              const score = clampPct(item.confidence);
              const accent = ACCENTS[index % ACCENTS.length];
              return (
                <article
                  key={item.cause}
                  className={styles.aiCauseTile}
                  style={{ "--score": `${score}%`, "--accent": accent } as CSSProperties}
                >
                  <div className={styles.aiCauseTileTop}>
                    <span>{item.label}</span>
                    <strong>{pct(item.confidence)}</strong>
                  </div>
                  <div className={styles.aiCauseMeter} />
                </article>
              );
            })}
          </div>
        </section>
      )}

      {result.evidence.length > 0 && (
        <section>
          <h3 className={styles.aiSectionTitle}>{evidenceTitle}</h3>
          <div className={styles.aiEvidenceGrid}>
            {result.evidence.map((raw, index) => {
              const item: AIEvidenceItem =
                typeof raw === "string" ? { text: raw, severity: "info" } : raw;
              const severity = item.severity ?? "info";
              const weight = typeof item.weight === "number" ? item.weight : undefined;
              return (
                <article
                  key={index}
                  className={styles.aiEvidenceCard}
                  data-severity={severity}
                >
                  <span className={styles.aiEvidenceIndex}>{String(index + 1).padStart(2, "0")}</span>
                  <div className={styles.aiEvidenceBody}>
                    <div className={styles.aiEvidenceHead}>
                      <span className={styles.aiEvidenceBadge} data-severity={severity}>{severity}</span>
                      {item.kind && <span className={styles.aiEvidenceKind}>{item.kind}</span>}
                      {weight !== undefined && (
                        <span className={styles.aiEvidenceWeight} title={`weight ${weight.toFixed(2)}`}>
                          w {weight.toFixed(2)}
                        </span>
                      )}
                    </div>
                    <p>{item.text}</p>
                  </div>
                </article>
              );
            })}
          </div>
        </section>
      )}

      {(result.tier1?.fired || (result.applied_caps && result.applied_caps.length > 0) || result.meta) && (
        <section className={styles.aiProvenance}>
          <h3 className={styles.aiSectionTitle}>Provenance &amp; Model Metadata</h3>

          {result.tier1?.fired && (
            <div className={styles.aiProvenanceBlock} data-tone="rule">
              <strong>Tier 1 rule fired:</strong> <code>{result.tier1.rule_name}</code> → {result.tier1.label}
              {typeof result.tier1.confidence === "number" && ` (conf ${pct(result.tier1.confidence)})`}
              {result.tier1.evidence && <div className={styles.aiProvenanceNote}>{result.tier1.evidence}</div>}
            </div>
          )}

          {result.applied_caps && result.applied_caps.length > 0 && (
            <div className={styles.aiProvenanceBlock} data-tone="cap">
              <strong>Confidence caps applied ({result.applied_caps.length}):</strong>
              <ul className={styles.aiProvenanceList}>
                {result.applied_caps.map((cap, i) => (
                  <li key={i}>
                    <code>{cap.name}</code>: {pct(cap.before)} → {pct(cap.after)}
                    {cap.reason && <span className={styles.aiProvenanceNote}> — {cap.reason}</span>}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.meta && (
            <div className={styles.aiProvenanceBlock} data-tone="meta">
              <strong>Model:</strong>{" "}
              <code>{String((result.meta as Record<string, unknown>).model_version ?? "unknown")}</code>
              {" · "}
              <strong>Calibration:</strong>{" "}
              <code>{String((result.meta as Record<string, unknown>).calibration_method_used ?? (result.meta as Record<string, unknown>).calibration ?? "—")}</code>
              {" · "}
              <strong>Feature schema:</strong>{" "}
              <code>{String((result.meta as Record<string, unknown>).feature_version ?? "—")}</code>
            </div>
          )}
        </section>
      )}

      <section className={styles.aiInspector}>
        <button
          type="button"
          className={styles.aiInspectorToggle}
          onClick={() => setInspectorOpen((open) => !open)}
          aria-expanded={inspectorOpen}
        >
          <span className={styles.aiInspectorChevron}>{inspectorOpen ? "▾" : "▸"}</span>
          API &amp; JSON Inspector
          {apiTrace?.endpoint && <code className={styles.aiInspectorEndpoint}>{apiTrace.method ?? "POST"} {apiTrace.endpoint}</code>}
        </button>

        {inspectorOpen && (
          <div className={styles.aiInspectorBody}>
            <div className={styles.aiInspectorMeta}>
              {apiTrace?.endpoint && (
                <span><strong>Endpoint:</strong> <code>{apiTrace.method ?? "POST"} {apiTrace.endpoint}</code></span>
              )}
              {apiTrace?.status != null && <span><strong>HTTP:</strong> {apiTrace.status}</span>}
              {apiTrace?.durationMs != null && <span><strong>Latency:</strong> {apiTrace.durationMs.toFixed(0)} ms</span>}
              {apiTrace?.startedAt && <span><strong>Called:</strong> {apiTrace.startedAt}</span>}
            </div>

            <div className={styles.aiInspectorGrid}>
              <div className={styles.aiInspectorBlock}>
                <div className={styles.aiInspectorBlockHead}>
                  <span>Request payload</span>
                  <button
                    type="button"
                    className={styles.aiInspectorCopy}
                    onClick={() => void copyToClipboard(requestJson)}
                  >
                    Copy
                  </button>
                </div>
                <pre className={styles.aiInspectorPre}>{requestJson}</pre>
              </div>

              <div className={styles.aiInspectorBlock}>
                <div className={styles.aiInspectorBlockHead}>
                  <span>Response payload</span>
                  <button
                    type="button"
                    className={styles.aiInspectorCopy}
                    onClick={() => void copyToClipboard(responseJson)}
                  >
                    Copy
                  </button>
                </div>
                <pre className={styles.aiInspectorPre}>{responseJson}</pre>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
