import type { CSSProperties } from "react";

import styles from "../../panels/Panel.module.css";

export interface AICauseRank {
  cause: string;
  label: string;
  confidence: number;
}

export interface AIFaultResult {
  cause_ranking: AICauseRank[];
  fault_type: string;
  overall_confidence: number;
  evidence: string[];
}

interface Props {
  result: AIFaultResult;
  classificationTitle?: string;
  evidenceTitle?: string;
  permanentLabel?: string;
  transientLabel?: string;
}

const ACCENTS = ["#2563eb", "#0891b2", "#7c3aed", "#ea580c", "#16a34a", "#dc2626", "#475569"];

function clampPct(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value * 100));
}

function pct(value: number) {
  return `${clampPct(value).toFixed(0)}%`;
}

export default function AIFaultResultView({
  result,
  classificationTitle = "Cause Fingerprint",
  evidenceTitle = "Diagnostic Notes",
  permanentLabel = "Permanent fault behavior",
  transientLabel = "Transient fault behavior",
}: Props) {
  const topCause = result.cause_ranking[0];
  const isPermanent = result.fault_type === "permanent";
  const confidence = clampPct(result.overall_confidence);

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
            {result.evidence.map((item, index) => (
              <article key={index} className={styles.aiEvidenceCard}>
                <span className={styles.aiEvidenceIndex}>{String(index + 1).padStart(2, "0")}</span>
                <p>{item}</p>
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
