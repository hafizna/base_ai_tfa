import type { ReconstructionOut } from "../../api/client";
import styles from "./NarrativePanel.module.css";

interface Props {
  reconstruction: ReconstructionOut;
}

export default function NarrativePanel({ reconstruction }: Props) {
  const rootCauseConfirmed = reconstruction.physical_cause_evidence.incident_root_cause !== "UNCONFIRMED";

  return (
    <div className={styles.wrap}>
      <div className={styles.narrativeBlock}>
        <span className={styles.label}>Narrative (deterministic, generated from structured reconstruction data)</span>
        <p className={styles.narrativeText}>{reconstruction.narrative}</p>
      </div>

      <div className={styles.rootCauseStatus}>
        Root cause confirmation status:{" "}
        <strong className={rootCauseConfirmed ? styles.confirmed : styles.unconfirmed}>
          {reconstruction.physical_cause_evidence.incident_root_cause}
        </strong>
      </div>

      <div className={styles.section}>
        <span className={styles.label}>Incident hypotheses</span>
        {reconstruction.incident_hypotheses.length === 0 ? (
          <p className={styles.muted}>No incident-level hypotheses raised.</p>
        ) : (
          <div className={styles.hypothesesList}>
            {reconstruction.incident_hypotheses.map((h, i) => (
              <div key={i} className={styles.hypothesisCard}>
                <div className={styles.hypothesisHeader}>
                  <strong>{h.hypothesis.replace(/_/g, " ")}</strong>
                  <span>{Math.round(h.confidence * 100)}% confidence</span>
                </div>
                <div className={styles.evidenceCols}>
                  <div>
                    <span className={styles.evidenceLabel}>Evidence for</span>
                    <ul>{h.evidence_for.map((e, j) => <li key={j}>{e}</li>)}</ul>
                  </div>
                  <div>
                    <span className={styles.evidenceLabel}>Evidence against</span>
                    {h.evidence_against.length === 0 ? (
                      <p className={styles.muted}>None</p>
                    ) : (
                      <ul>{h.evidence_against.map((e, j) => <li key={j}>{e}</li>)}</ul>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className={styles.section}>
        <span className={styles.label}>Clock / alignment caveats</span>
        {reconstruction.alignment.assumptions.length === 0 && reconstruction.alignment.warnings.length === 0 ? (
          <p className={styles.muted}>No caveats raised.</p>
        ) : (
          <>
            {reconstruction.alignment.assumptions.map((a, i) => (
              <p key={`assumption-${i}`} className={styles.caveat}>
                {a}
              </p>
            ))}
            {reconstruction.alignment.warnings.map((w, i) => (
              <p key={`warning-${i}`} className={styles.caveatWarning}>
                {(w.description as string) || w.type}
              </p>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
