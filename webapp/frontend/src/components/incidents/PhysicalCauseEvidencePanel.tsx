import type { FaultEpisodeOut, PhysicalCauseEvidenceOut } from "../../api/client";
import styles from "./PhysicalCauseEvidencePanel.module.css";

interface Props {
  physicalCauseEvidence: PhysicalCauseEvidenceOut;
  episodes: FaultEpisodeOut[];
  recordLabel: (recordId: string) => string;
}

const CONSISTENCY_TONE: Record<string, string> = {
  CONSISTENT: "good",
  MOSTLY_CONSISTENT: "neutral",
  MIXED: "warn",
  CONTRADICTORY: "warn",
  INSUFFICIENT: "neutral",
};

export default function PhysicalCauseEvidencePanel({ physicalCauseEvidence, episodes, recordLabel }: Props) {
  const episodeByRecordId = new Map<string, number>();
  episodes.forEach((ep) => ep.member_record_ids.forEach((rid) => episodeByRecordId.set(rid, ep.episode_index)));

  return (
    <div className={styles.wrap}>
      <div className={styles.headerRow}>
        <span className={styles.scopeTag}>{physicalCauseEvidence.scope}</span>
        <span className={`${styles.consistencyBadge} ${styles[`tone_${CONSISTENCY_TONE[physicalCauseEvidence.consistency] ?? "neutral"}`]}`}>
          {physicalCauseEvidence.consistency}
        </span>
        <span className={styles.rootCause}>Incident root cause: {physicalCauseEvidence.incident_root_cause}</span>
      </div>

      <p className={styles.disclaimer}>
        Each row is an independent per-record LightGBM prediction — these are never averaged into a single
        incident-level probability. Duplicate captures, different episodes, and different mechanisms are not
        combined.
      </p>

      {physicalCauseEvidence.records.length === 0 ? (
        <div className={styles.empty}>No physical-cause evidence available.</div>
      ) : (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Record</th>
                <th>Episode</th>
                <th>Top hypothesis</th>
                <th>Confidence</th>
                <th>Ranked candidates</th>
                <th>Model version</th>
                <th>Timing source</th>
                <th>Caps applied</th>
              </tr>
            </thead>
            <tbody>
              {physicalCauseEvidence.records.map((r) => {
                const episodeIndex = episodeByRecordId.get(r.incident_record_id);
                return (
                  <tr key={r.incident_record_id}>
                    <td>{recordLabel(r.incident_record_id)}</td>
                    <td>{episodeIndex != null ? `Episode ${episodeIndex + 1}` : "-"}</td>
                    <td>{r.top_hypothesis ?? "-"}</td>
                    <td>{r.confidence != null ? `${Math.round(r.confidence * 100)}%` : "-"}</td>
                    <td>
                      {r.cause_ranking.slice(0, 3).map((c) => `${c.cause} (${Math.round(c.confidence * 100)}%)`).join(", ") || "-"}
                    </td>
                    <td className={styles.mono}>{r.model_version ?? "-"}</td>
                    <td>{r.timing_source ?? "-"}</td>
                    <td>{r.applied_caps.length > 0 ? r.applied_caps.map((c) => c.name).join(", ") : "none"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
