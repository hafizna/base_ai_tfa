import type { ReactNode } from "react";
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
        combined. A record marked <strong>aftermath</strong> only captures a reclose/continuation/duplicate of a
        preceding record's fault — its reading is shown for audit, but it does not count as independent cause
        evidence and is excluded from the consistency badge above. An <strong>inception</strong> record's
        confidence may be nudged up or down by its own aftermath record's reclose outcome (transient causes are
        expected to self-clear; permanent causes are expected to persist) — see that row's caps.
      </p>

      {physicalCauseEvidence.records.length === 0 ? (
        <div className={styles.empty}>No physical-cause evidence available.</div>
      ) : (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Record</th>
                <th>Role</th>
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
                if (r.skip_reason) {
                  return (
                    <tr key={r.incident_record_id} className={styles.rowSkipped}>
                      <td>{recordLabel(r.incident_record_id)}</td>
                      <td>
                        <span className={styles.roleAftermath}>{r.evidence_role}</span>
                      </td>
                      <td>{episodeIndex != null ? `Episode ${episodeIndex + 1}` : "-"}</td>
                      <td colSpan={6} className={styles.skipCell}>
                        No AI cause reading — {r.skip_reason === "unsupported_protection_type"
                          ? "this record's protection type has no line-fault classifier support (e.g. transformer differential)."
                          : r.skip_reason}
                      </td>
                    </tr>
                  );
                }
                return (
                  <tr key={r.incident_record_id} className={r.requires_review ? styles.rowReview : undefined}>
                    <td>{recordLabel(r.incident_record_id)}</td>
                    <td>
                      <span className={r.evidence_role === "aftermath" ? styles.roleAftermath : styles.roleInception}>
                        {r.evidence_role}
                      </span>
                    </td>
                    <td>{episodeIndex != null ? `Episode ${episodeIndex + 1}` : "-"}</td>
                    <td>
                      {r.top_hypothesis ?? "-"}
                      {r.requires_review && <span className={styles.reviewFlag} title="Reclose outcome conflicts with this cause's expected fault_type">⚠ review</span>}
                    </td>
                    <td>{r.confidence != null ? `${Math.round(r.confidence * 100)}%` : "-"}</td>
                    <td>
                      {r.cause_ranking.slice(0, 3).map((c) => `${c.cause} (${Math.round(c.confidence * 100)}%)`).join(", ") || "-"}
                    </td>
                    <td className={styles.mono}>{r.model_version ?? "-"}</td>
                    <td>{r.timing_source ?? "-"}</td>
                    <td>
                      {r.applied_caps.length > 0
                        ? r.applied_caps.map((c) => (
                            <span
                              key={c.name}
                              className={c.name.includes("conflict") ? styles.capConflict : c.name.includes("consistency") ? styles.capConsistency : undefined}
                              title={c.reason}
                            >
                              {c.name}
                            </span>
                          )).reduce((acc, el, i) => (i === 0 ? [el] : [...acc, ", ", el]), [] as ReactNode[])
                        : "none"}
                    </td>
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
