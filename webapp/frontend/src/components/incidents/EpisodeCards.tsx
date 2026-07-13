import type { FaultEpisodeOut, IncidentRecordOut } from "../../api/client";
import styles from "./EpisodeCards.module.css";

interface Props {
  episodes: FaultEpisodeOut[];
  records: IncidentRecordOut[];
}

function formatTime(iso: string | null) {
  if (!iso) return "No absolute time";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function formatDuration(ms: number | null) {
  if (ms == null) return "-";
  return `${ms.toFixed(0)} ms`;
}

export default function EpisodeCards({ episodes, records }: Props) {
  const recordsById = new Map(records.map((r) => [r.incident_record_id, r]));

  if (episodes.length === 0) {
    return <div className={styles.empty}>No episodes reconstructed yet. Trigger a reconstruction to build episodes.</div>;
  }

  return (
    <div className={styles.list}>
      {episodes.map((episode) => (
        <div key={episode.episode_id} className={styles.card}>
          <div className={styles.cardHeader}>
            <span className={styles.episodeTitle}>Episode {episode.episode_index + 1}</span>
            {episode.relationship_to_previous && (
              <span className={styles.relationBadge}>{episode.relationship_to_previous.replace(/_/g, " ")}</span>
            )}
            <span className={styles.confidenceBadge}>{Math.round(episode.confidence * 100)}% timing confidence</span>
          </div>

          <div className={styles.metaGrid}>
            <div><dt>Start</dt><dd>{formatTime(episode.start_iso)}</dd></div>
            <div><dt>End</dt><dd>{formatTime(episode.end_iso)}</dd></div>
            <div><dt>Duration</dt><dd>{formatDuration(episode.duration_ms)}</dd></div>
            <div><dt>Fault type</dt><dd>{episode.fault_type ?? "-"}</dd></div>
            <div><dt>Faulted phases</dt><dd>{episode.faulted_phases.join(", ") || "-"}</dd></div>
            <div><dt>Zone operations</dt><dd>{episode.zone_operations.join(", ") || "-"}</dd></div>
            <div><dt>Trip types</dt><dd>{episode.trip_types.join(", ") || "-"}</dd></div>
            <div><dt>Reclose</dt><dd>{episode.reclose_outcome ?? "-"}</dd></div>
          </div>

          <div className={styles.section}>
            <span className={styles.sectionLabel}>Member records ({episode.member_record_ids.length})</span>
            <ul className={styles.memberList}>
              {episode.member_record_ids.map((rid) => {
                const record = recordsById.get(rid);
                return (
                  <li key={rid}>
                    {record?.source_filename || record?.analysis_id || rid}
                    {record?.relay_id ? ` (${record.relay_id})` : ""}
                  </li>
                );
              })}
            </ul>
          </div>

          <div className={styles.factBlock}>
            <span className={styles.factLabel}>Observed</span>
            <p>{JSON.stringify(episode.observed_facts)}</p>
          </div>

          {episode.interpretation.event_classes && episode.interpretation.event_classes.length > 0 && (
            <div className={styles.factBlock}>
              <span className={styles.factLabel}>Interpretation</span>
              <p>{episode.interpretation.event_classes.join(", ")}</p>
            </div>
          )}

          <div className={styles.factBlock}>
            <span className={styles.factLabelHypothesis}>Local cause hypotheses (record-local, not confirmed root cause)</span>
            {episode.local_cause_hypotheses.length === 0 ? (
              <p className={styles.mutedNote}>No cause hypotheses available.</p>
            ) : (
              <ul className={styles.hypothesisList}>
                {episode.local_cause_hypotheses.map((h, i) => (
                  <li key={i}>
                    {h.top_hypothesis ?? "unknown"}
                    {typeof h.confidence === "number" ? ` — ${Math.round(h.confidence * 100)}%` : ""}
                    <span className={styles.scopeTag}>{h.scope}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {episode.missing_evidence.length > 0 && (
            <div className={styles.factBlock}>
              <span className={styles.factLabel}>Missing evidence</span>
              <ul className={styles.missingList}>
                {episode.missing_evidence.map((m, i) => (
                  <li key={i}>{m.description}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
