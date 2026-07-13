import { useNavigate } from "react-router-dom";
import type { FaultEpisodeOut, IncidentRecordOut, RecordRelationshipOut } from "../../api/client";
import styles from "./IncidentRecordsTable.module.css";

interface Props {
  records: IncidentRecordOut[];
  episodes: FaultEpisodeOut[];
  relationships: RecordRelationshipOut[];
  manualOrderConflict: boolean;
  onMove: (records: IncidentRecordOut[], index: number, direction: -1 | 1) => void;
  onDetach: (incidentRecordId: string) => void;
}

function formatTime(iso: string | null) {
  if (!iso) return "No absolute time";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function inferWorkspaceRoute(record: IncidentRecordOut): string {
  const protectionType = (record.protection_type || "").toUpperCase();
  return `/workspace/${protectionType || "21"}/${record.analysis_id}`;
}

export default function IncidentRecordsTable({
  records,
  episodes,
  relationships,
  manualOrderConflict,
  onMove,
  onDetach,
}: Props) {
  const navigate = useNavigate();

  const orderedRecords = [...records].sort((a, b) => {
    const aOrder = a.manual_order ?? a.sequence_index;
    const bOrder = b.manual_order ?? b.sequence_index;
    return aOrder - bOrder;
  });

  const episodeByRecordId = new Map<string, FaultEpisodeOut>();
  episodes.forEach((ep) => ep.member_record_ids.forEach((rid) => episodeByRecordId.set(rid, ep)));

  const duplicateGroupSize = new Map<string, number>();
  episodes.forEach((ep) => {
    if (ep.member_record_ids.length > 1) {
      ep.member_record_ids.forEach((rid) => duplicateGroupSize.set(rid, ep.member_record_ids.length));
    }
  });

  if (orderedRecords.length === 0) {
    return <div className={styles.empty}>No records attached yet. Use "Add to Incident" or batch upload above.</div>;
  }

  return (
    <div className={styles.wrap}>
      {manualOrderConflict && (
        <div className={styles.orderWarning}>
          Manual record order conflicts with trusted absolute timestamps. Review ordering before relying on
          sequence-based conclusions.
        </div>
      )}
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>#</th>
              <th>Record</th>
              <th>Relay/device</th>
              <th>Protection</th>
              <th>Timestamp</th>
              <th>Phase/type</th>
              <th>Zone</th>
              <th>Trip</th>
              <th>Reclose</th>
              <th>Timing</th>
              <th>Episode</th>
              <th>Dup. group</th>
              <th>Top cause</th>
              <th>Warnings</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {orderedRecords.map((record, index) => {
              const snapshot = (record.canonical_snapshot || {}) as Record<string, any>;
              const observed = snapshot.observed_facts || {};
              const interpretation = snapshot.protection_interpretation || {};
              const window = snapshot.event_window || {};
              const causeHyps = snapshot.cause_hypotheses || [];
              const topCause = causeHyps[0];
              const episode = episodeByRecordId.get(record.incident_record_id);
              const dupSize = duplicateGroupSize.get(record.incident_record_id);
              const reclose = (observed.reclose_events || []).slice(-1)[0];
              const recloseLabel = reclose ? (reclose.success === true ? "successful" : reclose.success === false ? "failed" : "attempted") : "-";

              return (
                <tr key={record.incident_record_id}>
                  <td>
                    <div className={styles.orderControls}>
                      <button type="button" onClick={() => onMove(orderedRecords, index, -1)} disabled={index === 0}>
                        ↑
                      </button>
                      <span>{index + 1}</span>
                      <button
                        type="button"
                        onClick={() => onMove(orderedRecords, index, 1)}
                        disabled={index === orderedRecords.length - 1}
                      >
                        ↓
                      </button>
                    </div>
                  </td>
                  <td>
                    <button
                      type="button"
                      className={styles.recordLink}
                      onClick={() => navigate(inferWorkspaceRoute(record))}
                      title="Open single-record analysis"
                    >
                      {record.source_filename || record.analysis_id.slice(0, 10)}
                    </button>
                    <div className={styles.orderSourceTag}>{record.order_source}</div>
                  </td>
                  <td>{[record.relay_id, record.relay_model].filter(Boolean).join(" / ") || "-"}</td>
                  <td>{record.protection_type || "-"}</td>
                  <td>{formatTime(record.trigger_time_iso)}</td>
                  <td>
                    {interpretation.event_class || "UNDETERMINED"}
                    <div className={styles.subMeta}>{(observed.faulted_phases || []).join(", ") || "-"}</div>
                  </td>
                  <td>{window.zone ?? "-"}</td>
                  <td>{window.trip_type ?? "-"}</td>
                  <td>{recloseLabel}</td>
                  <td>
                    {window.method ?? "-"}
                    <div className={styles.subMeta}>
                      {typeof window.confidence === "number" ? `${Math.round(window.confidence * 100)}%` : "-"}
                    </div>
                  </td>
                  <td>{episode ? `Episode ${episode.episode_index + 1}` : "-"}</td>
                  <td>{dupSize ? `${dupSize} records` : "-"}</td>
                  <td>
                    {topCause?.cause ?? "-"}
                    {typeof topCause?.confidence === "number" && (
                      <div className={styles.subMeta}>{Math.round(topCause.confidence * 100)}%</div>
                    )}
                  </td>
                  <td>
                    {record.attachment_warnings.length === 0 ? (
                      "-"
                    ) : (
                      <div className={styles.warningBadges}>
                        {record.attachment_warnings.map((w, i) => (
                          <span key={i} className={styles.warningBadge} title={w.description as string}>
                            {(w.type as string).replace(/_/g, " ")}
                          </span>
                        ))}
                      </div>
                    )}
                  </td>
                  <td>
                    <button type="button" className={styles.detachButton} onClick={() => onDetach(record.incident_record_id)}>
                      Remove
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className={styles.note}>
        Relationships used to derive episode/duplicate-group columns: {relationships.length} pairwise result(s) available in the Relationship Inspector below.
      </div>
    </div>
  );
}
