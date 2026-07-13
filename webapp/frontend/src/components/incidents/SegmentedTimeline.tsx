import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { FaultEpisodeOut, IncidentRecordOut, IncidentTimelineEventOut, RecordRelationshipOut } from "../../api/client";
import styles from "./SegmentedTimeline.module.css";

interface Props {
  records: IncidentRecordOut[];
  timeline: IncidentTimelineEventOut[];
  episodes: FaultEpisodeOut[];
  relationships: RecordRelationshipOut[];
}

const MIN_SEGMENT_WIDTH = 140;
const MAX_SEGMENT_WIDTH = 360;
const GAP_WIDTH_COMPRESSED = 36;
const GAP_WIDTH_CHRONOLOGICAL_MAX = 220;

function formatMs(ms: number | null | undefined): string {
  if (ms == null) return "?";
  if (Math.abs(ms) < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function markerLabel(eventType: string): string {
  return eventType.replace(/_/g, " ");
}

export default function SegmentedTimeline({ records, timeline, episodes, relationships }: Props) {
  const navigate = useNavigate();
  const [scaleMode, setScaleMode] = useState<"compressed" | "chronological">("compressed");
  const [selectedEvent, setSelectedEvent] = useState<IncidentTimelineEventOut | null>(null);

  const orderedRecords = useMemo(
    () =>
      [...records].sort((a, b) => {
        const aOrder = a.manual_order ?? a.sequence_index;
        const bOrder = b.manual_order ?? b.sequence_index;
        return aOrder - bOrder;
      }),
    [records],
  );

  const episodeByRecordId = useMemo(() => {
    const map = new Map<string, FaultEpisodeOut>();
    episodes.forEach((ep) => ep.member_record_ids.forEach((rid) => map.set(rid, ep)));
    return map;
  }, [episodes]);

  const duplicateGroupIds = useMemo(() => {
    const groups = new Set<string>();
    episodes.forEach((ep) => {
      if (ep.member_record_ids.length > 1) groups.add(ep.episode_id);
    });
    return groups;
  }, [episodes]);

  const overlapRecordIds = useMemo(() => {
    const set = new Set<string>();
    relationships.forEach((rel) => {
      if (rel.relationship_type === "OVERLAPPING_CAPTURE" || rel.relationship_type === "DUPLICATE_TRIGGER") {
        set.add(rel.left_record_id);
        set.add(rel.right_record_id);
      }
    });
    return set;
  }, [relationships]);

  const eventsByRecordId = useMemo(() => {
    const map = new Map<string, IncidentTimelineEventOut[]>();
    timeline.forEach((event) => {
      if (!event.incident_record_id) return;
      const list = map.get(event.incident_record_id) ?? [];
      list.push(event);
      map.set(event.incident_record_id, list);
    });
    return map;
  }, [timeline]);

  const gapEvents = timeline.filter((e) => e.event_type === "DATA_GAP");

  if (orderedRecords.length === 0) {
    return <div className={styles.empty}>No records to display on the timeline.</div>;
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.toolbar}>
        <div className={styles.scaleToggle}>
          <button
            type="button"
            className={scaleMode === "compressed" ? styles.scaleActive : ""}
            onClick={() => setScaleMode("compressed")}
          >
            Compressed
          </button>
          <button
            type="button"
            className={scaleMode === "chronological" ? styles.scaleActive : ""}
            onClick={() => setScaleMode("chronological")}
          >
            Chronological
          </button>
        </div>
        <div className={styles.legend}>
          <span className={styles.legendItem}><span className={styles.legendSwatchOverlap} /> Overlap/duplicate</span>
          <span className={styles.legendItem}><span className={styles.legendSwatchGap} /> Gap (actual duration shown)</span>
        </div>
      </div>

      <div className={styles.track}>
        {orderedRecords.map((record, index) => {
          const episode = episodeByRecordId.get(record.incident_record_id);
          const events = eventsByRecordId.get(record.incident_record_id) ?? [];
          const isDuplicateGroup = episode && duplicateGroupIds.has(episode.episode_id);
          const isOverlap = overlapRecordIds.has(record.incident_record_id);

          const gapBefore = index > 0
            ? gapEvents.find((g) => (g.details as { left_incident_record_id?: string }).left_incident_record_id === orderedRecords[index - 1].incident_record_id)
            : undefined;

          const gapWidth = gapBefore
            ? scaleMode === "compressed"
              ? GAP_WIDTH_COMPRESSED
              : Math.min(GAP_WIDTH_CHRONOLOGICAL_MAX, Math.max(GAP_WIDTH_COMPRESSED, Math.log10(Math.max((gapBefore.details.gap_ms as number) ?? 1, 10)) * 40))
            : 0;

          const recordDurationMs = record.canonical_snapshot && typeof record.canonical_snapshot === "object"
            ? ((record.canonical_snapshot as Record<string, any>).event_window?.fault_duration_ms as number | undefined)
            : undefined;
          const segmentWidth = Math.min(
            MAX_SEGMENT_WIDTH,
            Math.max(MIN_SEGMENT_WIDTH, recordDurationMs ? recordDurationMs * 1.5 : MIN_SEGMENT_WIDTH),
          );

          return (
            <div key={record.incident_record_id} className={styles.segmentGroup}>
              {gapBefore && (
                <div className={styles.gapBlock} style={{ width: gapWidth }} title={`Gap: ${formatMs(gapBefore.details.gap_ms as number)}`}>
                  <span className={styles.gapLabel}>{formatMs(gapBefore.details.gap_ms as number)}</span>
                  <span className={styles.gapTick} />
                </div>
              )}
              <div
                className={`${styles.segment} ${isOverlap ? styles.segmentOverlap : ""} ${isDuplicateGroup ? styles.segmentDuplicate : ""}`}
                style={{ width: segmentWidth }}
              >
                <button
                  type="button"
                  className={styles.segmentHeader}
                  onClick={() => navigate(`/workspace/${(record.protection_type || "21").toUpperCase()}/${record.analysis_id}`)}
                  title="Open single-record Workspace"
                >
                  <span className={styles.segmentIndex}>#{index + 1}</span>
                  <span className={styles.segmentName}>{record.source_filename || record.analysis_id.slice(0, 8)}</span>
                </button>
                {episode && (
                  <div className={styles.episodeBadge}>
                    Episode {episode.episode_index + 1}
                    {isDuplicateGroup && <span className={styles.dupTag}>duplicate group</span>}
                  </div>
                )}
                <div className={styles.markerRow}>
                  {events.map((event) => (
                    <button
                      key={event.timeline_event_id}
                      type="button"
                      className={`${styles.marker} ${styles[`marker_${event.event_type}`] ?? ""}`}
                      onClick={() => setSelectedEvent(event)}
                      title={`${markerLabel(event.event_type)} — ${event.label}`}
                    >
                      <span className={styles.markerDot} />
                    </button>
                  ))}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {selectedEvent && (
        <div className={styles.detailOverlay} onClick={() => setSelectedEvent(null)}>
          <div className={styles.detailCard} onClick={(e) => e.stopPropagation()}>
            <h4>{markerLabel(selectedEvent.event_type)}</h4>
            <p>{selectedEvent.label}</p>
            <dl className={styles.detailList}>
              <div><dt>Absolute time</dt><dd>{selectedEvent.absolute_time_iso ?? "No absolute time"}</dd></div>
              <div><dt>Relative to incident</dt><dd>{formatMs(selectedEvent.relative_incident_ms)}</dd></div>
              <div><dt>Relative to record</dt><dd>{formatMs(selectedEvent.relative_record_ms)}</dd></div>
              <div><dt>Confidence</dt><dd>{Math.round(selectedEvent.confidence * 100)}%</dd></div>
              <div><dt>Source</dt><dd>{selectedEvent.source}</dd></div>
            </dl>
            <button type="button" onClick={() => setSelectedEvent(null)}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
