import { useState } from "react";
import { overrideRelationship, type RecordRelationshipOut, type RelationshipType } from "../../api/client";
import styles from "./RelationshipInspector.module.css";

interface Props {
  incidentId: string;
  relationships: RecordRelationshipOut[];
  recordLabel: (recordId: string) => string;
  onOverridden: () => void;
}

type FilterGroup = "ALL" | "DUPLICATE_OVERLAP" | "CONTINUATION_RECLOSE" | "REPEATED_EVOLVING" | "UNCERTAIN_UNRELATED";

const FILTER_GROUPS: Record<FilterGroup, RelationshipType[] | null> = {
  ALL: null,
  DUPLICATE_OVERLAP: ["DUPLICATE_TRIGGER", "OVERLAPPING_CAPTURE"],
  CONTINUATION_RECLOSE: ["CONTINUATION", "RECLOSE_SEQUENCE"],
  REPEATED_EVOLVING: ["REPEATED_FAULT", "POSSIBLE_EVOLVING_FAULT", "NEW_FAULT_EPISODE"],
  UNCERTAIN_UNRELATED: ["UNCERTAIN", "UNRELATED"],
};

const FILTER_LABELS: Record<FilterGroup, string> = {
  ALL: "All",
  DUPLICATE_OVERLAP: "Duplicate / overlap",
  CONTINUATION_RECLOSE: "Continuation / reclose",
  REPEATED_EVOLVING: "Repeated / evolving",
  UNCERTAIN_UNRELATED: "Uncertain / unrelated",
};

const RELATIONSHIP_TYPE_OPTIONS: RelationshipType[] = [
  "DUPLICATE_TRIGGER",
  "OVERLAPPING_CAPTURE",
  "CONTINUATION",
  "RECLOSE_SEQUENCE",
  "NEW_FAULT_EPISODE",
  "REPEATED_FAULT",
  "POSSIBLE_EVOLVING_FAULT",
  "UNRELATED",
  "UNCERTAIN",
];

function formatEvidence(items: Array<Record<string, unknown>>): string[] {
  return items.map((item) => {
    if (typeof item.description === "string") return item.description;
    if (typeof item.type === "string") return item.type.replace(/_/g, " ");
    return JSON.stringify(item);
  });
}

export default function RelationshipInspector({ incidentId, relationships, recordLabel, onOverridden }: Props) {
  const [filter, setFilter] = useState<FilterGroup>("ALL");
  const [overridingId, setOverridingId] = useState<string | null>(null);

  const allowedTypes = FILTER_GROUPS[filter];
  const filtered = allowedTypes ? relationships.filter((r) => allowedTypes.includes(r.relationship_type)) : relationships;

  if (relationships.length === 0) {
    return <div className={styles.empty}>No relationships computed yet. Trigger a reconstruction first.</div>;
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.filters}>
        {(Object.keys(FILTER_GROUPS) as FilterGroup[]).map((key) => (
          <button
            key={key}
            type="button"
            className={filter === key ? styles.filterActive : styles.filterButton}
            onClick={() => setFilter(key)}
          >
            {FILTER_LABELS[key]}
          </button>
        ))}
      </div>

      {filtered.length === 0 ? (
        <div className={styles.empty}>No relationships match this filter.</div>
      ) : (
        <div className={styles.list}>
          {filtered.map((rel) => (
            <RelationshipCard
              key={rel.relationship_id}
              incidentId={incidentId}
              relationship={rel}
              recordLabel={recordLabel}
              overriding={overridingId === rel.relationship_id}
              onStartOverride={() => setOverridingId(rel.relationship_id)}
              onCancelOverride={() => setOverridingId(null)}
              onOverridden={() => {
                setOverridingId(null);
                onOverridden();
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function RelationshipCard({
  incidentId,
  relationship,
  recordLabel,
  overriding,
  onStartOverride,
  onCancelOverride,
  onOverridden,
}: {
  incidentId: string;
  relationship: RecordRelationshipOut;
  recordLabel: (recordId: string) => string;
  overriding: boolean;
  onStartOverride: () => void;
  onCancelOverride: () => void;
  onOverridden: () => void;
}) {
  const [correctedType, setCorrectedType] = useState<RelationshipType>(relationship.relationship_type);
  const [operator, setOperator] = useState("");
  const [reason, setReason] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const similarity = relationship.metrics.waveform_similarity;

  async function submitOverride() {
    if (!operator.trim() || !reason.trim()) {
      setError("Operator name and reason are required.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await overrideRelationship(incidentId, relationship.relationship_id, {
        corrected_relationship: correctedType,
        operator: operator.trim(),
        reason: reason.trim(),
      });
      onOverridden();
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail || (err instanceof Error ? err.message : "Override failed."));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className={`${styles.card} ${relationship.relationship_type === "UNCERTAIN" ? styles.cardUncertain : ""}`}>
      <div className={styles.cardHeader}>
        <span className={styles.recordPair}>
          {recordLabel(relationship.left_record_id)} ↔ {recordLabel(relationship.right_record_id)}
        </span>
        <span className={`${styles.typeBadge} ${styles[`type_${relationship.relationship_type}`] ?? ""}`}>
          {relationship.relationship_type.replace(/_/g, " ")}
        </span>
        <span className={styles.confidence}>{Math.round(relationship.confidence * 100)}% confidence</span>
        {relationship.overridden && <span className={styles.overriddenBadge}>overridden</span>}
      </div>

      {relationship.relationship_type === "UNCERTAIN" && (
        <div className={styles.uncertainNote}>
          Insufficient evidence to classify this pair with confidence. Treat as unresolved rather than assuming a relationship.
        </div>
      )}

      <div className={styles.evidenceGrid}>
        <div>
          <span className={styles.evidenceLabel}>Evidence for</span>
          {relationship.evidence_for.length === 0 ? (
            <p className={styles.muted}>None</p>
          ) : (
            <ul>{formatEvidence(relationship.evidence_for).map((e, i) => <li key={i}>{e}</li>)}</ul>
          )}
        </div>
        <div>
          <span className={styles.evidenceLabel}>Evidence against</span>
          {relationship.evidence_against.length === 0 ? (
            <p className={styles.muted}>None</p>
          ) : (
            <ul>{formatEvidence(relationship.evidence_against).map((e, i) => <li key={i}>{e}</li>)}</ul>
          )}
        </div>
      </div>

      {(similarity || relationship.metrics.digital_sequence_similarity !== undefined || relationship.metrics.gap_seconds !== undefined) && (
        <div className={styles.metricsRow}>
          {relationship.metrics.gap_seconds !== undefined && (
            <span>Gap: {relationship.metrics.gap_seconds.toFixed(1)}s</span>
          )}
          {similarity?.computed && (
            <>
              <span>Waveform correlation: {similarity.mean_correlation != null ? similarity.mean_correlation.toFixed(3) : "-"}</span>
              <span>RMS diff: {similarity.mean_rms_relative_diff != null ? similarity.mean_rms_relative_diff.toFixed(3) : "-"}</span>
            </>
          )}
          {similarity && !similarity.computed && <span>Waveform similarity not computed: {similarity.reason}</span>}
          {relationship.metrics.digital_sequence_similarity !== undefined && (
            <span>Digital sequence similarity: {(relationship.metrics.digital_sequence_similarity as number).toFixed(2)}</span>
          )}
        </div>
      )}

      {relationship.assumptions.length > 0 && (
        <div className={styles.assumptions}>
          <span className={styles.evidenceLabel}>Assumptions</span>
          <ul>{relationship.assumptions.map((a, i) => <li key={i}>{a}</li>)}</ul>
        </div>
      )}

      {relationship.warnings.length > 0 && (
        <div className={styles.warnings}>
          <span className={styles.evidenceLabel}>Warnings</span>
          <ul>{formatEvidence(relationship.warnings).map((w, i) => <li key={i}>{w}</li>)}</ul>
        </div>
      )}

      {relationship.overridden && (
        <div className={styles.overrideProvenance}>
          Original: <strong>{relationship.override_previous_type}</strong> → Overridden to{" "}
          <strong>{relationship.relationship_type}</strong> by {relationship.override_operator} at{" "}
          {relationship.override_at_iso ? new Date(relationship.override_at_iso).toLocaleString() : "-"}.
          {relationship.override_reason && <> Reason: {relationship.override_reason}</>}
        </div>
      )}

      {!overriding ? (
        <button type="button" className={styles.overrideButton} onClick={onStartOverride}>
          Override relationship
        </button>
      ) : (
        <div className={styles.overrideForm}>
          <label>
            Corrected relationship type
            <select value={correctedType} onChange={(e) => setCorrectedType(e.target.value as RelationshipType)}>
              {RELATIONSHIP_TYPE_OPTIONS.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>
          <label>
            Operator name (required)
            <input value={operator} onChange={(e) => setOperator(e.target.value)} placeholder="e.g. engineer1" />
          </label>
          <label>
            Reason (required)
            <textarea value={reason} onChange={(e) => setReason(e.target.value)} rows={2} placeholder="Why is the algorithm result wrong?" />
          </label>
          {error && <div className={styles.error}>{error}</div>}
          <div className={styles.overrideActions}>
            <button type="button" onClick={onCancelOverride}>
              Cancel
            </button>
            <button type="button" className={styles.confirmButton} onClick={submitOverride} disabled={submitting}>
              {submitting ? "Saving…" : "Save override"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
