import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  addIncidentEvidence,
  deleteIncident,
  detachIncidentRecord,
  fetchIncident,
  fetchIncidentEpisodes,
  fetchIncidentRelationships,
  fetchIncidentTimeline,
  fetchReconstruction,
  listIncidentEvidence,
  listReconstructions,
  reconstructIncident,
  removeIncidentEvidence,
  reorderIncidentRecords,
  updateIncident,
  type AssetType,
  type BatchUploadResponse,
  type ClockAssessment,
  type EvidenceConfidence,
  type EvidenceType,
  type FaultEpisodeOut,
  type IncidentEvidenceOut,
  type IncidentOut,
  type IncidentRecordOut,
  type IncidentStatus,
  type IncidentTimelineEventOut,
  type ProtectionFamily,
  type ReconstructionOut,
  type RecordRelationshipOut,
} from "../api/client";
import BatchUploadPanel from "../components/incidents/BatchUploadPanel";
import EpisodeCards from "../components/incidents/EpisodeCards";
import IncidentRecordsTable from "../components/incidents/IncidentRecordsTable";
import NarrativePanel from "../components/incidents/NarrativePanel";
import PhysicalCauseEvidencePanel from "../components/incidents/PhysicalCauseEvidencePanel";
import ReconstructionControls from "../components/incidents/ReconstructionControls";
import ReconstructionSummary from "../components/incidents/ReconstructionSummary";
import RelationshipInspector from "../components/incidents/RelationshipInspector";
import SegmentedTimeline from "../components/incidents/SegmentedTimeline";
import { useMultiComtradeEnabled } from "../hooks/useFeatureFlags";
import styles from "./IncidentWorkspace.module.css";

const STATUS_OPTIONS: IncidentStatus[] = ["DRAFT", "OPEN", "UNDER_REVIEW", "CONFIRMED", "CLOSED", "ARCHIVED"];
const ASSET_TYPE_OPTIONS: AssetType[] = [
  "TRANSMISSION_LINE",
  "TRANSFORMER",
  "BUSBAR",
  "FEEDER",
  "REACTOR",
  "CAPACITOR",
  "OTHER",
  "UNKNOWN",
];
const PROTECTION_FAMILY_OPTIONS: ProtectionFamily[] = [
  "DISTANCE",
  "LINE_DIFFERENTIAL",
  "TRANSFORMER_DIFFERENTIAL",
  "OVERCURRENT",
  "REF",
  "SBEF",
  "MIXED",
  "UNKNOWN",
];
const CLOCK_ASSESSMENT_OPTIONS: ClockAssessment[] = [
  "SYNCHRONIZED",
  "LIKELY_SYNCHRONIZED",
  "ORDER_ONLY",
  "UNTRUSTED",
  "UNKNOWN",
];
const EVIDENCE_TYPE_OPTIONS: EvidenceType[] = [
  "COMTRADE_RECORD",
  "REMOTE_END_COMTRADE",
  "RELAY_EVENT_REPORT",
  "OPERATOR_SOE",
  "FIELD_INSPECTION",
  "PATROL_REPORT",
  "LIGHTNING_DETECTION",
  "PROTECTION_ENGINEER_NOTE",
  "PHOTO",
  "OTHER",
];
const EVIDENCE_CONFIDENCE_OPTIONS: EvidenceConfidence[] = ["CONFIRMED", "PROBABLE", "POSSIBLE", "UNKNOWN"];

function formatTime(iso: string | null) {
  if (!iso) return "No absolute time";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function computeRecordSignature(records: IncidentRecordOut[]): string {
  return JSON.stringify(
    records.map((r) => ({ id: r.analysis_id, order: r.manual_order })).sort((a, b) => a.id.localeCompare(b.id)),
  );
}

export default function IncidentWorkspace() {
  const { incidentId } = useParams<{ incidentId: string }>();
  const navigate = useNavigate();
  const [incident, setIncident] = useState<IncidentOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [savingHeader, setSavingHeader] = useState(false);
  const [showEvidenceForm, setShowEvidenceForm] = useState(false);
  const [evidenceDraft, setEvidenceDraft] = useState({
    evidence_type: "OTHER" as EvidenceType,
    source: "",
    description: "",
    confidence: "UNKNOWN" as EvidenceConfidence,
  });

  const multiComtradeEnabled = useMultiComtradeEnabled();

  // --- Stage 2: reconstruction state ---
  const [reconstruction, setReconstruction] = useState<ReconstructionOut | null>(null);
  const [reconstructions, setReconstructions] = useState<ReconstructionOut[]>([]);
  const [timeline, setTimeline] = useState<IncidentTimelineEventOut[]>([]);
  const [relationships, setRelationships] = useState<RecordRelationshipOut[]>([]);
  const [episodes, setEpisodes] = useState<FaultEpisodeOut[]>([]);
  const [reconstructionLoading, setReconstructionLoading] = useState(false);
  const [reconstructionError, setReconstructionError] = useState<string | null>(null);
  const [reconstructedAtRecordCount, setReconstructedAtRecordCount] = useState<number | null>(null);
  const [reconstructedAtSnapshotSignature, setReconstructedAtSnapshotSignature] = useState<string | null>(null);
  const [showBatchUpload, setShowBatchUpload] = useState(false);

  async function load() {
    if (!incidentId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchIncident(incidentId);
      setIncident(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load incident.");
    } finally {
      setLoading(false);
    }
  }

  async function loadReconstructionState(targetReconstructionId?: string) {
    if (!incidentId || !multiComtradeEnabled) return;
    setReconstructionLoading(true);
    setReconstructionError(null);
    try {
      const [recon, versions] = await Promise.all([
        fetchReconstruction(incidentId, targetReconstructionId).catch((err) => {
          if (err?.response?.status === 404) return null;
          throw err;
        }),
        listReconstructions(incidentId).catch(() => []),
      ]);
      setReconstruction(recon);
      setReconstructions(versions);
      if (recon) {
        setTimeline(recon.timeline ?? (await fetchIncidentTimeline(incidentId).catch(() => [])));
        setRelationships(recon.relationships ?? (await fetchIncidentRelationships(incidentId).catch(() => [])));
        setEpisodes(recon.episodes ?? (await fetchIncidentEpisodes(incidentId).catch(() => [])));
        if (recon.is_latest && incident) {
          setReconstructedAtRecordCount(recon.observed_incident_facts.record_count);
          setReconstructedAtSnapshotSignature(computeRecordSignature(incident.records));
        }
      } else {
        setTimeline([]);
        setRelationships([]);
        setEpisodes([]);
      }
    } catch (err) {
      setReconstructionError(err instanceof Error ? err.message : "Failed to load reconstruction.");
    } finally {
      setReconstructionLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [incidentId]);

  useEffect(() => {
    if (multiComtradeEnabled) {
      loadReconstructionState();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [incidentId, multiComtradeEnabled]);

  async function handleReconstruct() {
    if (!incidentId) return;
    setReconstructionError(null);
    try {
      const recon = await reconstructIncident(incidentId);
      const freshIncident = await fetchIncident(incidentId);
      setIncident(freshIncident);
      setReconstruction(recon);
      setTimeline(recon.timeline ?? []);
      setRelationships(recon.relationships ?? []);
      setEpisodes(recon.episodes ?? []);
      setReconstructedAtRecordCount(recon.observed_incident_facts.record_count);
      setReconstructedAtSnapshotSignature(computeRecordSignature(freshIncident.records));
      const versions = await listReconstructions(incidentId).catch(() => []);
      setReconstructions(versions);
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setReconstructionError(detail || (err instanceof Error ? err.message : "Reconstruction failed."));
    }
  }

  async function handleBatchUploaded(result: BatchUploadResponse, autoReconstruct: boolean) {
    await load();
    if (autoReconstruct && result.records_created.length > 0) {
      await handleReconstruct();
    }
  }

  const staleness = useMemo(() => {
    if (!incident || !reconstruction) return { stale: false, reason: null as string | null };
    const currentRecordCount = incident.records.length;
    if (reconstructedAtRecordCount !== null && currentRecordCount !== reconstructedAtRecordCount) {
      return { stale: true, reason: "the number of attached records has changed since this reconstruction" };
    }
    const currentSignature = JSON.stringify(
      incident.records
        .map((r) => ({ id: r.analysis_id, order: r.manual_order }))
        .sort((a, b) => a.id.localeCompare(b.id)),
    );
    if (reconstructedAtSnapshotSignature !== null && currentSignature !== reconstructedAtSnapshotSignature) {
      return { stale: true, reason: "record order or membership changed since this reconstruction" };
    }
    return { stale: false, reason: null };
  }, [incident, reconstruction, reconstructedAtRecordCount, reconstructedAtSnapshotSignature]);

  function recordLabel(recordId: string): string {
    const record = incident?.records.find((r) => r.incident_record_id === recordId);
    return record?.source_filename || record?.analysis_id.slice(0, 10) || recordId.slice(0, 8);
  }

  async function patchField(field: string, value: unknown) {
    if (!incidentId) return;
    setSavingHeader(true);
    try {
      const updated = await updateIncident(incidentId, { [field]: value });
      setIncident(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update incident.");
    } finally {
      setSavingHeader(false);
    }
  }

  async function handleDetach(incidentRecordId: string) {
    if (!incidentId) return;
    if (!window.confirm("Remove this record from the incident? The underlying analysis is not deleted.")) return;
    try {
      await detachIncidentRecord(incidentId, incidentRecordId);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to detach record.");
    }
  }

  async function handleMove(records: IncidentRecordOut[], index: number, direction: -1 | 1) {
    if (!incidentId) return;
    const target = index + direction;
    if (target < 0 || target >= records.length) return;
    const ids = records.map((r) => r.incident_record_id);
    [ids[index], ids[target]] = [ids[target], ids[index]];
    try {
      await reorderIncidentRecords(incidentId, ids);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reorder records.");
    }
  }

  async function handleAddEvidence() {
    if (!incidentId) return;
    try {
      await addIncidentEvidence(incidentId, evidenceDraft);
      setShowEvidenceForm(false);
      setEvidenceDraft({ evidence_type: "OTHER", source: "", description: "", confidence: "UNKNOWN" });
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add evidence.");
    }
  }

  async function handleArchive() {
    if (!incidentId) return;
    if (!window.confirm("Archive this incident? It will be hidden from the default incident list.")) return;
    try {
      await deleteIncident(incidentId);
      navigate("/incidents");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to archive incident.");
    }
  }

  if (loading) return <div className={styles.page}>Loading incident…</div>;
  if (error && !incident) return <div className={styles.page}>{error}</div>;
  if (!incident) return null;

  return (
    <div className={styles.page}>
      <button type="button" className={styles.backLink} onClick={() => navigate("/incidents")}>
        ← Back to incidents
      </button>

      {error && <div className={styles.error}>{error}</div>}

      <section className={styles.headerCard}>
        <div className={styles.headerTop}>
          <input
            className={styles.titleInput}
            value={incident.title}
            onChange={(e) => setIncident({ ...incident, title: e.target.value })}
            onBlur={(e) => patchField("title", e.target.value)}
          />
          <select
            className={styles.statusSelect}
            value={incident.status}
            onChange={(e) => patchField("status", e.target.value)}
            disabled={savingHeader}
          >
            {STATUS_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <button type="button" className={styles.archiveButton} onClick={handleArchive}>
            Archive
          </button>
        </div>

        <div className={styles.headerGrid}>
          <label className={styles.field}>
            Station
            <input
              defaultValue={incident.station_name ?? ""}
              onBlur={(e) => patchField("station_name", e.target.value || null)}
            />
          </label>
          <label className={styles.field}>
            Bay
            <input
              defaultValue={incident.bay_name ?? ""}
              onBlur={(e) => patchField("bay_name", e.target.value || null)}
            />
          </label>
          <label className={styles.field}>
            Asset name
            <input
              defaultValue={incident.asset_name ?? ""}
              onBlur={(e) => patchField("asset_name", e.target.value || null)}
            />
          </label>
          <label className={styles.field}>
            Asset type
            <select
              value={incident.asset_type ?? "UNKNOWN"}
              onChange={(e) => patchField("asset_type", e.target.value)}
            >
              {ASSET_TYPE_OPTIONS.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>
          <label className={styles.field}>
            Voltage (kV)
            <input
              type="number"
              defaultValue={incident.voltage_level_kv ?? ""}
              onBlur={(e) => patchField("voltage_level_kv", e.target.value ? Number(e.target.value) : null)}
            />
          </label>
          <label className={styles.field}>
            Protection family
            <select
              value={incident.protection_family ?? "UNKNOWN"}
              onChange={(e) => patchField("protection_family", e.target.value)}
            >
              {PROTECTION_FAMILY_OPTIONS.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>
          <label className={styles.field}>
            Clock assessment
            <select
              value={incident.clock_assessment}
              onChange={(e) => patchField("clock_assessment", e.target.value)}
            >
              {CLOCK_ASSESSMENT_OPTIONS.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>
          <label className={styles.field}>
            Incident start
            <input value={formatTime(incident.incident_start_iso)} readOnly />
          </label>
          <label className={styles.field}>
            Incident end
            <input value={formatTime(incident.incident_end_iso)} readOnly />
          </label>
        </div>

        <label className={styles.field}>
          Operator notes
          <textarea
            defaultValue={incident.operator_notes ?? ""}
            onBlur={(e) => patchField("operator_notes", e.target.value || null)}
            rows={2}
          />
        </label>
      </section>

      {multiComtradeEnabled && (
        <section className={styles.section}>
          <div className={styles.sectionHeaderRow}>
            <h2>Batch upload</h2>
            <button type="button" className={styles.smallButton} onClick={() => setShowBatchUpload((v) => !v)}>
              {showBatchUpload ? "Hide" : "Add records"}
            </button>
          </div>
          {showBatchUpload && <BatchUploadPanel incidentId={incident.incident_id} onUploaded={handleBatchUploaded} />}
        </section>
      )}

      <section className={styles.section}>
        <h2>Attached records</h2>
        <IncidentRecordsTable
          records={incident.records}
          episodes={episodes}
          relationships={relationships}
          manualOrderConflict={incident.missing_evidence.some((m) => m.type === "RECORD_ORDER_REQUIRES_REVIEW")}
          onMove={handleMove}
          onDetach={handleDetach}
        />
      </section>

      {multiComtradeEnabled && (
        <section className={styles.section}>
          <h2>Reconstruction</h2>
          <ReconstructionControls
            reconstruction={reconstruction}
            reconstructions={reconstructions}
            stale={staleness.stale}
            staleReason={staleness.reason}
            loading={reconstructionLoading}
            recordCount={incident.records.length}
            onReconstruct={handleReconstruct}
            onSelectVersion={(id) => loadReconstructionState(id)}
            selectedVersionId={reconstruction?.reconstruction_id ?? null}
          />
          {reconstructionError && <div className={styles.error}>{reconstructionError}</div>}

          {reconstruction && (
            <>
              <div className={styles.subsection}>
                <h3>Summary</h3>
                <ReconstructionSummary reconstruction={reconstruction} />
              </div>

              <div className={styles.subsection}>
                <h3>Segmented incident timeline</h3>
                <SegmentedTimeline
                  records={incident.records}
                  timeline={timeline}
                  episodes={episodes}
                  relationships={relationships}
                />
              </div>

              <div className={styles.subsection}>
                <h3>Episodes</h3>
                <EpisodeCards episodes={episodes} records={incident.records} />
              </div>

              <div className={styles.subsection}>
                <h3>Relationship inspector</h3>
                <RelationshipInspector
                  incidentId={incident.incident_id}
                  relationships={relationships}
                  recordLabel={recordLabel}
                  onOverridden={() => loadReconstructionState(reconstruction.reconstruction_id)}
                />
              </div>

              <div className={styles.subsection}>
                <h3>Narrative and uncertainty</h3>
                <NarrativePanel reconstruction={reconstruction} />
              </div>

              <div className={styles.subsection}>
                <h3>Physical-cause evidence</h3>
                <PhysicalCauseEvidencePanel
                  physicalCauseEvidence={reconstruction.physical_cause_evidence}
                  episodes={episodes}
                  recordLabel={recordLabel}
                />
              </div>
            </>
          )}
        </section>
      )}

      <section className={styles.section}>
        <div className={styles.sectionHeaderRow}>
          <h2>Evidence</h2>
          <button type="button" className={styles.smallButton} onClick={() => setShowEvidenceForm((v) => !v)}>
            + Add evidence
          </button>
        </div>
        {showEvidenceForm && (
          <div className={styles.evidenceForm}>
            <select
              value={evidenceDraft.evidence_type}
              onChange={(e) => setEvidenceDraft({ ...evidenceDraft, evidence_type: e.target.value as EvidenceType })}
            >
              {EVIDENCE_TYPE_OPTIONS.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <select
              value={evidenceDraft.confidence}
              onChange={(e) => setEvidenceDraft({ ...evidenceDraft, confidence: e.target.value as EvidenceConfidence })}
            >
              {EVIDENCE_CONFIDENCE_OPTIONS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <input
              placeholder="Source (e.g. Field team, BMKG)"
              value={evidenceDraft.source}
              onChange={(e) => setEvidenceDraft({ ...evidenceDraft, source: e.target.value })}
            />
            <textarea
              placeholder="Description / notes"
              value={evidenceDraft.description}
              onChange={(e) => setEvidenceDraft({ ...evidenceDraft, description: e.target.value })}
              rows={2}
            />
            <div className={styles.modalActions}>
              <button type="button" onClick={() => setShowEvidenceForm(false)}>
                Cancel
              </button>
              <button type="button" className={styles.smallButton} onClick={handleAddEvidence}>
                Save
              </button>
            </div>
          </div>
        )}
        {incident.evidence_ids.length === 0 ? (
          <div className={styles.empty}>No evidence added yet.</div>
        ) : (
          <EvidenceList incidentId={incident.incident_id} onChanged={load} />
        )}
      </section>

      <section className={styles.section}>
        <h2>Record collection summary</h2>
        <p className={styles.summaryNote}>{incident.incident_interpretation.summary}</p>
        <div className={styles.summaryGrid}>
          <div>
            <strong>{incident.observed_summary.record_count}</strong>
            <span>records</span>
          </div>
          <div>
            <strong>{incident.observed_summary.records_with_absolute_time}</strong>
            <span>with absolute time</span>
          </div>
          <div>
            <strong>{incident.observed_summary.records_without_absolute_time}</strong>
            <span>without absolute time</span>
          </div>
          <div>
            <strong>{incident.observed_summary.protection_types.join(", ") || "-"}</strong>
            <span>protection types</span>
          </div>
        </div>
      </section>

      <section className={styles.section}>
        <h2>Missing evidence</h2>
        {incident.missing_evidence.length === 0 ? (
          <div className={styles.empty}>Nothing flagged.</div>
        ) : (
          <ul className={styles.missingList}>
            {incident.missing_evidence.map((m, i) => (
              <li key={i}>{m.description}</li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function EvidenceList({ incidentId, onChanged }: { incidentId: string; onChanged: () => void }) {
  const [items, setItems] = useState<IncidentEvidenceOut[] | null>(null);

  useEffect(() => {
    listIncidentEvidence(incidentId).then(setItems);
  }, [incidentId]);

  async function remove(evidenceId: string) {
    await removeIncidentEvidence(incidentId, evidenceId);
    onChanged();
    setItems(await listIncidentEvidence(incidentId));
  }

  if (!items) return <div className={styles.empty}>Loading evidence…</div>;
  if (items.length === 0) return <div className={styles.empty}>No evidence added yet.</div>;

  return (
    <div className={styles.evidenceListWrap}>
      {items.map((ev) => (
        <div key={ev.evidence_id} className={styles.evidenceItem}>
          <div>
            <span className={styles.evidenceType}>{ev.evidence_type}</span>
            <span className={styles.evidenceConfidence}>{ev.confidence}</span>
          </div>
          <div className={styles.evidenceDescription}>{ev.description || "-"}</div>
          <div className={styles.evidenceSource}>{ev.source}</div>
          <button type="button" onClick={() => remove(ev.evidence_id)}>
            Remove
          </button>
        </div>
      ))}
    </div>
  );
}
