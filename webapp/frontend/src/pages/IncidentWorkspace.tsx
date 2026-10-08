import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
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
import { buildIncidentStory, isReconstructionStale, type IncidentStory } from "../components/incidents/incidentStory";
import {
  CauseSection,
  ChecklistSection,
  RecordsSection,
  SequenceSection,
  SummarySection,
} from "../components/incidents/IncidentStoryView";
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

function recordSetSignature(records: IncidentRecordOut[]): string {
  return records
    .map((r) => r.analysis_id)
    .sort()
    .join(",");
}

function hasFiles(e: React.DragEvent) {
  return Array.from(e.dataTransfer?.types ?? []).includes("Files");
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

  const [reconstruction, setReconstruction] = useState<ReconstructionOut | null>(null);
  const [reconstructions, setReconstructions] = useState<ReconstructionOut[]>([]);
  const [timeline, setTimeline] = useState<IncidentTimelineEventOut[]>([]);
  const [relationships, setRelationships] = useState<RecordRelationshipOut[]>([]);
  const [episodes, setEpisodes] = useState<FaultEpisodeOut[]>([]);
  const [reconstructionLoaded, setReconstructionLoaded] = useState(false);
  const [reconstructionLoading, setReconstructionLoading] = useState(false);
  const [reconstructionError, setReconstructionError] = useState<string | null>(null);
  const [rebuilding, setRebuilding] = useState(false);
  // Record set an automatic rebuild was already started for, so a failing
  // rebuild is not retried in a loop.
  const autoRebuiltFor = useRef<string | null>(null);

  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadKey, setUploadKey] = useState(0);
  const [droppedFiles, setDroppedFiles] = useState<File[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const dragDepth = useRef(0);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const detailsRef = useRef<HTMLElement>(null);

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

  function applyReconstruction(recon: ReconstructionOut | null) {
    setReconstruction(recon);
    setTimeline(recon?.timeline ?? []);
    setRelationships(recon?.relationships ?? []);
    setEpisodes(recon?.episodes ?? []);
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
      if (recon) {
        applyReconstruction({
          ...recon,
          timeline: recon.timeline ?? (await fetchIncidentTimeline(incidentId).catch(() => [])),
          relationships: recon.relationships ?? (await fetchIncidentRelationships(incidentId).catch(() => [])),
          episodes: recon.episodes ?? (await fetchIncidentEpisodes(incidentId).catch(() => [])),
        });
      } else {
        applyReconstruction(null);
      }
      setReconstructions(versions);
    } catch (err) {
      setReconstructionError(err instanceof Error ? err.message : "Failed to load reconstruction.");
    } finally {
      setReconstructionLoading(false);
      setReconstructionLoaded(true);
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
    setRebuilding(true);
    try {
      const recon = await reconstructIncident(incidentId);
      const freshIncident = await fetchIncident(incidentId);
      setIncident(freshIncident);
      applyReconstruction(recon);
      setReconstructions(await listReconstructions(incidentId).catch(() => []));
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setReconstructionError(detail || (err instanceof Error ? err.message : "Reconstruction failed."));
    } finally {
      setRebuilding(false);
    }
  }

  // Added, removed or reordered records rebuild the sequence automatically:
  // there is no separate "reconstruct" step to remember.
  useEffect(() => {
    if (!incident || !multiComtradeEnabled || !reconstructionLoaded || reconstructionLoading || rebuilding) return;
    if (reconstruction && !reconstruction.is_latest) return; // an older version is being viewed on purpose
    if (!isReconstructionStale(incident, reconstruction)) return;
    const signature = recordSetSignature(incident.records);
    if (autoRebuiltFor.current === signature) return;
    autoRebuiltFor.current = signature;
    void handleReconstruct();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [incident, reconstruction, multiComtradeEnabled, reconstructionLoaded, reconstructionLoading, rebuilding]);

  const story = useMemo<IncidentStory | null>(() => {
    if (!incident || !reconstruction) return null;
    try {
      return buildIncidentStory(incident, reconstruction);
    } catch {
      return null; // keep the page usable; the technical details still show everything
    }
  }, [incident, reconstruction]);

  function openUpload(files: File[] = []) {
    setDroppedFiles(files);
    setUploadKey((k) => k + 1);
    setUploadOpen(true);
  }

  async function handleUploaded(result: BatchUploadResponse) {
    if (result.records_created.length > 0 && result.errors.length === 0) setUploadOpen(false);
    await load();
  }

  function handleDragEnter(e: React.DragEvent) {
    if (!multiComtradeEnabled || uploadOpen || !hasFiles(e)) return;
    dragDepth.current += 1;
    setDragActive(true);
  }

  function handleDragLeave() {
    if (!dragActive) return;
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragActive(false);
  }

  function handleDrop(e: React.DragEvent) {
    if (!dragActive) return;
    e.preventDefault();
    dragDepth.current = 0;
    setDragActive(false);
    const files = Array.from(e.dataTransfer.files || []);
    if (files.length > 0) openUpload(files);
  }

  function openDetails() {
    setDetailsOpen(true);
    window.setTimeout(() => detailsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
  }

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
    if (!window.confirm("Lepas rekaman ini dari insiden? Analisa rekamannya tidak dihapus.")) return;
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
      // Same record set, new order: the automatic rebuild only watches membership.
      await handleReconstruct();
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
    if (!window.confirm("Arsipkan insiden ini? Insiden akan disembunyikan dari daftar.")) return;
    try {
      await deleteIncident(incidentId);
      navigate("/incidents");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to archive incident.");
    }
  }

  if (loading && !incident) return <div className={styles.state}>Memuat insiden…</div>;
  if (error && !incident) return <div className={styles.state}>{error}</div>;
  if (!incident) return null;

  const hasRecords = incident.records.length > 0;

  return (
    <div
      className={styles.page}
      onDragEnter={handleDragEnter}
      onDragOver={(e) => {
        if (dragActive) e.preventDefault();
      }}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {dragActive && (
        <div className={styles.dropOverlay} aria-hidden="true">
          <div className={styles.dropOverlayBox}>Lepas file COMTRADE untuk menambahkan rekaman</div>
        </div>
      )}

      <header className={styles.topBar}>
        <div className={styles.topBarInner}>
          <nav aria-label="Breadcrumb" className={styles.breadcrumb}>
            <Link to="/incidents">Insiden</Link>
            <span aria-hidden="true">/</span>
            <span className={styles.breadcrumbCurrent}>{incident.title}</span>
          </nav>
          {multiComtradeEnabled && (
            <div className={styles.actions}>
              <button type="button" className={styles.primaryButton} onClick={() => openUpload()}>
                Tambah rekaman
              </button>
            </div>
          )}
        </div>
      </header>

      <main className={styles.main}>
        {multiComtradeEnabled && (
          <p className={styles.lead}>
            Rekaman yang ditambahkan langsung dianalisa dan urutan kejadian disusun ulang otomatis.
          </p>
        )}
        {error && <div className={styles.error}>{error}</div>}
        {reconstructionError && <div className={styles.error}>{reconstructionError}</div>}

        {!multiComtradeEnabled ? (
          <section className={styles.card}>
            <p className={styles.muted}>
              Rekonstruksi multi-COMTRADE nonaktif di server ini, jadi ringkasan dan urutan kejadian tidak tersedia.
              Rekaman dan data insiden tetap bisa dilihat di Detail teknis.
            </p>
          </section>
        ) : !hasRecords ? (
          <section className={styles.emptyCard}>
            <h1 className={styles.emptyTitle}>Belum ada rekaman</h1>
            <p className={styles.muted}>
              Tarik file COMTRADE dari kejadian ini ke mana saja di halaman, atau pilih filenya. Pasangan .cfg + .dat
              dan file .cff bisa dicampur.
            </p>
            <button type="button" className={styles.primaryButton} onClick={() => openUpload()}>
              Pilih file
            </button>
          </section>
        ) : story ? (
          <>
            <SummarySection story={story} rebuilding={rebuilding} />
            <SequenceSection story={story} />
            <div className={styles.twoColumns}>
              <CauseSection cause={story.cause} />
              <ChecklistSection incidentId={incident.incident_id} items={story.checklist} onOpenDetails={openDetails} />
            </div>
            <RecordsSection incidentId={incident.incident_id} rows={story.records} episodes={episodes} />
          </>
        ) : (
          <section className={styles.card}>
            <p className={styles.muted}>
              {rebuilding || reconstructionLoading
                ? "Menyusun urutan kejadian dari rekaman…"
                : "Urutan kejadian belum bisa disusun. Lihat Detail teknis untuk rekaman dan riwayat rekonstruksi."}
            </p>
          </section>
        )}

        <section
          id="detail-teknis"
          ref={detailsRef}
          className={styles.detailsCard}
          aria-label="Detail teknis"
        >
          <button
            type="button"
            className={styles.detailsToggle}
            aria-expanded={detailsOpen}
            onClick={() => setDetailsOpen((v) => !v)}
          >
            <span className={styles.detailsHeading}>
              <span className={styles.detailsTitle}>Detail teknis</span>
              <span className={styles.muted}>
                Data insiden, urutan rekaman, hubungan antar rekaman, audit model AI, evidence, riwayat rekonstruksi
              </span>
            </span>
            <svg
              className={detailsOpen ? styles.chevronOpen : styles.chevron}
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>

          {detailsOpen && (
            <div className={styles.detailsBody}>
              <div className={styles.subsection}>
                <div className={styles.subsectionHeader}>
                  <h3>Incident details</h3>
                  <button type="button" className={styles.dangerButton} onClick={handleArchive}>
                    Arsipkan insiden
                  </button>
                </div>
                <div className={styles.headerGrid}>
                  <label className={styles.field}>
                    Title
                    <input
                      value={incident.title}
                      onChange={(e) => setIncident({ ...incident, title: e.target.value })}
                      onBlur={(e) => patchField("title", e.target.value)}
                    />
                  </label>
                  <label className={styles.field}>
                    Status
                    <select
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
                  </label>
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
              </div>

              <div className={styles.subsection}>
                <h3>Attached records</h3>
                <IncidentRecordsTable
                  records={incident.records}
                  episodes={episodes}
                  relationships={relationships}
                  manualOrderConflict={incident.missing_evidence.some((m) => m.type === "RECORD_ORDER_REQUIRES_REVIEW")}
                  onMove={handleMove}
                  onDetach={handleDetach}
                />
              </div>

              {multiComtradeEnabled && (
                <>
                  <div className={styles.subsection}>
                    <h3>Reconstruction</h3>
                    <ReconstructionControls
                      reconstruction={reconstruction}
                      reconstructions={reconstructions}
                      stale={isReconstructionStale(incident, reconstruction) && Boolean(reconstruction)}
                      staleReason="the attached records changed since this reconstruction"
                      loading={reconstructionLoading || rebuilding}
                      recordCount={incident.records.length}
                      onReconstruct={handleReconstruct}
                      onSelectVersion={(id) => loadReconstructionState(id)}
                      selectedVersionId={reconstruction?.reconstruction_id ?? null}
                    />
                  </div>
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
                        <EpisodeCards incidentId={incidentId ?? ""} episodes={episodes} records={incident.records} />
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
                </>
              )}

              <div className={styles.subsection}>
                <div className={styles.subsectionHeader}>
                  <h3>Evidence</h3>
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
                      onChange={(e) =>
                        setEvidenceDraft({ ...evidenceDraft, confidence: e.target.value as EvidenceConfidence })
                      }
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
              </div>

              <div className={styles.subsection}>
                <h3>Record collection summary</h3>
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
              </div>

              <div className={styles.subsection}>
                <h3>Missing evidence</h3>
                {incident.missing_evidence.length === 0 ? (
                  <div className={styles.empty}>Nothing flagged.</div>
                ) : (
                  <ul className={styles.missingList}>
                    {incident.missing_evidence.map((m, i) => (
                      <li key={i}>{m.description}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}
        </section>
      </main>

      {uploadOpen && incident && (
        <div className={styles.modalOverlay} onClick={() => setUploadOpen(false)}>
          <div
            className={styles.modal}
            role="dialog"
            aria-modal="true"
            aria-labelledby="upload-title"
            onClick={(e) => e.stopPropagation()}
          >
            <div className={styles.modalHeader}>
              <h2 id="upload-title">Tambah rekaman</h2>
              <button type="button" className={styles.closeButton} aria-label="Tutup" onClick={() => setUploadOpen(false)}>
                ×
              </button>
            </div>
            <p className={styles.muted}>
              Rekaman yang diunggah langsung dianalisa, lalu urutan kejadian disusun ulang. Rekaman yang valid tetap
              dilampirkan meski ada file lain yang gagal.
            </p>
            <BatchUploadPanel
              key={uploadKey}
              incidentId={incident.incident_id}
              initialFiles={droppedFiles}
              defaultPartialSuccess
              alwaysReconstruct
              onUploaded={(result) => void handleUploaded(result)}
            />
          </div>
        </div>
      )}
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
