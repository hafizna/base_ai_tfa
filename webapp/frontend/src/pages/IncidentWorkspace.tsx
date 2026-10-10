import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  deleteIncident,
  detachIncidentRecord,
  fetchIncident,
  fetchIncidentEpisodes,
  fetchIncidentRelationships,
  fetchReconstruction,
  generateIncidentReport,
  reconstructIncident,
  refreshIncidentSnapshots,
  updateIncident,
  type BatchUploadResponse,
  type FaultEpisodeOut,
  type IncidentOut,
  type IncidentRecordOut,
  type ReconstructionOut,
  type RecordRelationshipOut,
} from "../api/client";
import BatchUploadPanel from "../components/incidents/BatchUploadPanel";
import { buildIncidentStory, isReconstructionStale, type IncidentStory } from "../components/incidents/incidentStory";
import {
  CauseSection,
  ChecklistSection,
  RecordsSection,
  SequenceSection,
  SummarySection,
} from "../components/incidents/IncidentStoryView";
import TechnicalDetails from "../components/incidents/TechnicalDetails";
import { useMultiComtradeEnabled } from "../hooks/useFeatureFlags";
import styles from "./IncidentWorkspace.module.css";

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

  const multiComtradeEnabled = useMultiComtradeEnabled();

  const [reconstruction, setReconstruction] = useState<ReconstructionOut | null>(null);
  const [relationships, setRelationships] = useState<RecordRelationshipOut[]>([]);
  const [episodes, setEpisodes] = useState<FaultEpisodeOut[]>([]);
  const [reconstructionLoaded, setReconstructionLoaded] = useState(false);
  const [reconstructionLoading, setReconstructionLoading] = useState(false);
  const [reconstructionError, setReconstructionError] = useState<string | null>(null);
  const [rebuilding, setRebuilding] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [printing, setPrinting] = useState(false);
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
    setRelationships(recon?.relationships ?? []);
    setEpisodes(recon?.episodes ?? []);
  }

  async function loadReconstructionState(targetReconstructionId?: string) {
    if (!incidentId || !multiComtradeEnabled) return;
    setReconstructionLoading(true);
    setReconstructionError(null);
    try {
      const recon = await fetchReconstruction(incidentId, targetReconstructionId).catch((err) => {
        if (err?.response?.status === 404) return null;
        throw err;
      });
      if (recon) {
        applyReconstruction({
          ...recon,
          relationships: recon.relationships ?? (await fetchIncidentRelationships(incidentId).catch(() => [])),
          episodes: recon.episodes ?? (await fetchIncidentEpisodes(incidentId).catch(() => [])),
        });
      } else {
        applyReconstruction(null);
      }
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

  async function handleSave(fields: Record<string, unknown>) {
    if (!incidentId || !incident) return;
    try {
      const stationChanged = (fields.station_name ?? null) !== (incident.station_name ?? null);
      const updated = await updateIncident(incidentId, fields);
      setIncident(updated);
      // The incident's substation picks the end the story is told from.
      if (stationChanged && incident.records.length > 0) await handleReconstruct();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update incident.");
    }
  }

  // Analyse the attached records again with the current engine (their
  // snapshots were taken when they were attached), then rebuild the sequence.
  async function handleRefreshAnalyses() {
    if (!incidentId) return;
    setRefreshing(true);
    try {
      await refreshIncidentSnapshots(incidentId);
      await handleReconstruct();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyse the records again.");
    } finally {
      setRefreshing(false);
    }
  }

  // The PDF prints the story this page shows, so it is built from the same
  // reconstruction the reader is looking at.
  async function handlePrint() {
    if (!incidentId || !incident || !story) return;
    setPrinting(true);
    try {
      const blob = await generateIncidentReport(incidentId, story);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `laporan_insiden_${incident.title.replace(/[^\w-]+/g, "_").replace(/^_+|_+$/g, "") || incidentId.slice(0, 8)}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Gagal membuat laporan PDF insiden.");
    } finally {
      setPrinting(false);
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
              {story && (
                <button
                  type="button"
                  className={styles.secondaryButton}
                  disabled={printing || rebuilding}
                  onClick={() => void handlePrint()}
                >
                  {printing ? "Menyiapkan PDF…" : "Cetak laporan"}
                </button>
              )}
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
          {!detailsOpen && (
            <button
              type="button"
              className={styles.detailsToggle}
              aria-expanded={false}
              onClick={() => setDetailsOpen(true)}
            >
              <span className={styles.detailsHeading}>
                <span className={styles.detailsTitle}>Detail teknis</span>
                <span className={styles.muted}>
                  Penalaran tiap gangguan, urutan sinyal, rekaman, dan data insiden
                </span>
              </span>
              <svg
                className={styles.chevron}
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
          )}

          {detailsOpen && (
            <div className={styles.detailsBody}>
              <TechnicalDetails
                incident={incident}
                reconstruction={reconstruction}
                episodes={episodes}
                relationships={relationships}
                story={story}
                refreshing={refreshing}
                onHide={() => setDetailsOpen(false)}
                onAddRecords={() => openUpload()}
                onDetach={(id) => void handleDetach(id)}
                onSave={handleSave}
                onArchive={() => void handleArchive()}
                onRefreshAnalyses={() => void handleRefreshAnalyses()}
                onRelationshipChanged={() => void loadReconstructionState(reconstruction?.reconstruction_id)}
              />
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
