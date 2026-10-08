import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  createIncident,
  deleteIncident,
  fetchIncident,
  listIncidents,
  updateIncident,
  uploadIncidentRecords,
  type IncidentOut,
  type IncidentStatus,
} from "../api/client";
import { formatDate, isoToMs, readFacts } from "../components/incidents/incidentStory";
import { useMultiComtradeEnabled } from "../hooks/useFeatureFlags";
import styles from "./IncidentList.module.css";

const STATUS_OPTIONS: Array<IncidentStatus | "ALL"> = [
  "ALL",
  "DRAFT",
  "OPEN",
  "UNDER_REVIEW",
  "CONFIRMED",
  "CLOSED",
  "ARCHIVED",
];

function formatTime(iso: string | null) {
  if (!iso) return "-";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

/** "GI BRINGIN · MJSNG2 · 21 Agu 2023", from the incident's first record. */
function titleFromRecords(incident: IncidentOut): { title: string; station: string | null } | null {
  const first = incident.records[0];
  if (!first) return null;
  const facts = readFacts(first);
  const when = isoToMs(first.trigger_time_iso ?? first.record_start_iso);
  const parts = [first.station_name, facts.selectedLine, when !== null ? formatDate(when) : null].filter(
    (p): p is string => Boolean(p),
  );
  return parts.length ? { title: parts.join(" · "), station: first.station_name } : null;
}

export default function IncidentList() {
  const navigate = useNavigate();
  const multiComtradeEnabled = useMultiComtradeEnabled();
  const [incidents, setIncidents] = useState<IncidentOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<IncidentStatus | "ALL">("ALL");
  const [stationFilter, setStationFilter] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [creating, setCreating] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const params: { status?: string } = {};
      if (statusFilter !== "ALL") params.status = statusFilter;
      const data = await listIncidents(params);
      setIncidents(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load incidents.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [statusFilter]);

  const stations = useMemo(() => {
    const set = new Set<string>();
    incidents.forEach((i) => {
      if (i.station_name) set.add(i.station_name);
    });
    return Array.from(set).sort();
  }, [incidents]);

  const filtered = useMemo(() => {
    if (!stationFilter) return incidents;
    return incidents.filter((i) => i.station_name === stationFilter);
  }, [incidents, stationFilter]);

  async function handleCreate() {
    setCreating(true);
    try {
      const incident = await createIncident({ title: newTitle || "Untitled incident" });
      setShowCreate(false);
      setNewTitle("");
      navigate(`/incidents/${incident.incident_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create incident.");
    } finally {
      setCreating(false);
    }
  }

  /** Dropped files become a new incident: create, upload, name it from the records, open it. */
  async function createFromFiles(files: File[]) {
    if (files.length === 0 || progress) return;
    setError(null);
    let incidentId: string | null = null;
    try {
      setProgress("Membuat insiden…");
      const created = await createIncident({ title: "Insiden baru" });
      incidentId = created.incident_id;
      setProgress(`Menganalisa ${files.length} file…`);
      const result = await uploadIncidentRecords(incidentId, files, { partialSuccess: true });
      if (result.records_created.length === 0) {
        await deleteIncident(incidentId).catch(() => undefined);
        const reasons = result.errors.map((e) => `${e.files.join(", ")}: ${e.reason}`).join(" · ");
        setError(`Tidak ada rekaman yang bisa dianalisa. ${reasons}`);
        return;
      }
      const incident = await fetchIncident(incidentId);
      const named = titleFromRecords(incident);
      if (named) {
        await updateIncident(incidentId, { title: named.title, station_name: incident.station_name ?? named.station });
      }
      navigate(`/incidents/${incidentId}`);
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail || (err instanceof Error ? err.message : "Gagal membuat insiden dari file."));
      if (incidentId) navigate(`/incidents/${incidentId}`);
    } finally {
      setProgress(null);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <div className={styles.eyebrow}>Insiden</div>
          <h1 className={styles.title}>Insiden gangguan</h1>
          <p className={styles.subtitle}>
            Satu insiden adalah satu kejadian gangguan, bisa terdiri dari beberapa rekaman: gangguan, reclose, dan
            gangguan ulang. Urutan kejadian disusun otomatis dari rekamannya.
          </p>
        </div>
        <button type="button" className={styles.secondaryButton} onClick={() => setShowCreate(true)}>
          + Insiden kosong
        </button>
      </header>

      {multiComtradeEnabled && (
        <div
          className={`${styles.dropzone} ${dragOver ? styles.dropzoneActive : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            void createFromFiles(Array.from(e.dataTransfer.files || []));
          }}
        >
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".cfg,.dat,.cff"
            className={styles.hiddenInput}
            aria-label="Pilih file COMTRADE untuk insiden baru"
            onChange={(e) => void createFromFiles(Array.from(e.target.files || []))}
          />
          {progress ? (
            <p className={styles.dropTitle} role="status">
              {progress}
            </p>
          ) : (
            <>
              <p className={styles.dropTitle}>Tarik file COMTRADE dari satu kejadian ke sini</p>
              <p className={styles.dropHint}>
                Pasangan .cfg + .dat atau file .cff, boleh beberapa rekaman sekaligus. Insiden baru langsung dibuat
                dan dianalisa.
              </p>
              <button type="button" className={styles.primaryButton} onClick={() => inputRef.current?.click()}>
                Pilih file
              </button>
            </>
          )}
        </div>
      )}

      <div className={styles.filters}>
        <label className={styles.filterField}>
          Status
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as IncidentStatus | "ALL")}>
            {STATUS_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <label className={styles.filterField}>
          Station
          <select value={stationFilter} onChange={(e) => setStationFilter(e.target.value)}>
            <option value="">All stations</option>
            {stations.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      {loading ? (
        <div className={styles.empty}>Memuat insiden…</div>
      ) : filtered.length === 0 ? (
        <div className={styles.empty}>Belum ada insiden.</div>
      ) : (
        <div className={styles.grid}>
          {filtered.map((incident) => (
            <button
              key={incident.incident_id}
              className={styles.card}
              onClick={() => navigate(`/incidents/${incident.incident_id}`)}
              type="button"
            >
              <div className={styles.cardTop}>
                <span className={styles.cardTitle}>{incident.title}</span>
                <span className={`${styles.statusBadge} ${styles[`status_${incident.status}`] ?? ""}`}>
                  {incident.status}
                </span>
              </div>
              <div className={styles.cardMeta}>
                <span>{incident.station_name || "Station not set"}</span>
                {incident.bay_name && <span>· {incident.bay_name}</span>}
                {incident.asset_name && <span>· {incident.asset_name}</span>}
              </div>
              <div className={styles.cardMeta}>
                <span>{incident.observed_summary.record_count} record(s)</span>
                <span>· Clock: {incident.clock_assessment}</span>
              </div>
              <div className={styles.cardTime}>
                {formatTime(incident.incident_start_iso)} → {formatTime(incident.incident_end_iso)}
              </div>
            </button>
          ))}
        </div>
      )}

      {showCreate && (
        <div className={styles.modalOverlay} onClick={() => setShowCreate(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <h2>Insiden kosong</h2>
            <label className={styles.modalField}>
              Title
              <input
                type="text"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                placeholder="e.g. GI COMAL Trafo 1 trip 28/09/2024"
                autoFocus
              />
            </label>
            <div className={styles.modalActions}>
              <button type="button" onClick={() => setShowCreate(false)}>
                Batal
              </button>
              <button type="button" className={styles.primaryButton} onClick={handleCreate} disabled={creating}>
                {creating ? "Membuat…" : "Buat"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
