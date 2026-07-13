import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createIncident, listIncidents, type IncidentOut, type IncidentStatus } from "../api/client";
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

export default function IncidentList() {
  const navigate = useNavigate();
  const [incidents, setIncidents] = useState<IncidentOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<IncidentStatus | "ALL">("ALL");
  const [stationFilter, setStationFilter] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [creating, setCreating] = useState(false);

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

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <div className={styles.eyebrow}>Incidents</div>
          <h1 className={styles.title}>Incident workspace</h1>
          <p className={styles.subtitle}>
            Group related COMTRADE records into one incident. Record relationships are manual in this stage —
            automatic reconstruction has not yet been implemented.
          </p>
        </div>
        <button type="button" className={styles.primaryButton} onClick={() => setShowCreate(true)}>
          + New incident
        </button>
      </header>

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
        <div className={styles.empty}>Loading incidents…</div>
      ) : filtered.length === 0 ? (
        <div className={styles.empty}>No incidents yet. Create one to start grouping records.</div>
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
            <h2>New incident</h2>
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
                Cancel
              </button>
              <button type="button" className={styles.primaryButton} onClick={handleCreate} disabled={creating}>
                {creating ? "Creating…" : "Create"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
