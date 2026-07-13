import { useEffect, useState } from "react";
import {
  attachIncidentRecord,
  createIncident,
  listIncidents,
  type IncidentOut,
  type RecordAttachmentRole,
} from "../../api/client";
import styles from "./AddToIncidentModal.module.css";

const ROLE_OPTIONS: RecordAttachmentRole[] = [
  "PRIMARY",
  "SUPPORTING",
  "REMOTE_END",
  "BACKUP_RELAY",
  "DFR_EXTERNAL",
  "OTHER",
  "UNKNOWN",
];

interface Props {
  analysisId: string;
  stationName?: string;
  onClose: () => void;
}

export default function AddToIncidentModal({ analysisId, stationName, onClose }: Props) {
  const [mode, setMode] = useState<"existing" | "new">("existing");
  const [incidents, setIncidents] = useState<IncidentOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedIncidentId, setSelectedIncidentId] = useState("");
  const [newTitle, setNewTitle] = useState("");
  const [bayName, setBayName] = useState("");
  const [role, setRole] = useState<RecordAttachmentRole>("PRIMARY");
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [stationMismatch, setStationMismatch] = useState(false);

  useEffect(() => {
    listIncidents()
      .then((data) => {
        setIncidents(data);
        if (data.length === 0) setMode("new");
      })
      .catch(() => setError("Failed to load incidents."))
      .finally(() => setLoading(false));
  }, []);

  async function handleSubmit(overrideWarnings = false) {
    setSubmitting(true);
    setError(null);
    try {
      let incidentId = selectedIncidentId;
      if (mode === "new") {
        const incident = await createIncident({ title: newTitle || "Untitled incident", station_name: stationName || null });
        incidentId = incident.incident_id;
      }
      if (!incidentId) {
        setError("Select or create an incident first.");
        setSubmitting(false);
        return;
      }
      await attachIncidentRecord(incidentId, {
        analysis_id: analysisId,
        attachment_role: role,
        bay_name: bayName || null,
        operator_notes: notes || null,
        override_warnings: overrideWarnings,
      });
      setSuccess(incidentId);
    } catch (err: unknown) {
      const status = (err as { response?: { status?: number; data?: { detail?: string } } })?.response?.status;
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      if (status === 409 && detail && detail.toLowerCase().includes("station mismatch")) {
        setStationMismatch(true);
        setError(detail);
      } else {
        setError(detail || (err instanceof Error ? err.message : "Failed to attach record."));
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <h2>Add to Incident</h2>

        {success ? (
          <div className={styles.successBlock}>
            <p>Record attached to the incident.</p>
            <div className={styles.actions}>
              <a href={`/incidents/${success}`} className={styles.primaryButton}>
                Open incident
              </a>
              <button type="button" onClick={onClose}>
                Close
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className={styles.modeTabs}>
              <button
                type="button"
                className={mode === "existing" ? styles.activeTab : ""}
                onClick={() => setMode("existing")}
                disabled={incidents.length === 0}
              >
                Existing incident
              </button>
              <button type="button" className={mode === "new" ? styles.activeTab : ""} onClick={() => setMode("new")}>
                New incident
              </button>
            </div>

            {mode === "existing" ? (
              loading ? (
                <div className={styles.hint}>Loading incidents…</div>
              ) : incidents.length === 0 ? (
                <div className={styles.hint}>No incidents yet. Create a new one.</div>
              ) : (
                <label className={styles.field}>
                  Incident
                  <select value={selectedIncidentId} onChange={(e) => setSelectedIncidentId(e.target.value)}>
                    <option value="">Select an incident…</option>
                    {incidents.map((i) => (
                      <option key={i.incident_id} value={i.incident_id}>
                        {i.title} ({i.status})
                      </option>
                    ))}
                  </select>
                </label>
              )
            ) : (
              <label className={styles.field}>
                Title
                <input value={newTitle} onChange={(e) => setNewTitle(e.target.value)} placeholder="New incident title" autoFocus />
              </label>
            )}

            <label className={styles.field}>
              Attachment role
              <select value={role} onChange={(e) => setRole(e.target.value as RecordAttachmentRole)}>
                {ROLE_OPTIONS.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </label>

            <label className={styles.field}>
              Bay name
              <input value={bayName} onChange={(e) => setBayName(e.target.value)} placeholder="Optional" />
            </label>

            <label className={styles.field}>
              Notes
              <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={2} placeholder="Optional" />
            </label>

            {error && <div className={styles.error}>{error}</div>}

            <div className={styles.actions}>
              <button type="button" onClick={onClose}>
                Cancel
              </button>
              {stationMismatch ? (
                <button type="button" className={styles.warnButton} onClick={() => handleSubmit(true)} disabled={submitting}>
                  Attach anyway
                </button>
              ) : (
                <button type="button" className={styles.primaryButton} onClick={() => handleSubmit(false)} disabled={submitting}>
                  {submitting ? "Attaching…" : "Attach"}
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
