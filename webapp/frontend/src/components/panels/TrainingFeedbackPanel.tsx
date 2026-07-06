import { useEffect, useMemo, useState } from "react";
import {
  clearTrainingArchive,
  downloadTrainingArchive,
  fetchTrainingStatus,
  submitTrainingFeedback,
  type TrainingStatus,
} from "../../api/client";
import styles from "./TrainingFeedbackPanel.module.css";

const TOKEN_KEY = "base_ai_tfa_training_admin_token";

const LABEL_OPTIONS = [
  "",
  "PETIR",
  "LAYANG",
  "POHON",
  "HEWAN",
  "BENDA_ASING",
  "KONDUKTOR",
  "PERALATAN",
  "INTERNAL_FAULT",
  "THROUGH_FAULT",
  "INRUSH",
  "OVEREXCITATION",
  "MAL_OPERATE",
  "NO_FAULT",
  "PERLU_INVESTIGASI",
];

const FAULT_TYPE_OPTIONS = [
  "",
  "transient",
  "permanent",
  "internal",
  "through_fault",
  "inrush",
  "no_fault",
  "unknown",
];

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let idx = 0;
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }
  return `${value.toFixed(value >= 10 || idx === 0 ? 0 : 1)} ${units[idx]}`;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export default function TrainingFeedbackPanel({
  analysisId,
  relayType,
}: {
  analysisId: string;
  relayType: string;
}) {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY) ?? "");
  const [operator, setOperator] = useState("");
  const [aiCorrect, setAiCorrect] = useState<boolean | null>(null);
  const [actualLabel, setActualLabel] = useState("");
  const [faultType, setFaultType] = useState("");
  const [includeForTraining, setIncludeForTraining] = useState(true);
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function refreshStatus() {
    try {
      setStatus(await fetchTrainingStatus());
    } catch {
      setStatus(null);
    }
  }

  useEffect(() => {
    refreshStatus();
  }, []);

  useEffect(() => {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  }, [token]);

  const statusText = useMemo(() => {
    if (!status) return "status unavailable";
    if (!status.enabled) return "retention off";
    return `${status.raw_record_count} raw | ${status.feedback_count} labels | ${formatBytes(status.total_bytes)}`;
  }, [status]);

  async function handleSubmit() {
    setError(null);
    setMessage(null);
    if (!token.trim()) {
      setError("Admin token is required to submit training feedback.");
      return;
    }
    setBusy(true);
    try {
      await submitTrainingFeedback(token.trim(), {
        analysis_id: analysisId,
        relay_type: relayType,
        ai_correct: aiCorrect,
        actual_label: actualLabel,
        fault_type: faultType,
        include_for_training: includeForTraining,
        operator,
        notes,
      });
      setMessage("Feedback saved for the training dataset.");
      setNotes("");
      await refreshStatus();
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(detail ?? "Failed to save feedback.");
    } finally {
      setBusy(false);
    }
  }

  async function handleExport() {
    setError(null);
    setMessage(null);
    if (!token.trim()) {
      setError("Admin token is required to export training data.");
      return;
    }
    setBusy(true);
    try {
      const blob = await downloadTrainingArchive(token.trim());
      downloadBlob(blob, `base-ai-tfa-training-data-${new Date().toISOString().slice(0, 10)}.zip`);
      setMessage("Training archive downloaded.");
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(detail ?? "Failed to export training archive.");
    } finally {
      setBusy(false);
    }
  }

  async function handleClear() {
    setError(null);
    setMessage(null);
    if (!token.trim()) {
      setError("Admin token is required to clear training data.");
      return;
    }
    const confirmed = window.confirm(
      "Delete retained raw uploads and feedback from the server? Download the archive first.",
    );
    if (!confirmed) return;

    setBusy(true);
    try {
      await clearTrainingArchive(token.trim());
      setMessage("Server-side training archive cleared.");
      await refreshStatus();
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(detail ?? "Failed to clear training archive.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className={styles.panel}>
      <div className={styles.header}>
        <div>
          <h2 className={styles.title}>Training Feedback</h2>
          <p className={styles.subtitle}>Retain this case for local review and future retraining.</p>
        </div>
        <span className={styles.statusPill}>{statusText}</span>
      </div>

      <div className={styles.grid}>
        <label className={`${styles.field} ${styles.wide}`}>
          <span className={styles.label}>Admin token</span>
          <input
            className={styles.input}
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="TRAINING_ADMIN_TOKEN"
          />
        </label>

        <label className={styles.field}>
          <span className={styles.label}>Operator</span>
          <input
            className={styles.input}
            value={operator}
            onChange={(e) => setOperator(e.target.value)}
            placeholder="Nama engineer"
          />
        </label>

        <label className={styles.field}>
          <span className={styles.label}>Actual label</span>
          <select className={styles.select} value={actualLabel} onChange={(e) => setActualLabel(e.target.value)}>
            {LABEL_OPTIONS.map((option) => (
              <option value={option} key={option || "empty"}>
                {option || "Pilih / belum tahu"}
              </option>
            ))}
          </select>
        </label>

        <div className={styles.field}>
          <span className={styles.label}>AI result</span>
          <div className={styles.segmented}>
            <button
              className={aiCorrect === true ? styles.active : ""}
              type="button"
              onClick={() => setAiCorrect(true)}
            >
              Correct
            </button>
            <button
              className={aiCorrect === false ? styles.active : ""}
              type="button"
              onClick={() => setAiCorrect(false)}
            >
              Wrong
            </button>
            <button
              className={aiCorrect === null ? styles.active : ""}
              type="button"
              onClick={() => setAiCorrect(null)}
            >
              Unsure
            </button>
          </div>
        </div>

        <label className={styles.field}>
          <span className={styles.label}>Fault type</span>
          <select className={styles.select} value={faultType} onChange={(e) => setFaultType(e.target.value)}>
            {FAULT_TYPE_OPTIONS.map((option) => (
              <option value={option} key={option || "empty"}>
                {option || "Pilih / belum tahu"}
              </option>
            ))}
          </select>
        </label>

        <label className={`${styles.field} ${styles.wide}`}>
          <span className={styles.label}>Notes</span>
          <textarea
            className={styles.textarea}
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Catatan lapangan, hasil inspeksi, alasan koreksi label, atau issue locus/AI."
          />
        </label>

        <label className={`${styles.checkRow} ${styles.wide}`}>
          <input
            type="checkbox"
            checked={includeForTraining}
            onChange={(e) => setIncludeForTraining(e.target.checked)}
          />
          Include this case when building the next training dataset
        </label>
      </div>

      <div className={styles.actions}>
        <button className={styles.primary} type="button" onClick={handleSubmit} disabled={busy}>
          Save feedback
        </button>
        <button className={styles.secondary} type="button" onClick={handleExport} disabled={busy}>
          Download archive
        </button>
        <button className={styles.danger} type="button" onClick={handleClear} disabled={busy}>
          Clear archive
        </button>
      </div>

      {message && <div className={styles.message}>{message}</div>}
      {error && <div className={`${styles.message} ${styles.error}`}>{error}</div>}
    </section>
  );
}
