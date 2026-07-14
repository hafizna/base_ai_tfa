import { useRef, useState } from "react";
import {
  previewFilePairing,
  uploadIncidentRecords,
  type BatchUploadResponse,
  type FilePairPreview,
} from "../../api/client";
import styles from "./BatchUploadPanel.module.css";

interface Props {
  incidentId: string;
  onUploaded: (result: BatchUploadResponse, autoReconstruct: boolean) => void;
}

export default function BatchUploadPanel({ incidentId, onUploaded }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [preview, setPreview] = useState<FilePairPreview | null>(null);
  const [partialSuccess, setPartialSuccess] = useState(false);
  const [overrideWarnings, setOverrideWarnings] = useState(false);
  const [autoReconstruct, setAutoReconstruct] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<BatchUploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function setSelectedFiles(next: File[]) {
    setFiles(next);
    setResult(null);
    setError(null);
    setPreview(next.length > 0 ? previewFilePairing(next) : null);
  }

  function handleFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    setSelectedFiles(Array.from(e.target.files || []));
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    setSelectedFiles(Array.from(e.dataTransfer.files || []));
  }

  async function handleUpload() {
    if (files.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      const response = await uploadIncidentRecords(incidentId, files, { partialSuccess, overrideWarnings });
      setResult(response);
      onUploaded(response, autoReconstruct);
      if (response.reconstruction_status !== "aborted_atomic") {
        setFiles([]);
        setPreview(null);
        if (inputRef.current) inputRef.current.value = "";
      }
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail || (err instanceof Error ? err.message : "Batch upload failed."));
    } finally {
      setUploading(false);
    }
  }

  const hasBlockingErrors = (preview?.errors.length ?? 0) > 0;
  const canUpload = files.length > 0 && (partialSuccess || !hasBlockingErrors);

  return (
    <div className={styles.panel}>
      <div
        className={`${styles.dropzone} ${dragOver ? styles.dropzoneActive : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".cfg,.dat,.cff"
          onChange={handleFileInput}
          className={styles.hiddenInput}
          aria-label="Select COMTRADE files"
        />
        <p>Drag and drop .cfg/.dat pairs or .cff files here, or click to browse.</p>
        {files.length > 0 && <p className={styles.fileCount}>{files.length} file(s) selected</p>}
      </div>

      {preview && (
        <div className={styles.previewWrap}>
          <h4>Detected record groups (preview — backend pairing is authoritative)</h4>
          {preview.groups.length === 0 && preview.errors.length === 0 && (
            <div className={styles.empty}>No recognizable COMTRADE files selected.</div>
          )}
          {preview.groups.length > 0 && (
            <ul className={styles.groupList}>
              {preview.groups.map((g, i) => (
                <li key={i} className={styles.groupItem}>
                  <span className={styles.groupBadge}>{g.kind === "cff" ? "CFF" : "CFG+DAT"}</span>
                  <span>{g.files.join(" + ")}</span>
                </li>
              ))}
            </ul>
          )}
          {preview.errors.length > 0 && (
            <ul className={styles.errorList}>
              {preview.errors.map((e, i) => (
                <li key={i} className={styles.errorItem}>
                  <strong>{e.files.join(", ")}:</strong> {e.reason}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className={styles.controlsRow}>
        <label className={styles.checkboxLabel}>
          <input type="checkbox" checked={partialSuccess} onChange={(e) => setPartialSuccess(e.target.checked)} />
          Partial-success mode (default: atomic — any invalid pair aborts the whole batch)
        </label>
        <label className={styles.checkboxLabel}>
          <input type="checkbox" checked={overrideWarnings} onChange={(e) => setOverrideWarnings(e.target.checked)} />
          Attach despite station-name mismatch with this incident
        </label>
        <label className={styles.checkboxLabel}>
          <input type="checkbox" checked={autoReconstruct} onChange={(e) => setAutoReconstruct(e.target.checked)} />
          Automatically reconstruct after upload
        </label>
      </div>

      {hasBlockingErrors && !partialSuccess && (
        <div className={styles.warning}>
          Pairing errors detected. In atomic mode (default), the entire batch will be rejected. Enable
          partial-success mode to upload only the valid groups.
        </div>
      )}

      {error && <div className={styles.error}>{error}</div>}

      <div className={styles.actions}>
        <button type="button" className={styles.uploadButton} onClick={handleUpload} disabled={!canUpload || uploading}>
          {uploading ? "Uploading…" : `Upload ${files.length || ""} file(s)`}
        </button>
      </div>

      {result && (
        <div className={styles.resultWrap}>
          <div className={`${styles.resultStatus} ${styles[`status_${result.reconstruction_status}`] ?? ""}`}>
            {result.reconstruction_status.toUpperCase().replace(/_/g, " ")}
          </div>
          {result.records_created.length > 0 && (
            <ul className={styles.resultList}>
              {result.records_created.map((r, i) => (
                <li key={i} className={styles.resultItem}>
                  <span className={styles.resultOk}>✓ {r.source_files.join(" + ")}</span>
                  <span className={styles.resultMeta}>
                    analysis_id: {r.analysis_id} · incident_record_id: {r.incident_record_id}
                  </span>
                </li>
              ))}
            </ul>
          )}
          {result.errors.length > 0 && (
            <ul className={styles.errorList}>
              {result.errors.map((e, i) => (
                <li key={i} className={styles.errorItem}>
                  <strong>{e.files.join(", ")}:</strong> {e.reason}
                </li>
              ))}
            </ul>
          )}
          {result.errors.some((e) => e.reason.toLowerCase().includes("station mismatch")) && !overrideWarnings && (
            <div className={styles.warning}>
              One or more records report a different station name than this incident. Check "Attach despite
              station-name mismatch" above and upload again if that mismatch is expected (e.g. remote-end record).
            </div>
          )}
        </div>
      )}

    </div>
  );
}
