import { useEffect, useRef, useState } from "react";
import {
  previewFilePairing,
  uploadIncidentRecords,
  type BatchUploadResponse,
  type FilePairPreview,
} from "../../api/client";
import { filesFromDrop, keepSupported } from "../../api/uploadFiles";
import styles from "./BatchUploadPanel.module.css";

interface Props {
  incidentId: string;
  onUploaded: (result: BatchUploadResponse, autoReconstruct: boolean) => void;
  /** Files dropped elsewhere on the page, preselected here. */
  initialFiles?: File[];
  defaultPartialSuccess?: boolean;
  /** The caller always reconstructs after an upload, so the checkbox is hidden. */
  alwaysReconstruct?: boolean;
}

export default function BatchUploadPanel({
  incidentId,
  onUploaded,
  initialFiles,
  defaultPartialSuccess = false,
  alwaysReconstruct = false,
}: Props) {
  const initial = keepSupported(initialFiles ?? []);
  const [files, setFiles] = useState<File[]>(initial.accepted);
  const [skipped, setSkipped] = useState(initial.skipped);
  const [preview, setPreview] = useState<FilePairPreview | null>(() =>
    initial.accepted.length > 0 ? previewFilePairing(initial.accepted) : null,
  );
  const [partialSuccess, setPartialSuccess] = useState(defaultPartialSuccess);
  const [overrideWarnings, setOverrideWarnings] = useState(false);
  const [autoReconstruct, setAutoReconstruct] = useState(alwaysReconstruct);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<BatchUploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Not in React's input typings: lets the picker choose a whole folder.
    folderInputRef.current?.setAttribute("webkitdirectory", "");
  }, []);

  function setSelectedFiles(picked: File[]) {
    const { accepted, skipped: dropped } = keepSupported(picked);
    setFiles(accepted);
    setSkipped(dropped);
    setResult(null);
    setError(null);
    setPreview(accepted.length > 0 ? previewFilePairing(accepted) : null);
  }

  function handleFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    setSelectedFiles(Array.from(e.target.files || []));
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    void filesFromDrop(e.dataTransfer).then(setSelectedFiles);
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
          accept=".cfg,.dat,.cff,.evt"
          onChange={handleFileInput}
          className={styles.hiddenInput}
          aria-label="Select COMTRADE files"
        />
        <input
          ref={folderInputRef}
          type="file"
          multiple
          onChange={handleFileInput}
          className={styles.hiddenInput}
          aria-label="Pilih folder rekaman"
        />
        <p>Drag and drop .cfg/.dat pairs or .cff files here, or click to browse.</p>
        <p className={styles.folderHint}>
          Bisa juga satu folder insiden sekaligus: nama folder tiap rele ikut terbaca, dan daftar event MiCOM
          (.evt) ikut disimpan.{" "}
          <button
            type="button"
            className={styles.folderButton}
            onClick={(e) => {
              e.stopPropagation();
              folderInputRef.current?.click();
            }}
          >
            Pilih folder
          </button>
        </p>
        {files.length > 0 && <p className={styles.fileCount}>{files.length} file(s) selected</p>}
        {skipped > 0 && (
          <p className={styles.fileCount}>{skipped} file lain dilewati (foto, setting, arsip — bukan rekaman)</p>
        )}
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
                  <span className={styles.groupBadge}>{g.kind === "cff" ? "CFF" : g.kind === "evt" ? "EVT" : "CFG+DAT"}</span>
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
        {!alwaysReconstruct && (
          <label className={styles.checkboxLabel}>
            <input type="checkbox" checked={autoReconstruct} onChange={(e) => setAutoReconstruct(e.target.checked)} />
            Automatically reconstruct after upload
          </label>
        )}
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
          {(result.event_files ?? []).length > 0 && (
            <ul className={styles.resultList}>
              {(result.event_files ?? []).map((path) => (
                <li key={path} className={styles.resultItem}>
                  <span className={styles.resultOk}>✓ {path}</span>
                  <span className={styles.resultMeta}>daftar event disimpan dengan insiden</span>
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
              One or more records report a station name that differs from this incident (see the exact names in
              the error message above). Only check "Attach despite station-name mismatch" and re-upload if this is
              the same substation written differently (e.g. "Mojosongo" vs "GI MOJOSONGO"). If the record is
              actually from a different bay/line — e.g. an external DFR that captured two lines in the same
              window — attach it to a separate incident instead of overriding.
            </div>
          )}
        </div>
      )}

    </div>
  );
}
