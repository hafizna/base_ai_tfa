import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAnalysis } from "../context/AnalysisContext";
import { uploadComtrade } from "../api/client";
import styles from "./Upload.module.css";

const RELAY_LABELS: Record<string, string> = {
  "21": "21 - Distance",
  "87L": "87L - Differential Line",
  "87T": "87T / REF",
  OCR: "50/51 / GFR",
  REF: "REF",
  SBEF: "SBEF",
};

interface DetectionSuggestion {
  analysisId: string;
  suggestedType: string;
  confidence: number;
}

export default function Upload() {
  const { relayType } = useAnalysis();
  const navigate = useNavigate();

  const cfgRef = useRef<HTMLInputElement>(null);
  const datRef = useRef<HTMLInputElement>(null);

  const [cfgFile, setCfgFile] = useState<File | null>(null);
  const [datFile, setDatFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestion, setSuggestion] = useState<DetectionSuggestion | null>(null);

  if (!relayType) {
    navigate("/");
    return null;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!cfgFile || !datFile) {
      setError("Please select both .cfg and .dat files.");
      return;
    }

    setError(null);
    setLoading(true);

    try {
      const data = await uploadComtrade(cfgFile, datFile);
      const suggested = data.suggested_relay_type;
      if (suggested && suggested !== relayType) {
        setSuggestion({
          analysisId: data.analysis_id,
          suggestedType: suggested,
          confidence: data.detection_confidence ?? 0,
        });
      } else {
        navigate(`/workspace/${relayType}/${data.analysis_id}`);
      }
    } catch (err: unknown) {
      const response = (err as { response?: { data?: { detail?: string } } }).response;
      const msg = response?.data?.detail
        ?? (response
          ? "Upload failed. Make sure the .cfg and .dat belong to the same COMTRADE record."
          : "Cannot reach the analysis API. In deployment, this usually means the frontend is pointing to the wrong backend URL.");
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.glowTop} />
      <div className={styles.glowBottom} />

      <button className={styles.back} onClick={() => navigate("/")} type="button">
        Back to relay selection
      </button>

      <div className={styles.card}>
        <div className={styles.badge}>{RELAY_LABELS[relayType]}</div>
        <h1 className={styles.title}>Upload COMTRADE Files</h1>
        <p className={styles.hint}>
          Select the matching <code>.cfg</code> and <code>.dat</code> files from your relay or DFR recorder.
        </p>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.dropRow}>
            <DropZone
              label=".cfg"
              accept=".cfg,.CFG"
              file={cfgFile}
              inputRef={cfgRef}
              onChange={setCfgFile}
            />
            <DropZone
              label=".dat"
              accept=".dat,.DAT"
              file={datFile}
              inputRef={datRef}
              onChange={setDatFile}
            />
          </div>

          {error && <div className={styles.error}>{error}</div>}

          <button
            type="submit"
            className={styles.submit}
            disabled={loading || !cfgFile || !datFile}
          >
            {loading ? "Parsing..." : "Analyze"}
          </button>
        </form>

        {suggestion && (
          <div style={{
            marginTop: 20,
            padding: "16px 18px",
            background: "#fffbeb",
            border: "1px solid #fbbf24",
            borderRadius: 10,
            fontSize: "0.85rem",
            color: "#92400e",
          }}>
            <div style={{ fontWeight: 600, marginBottom: 6 }}>
              Protection type detected from digital channels
            </div>
            <div style={{ marginBottom: 12, lineHeight: 1.6 }}>
              Recording signals match{" "}
              <strong>{RELAY_LABELS[suggestion.suggestedType] ?? suggestion.suggestedType}</strong>{" "}
              ({Math.round(suggestion.confidence * 100)}% confidence), but you selected{" "}
              <strong>{RELAY_LABELS[relayType] ?? relayType}</strong>.
              Which analysis would you like to open?
            </div>
            <div style={{ display: "flex", gap: 10 }}>
              <button
                style={{
                  flex: 1,
                  padding: "8px 0",
                  background: "#f59e0b",
                  color: "#fff",
                  border: "none",
                  borderRadius: 7,
                  fontWeight: 600,
                  cursor: "pointer",
                  fontSize: "0.83rem",
                }}
                onClick={() => navigate(`/workspace/${suggestion.suggestedType}/${suggestion.analysisId}`)}
              >
                Use detected: {RELAY_LABELS[suggestion.suggestedType] ?? suggestion.suggestedType}
              </button>
              <button
                style={{
                  flex: 1,
                  padding: "8px 0",
                  background: "#fff",
                  color: "#92400e",
                  border: "1px solid #fbbf24",
                  borderRadius: 7,
                  fontWeight: 600,
                  cursor: "pointer",
                  fontSize: "0.83rem",
                }}
                onClick={() => navigate(`/workspace/${relayType}/${suggestion.analysisId}`)}
              >
                Keep selected: {RELAY_LABELS[relayType] ?? relayType}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function DropZone({
  label,
  accept,
  file,
  inputRef,
  onChange,
}: {
  label: string;
  accept: string;
  file: File | null;
  inputRef: React.RefObject<HTMLInputElement | null>;
  onChange: (f: File) => void;
}) {
  const [over, setOver] = useState(false);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files[0];
    if (f) onChange(f);
  }

  return (
    <div
      className={`${styles.dropzone} ${over ? styles.dropzoneOver : ""} ${file ? styles.dropzoneFilled : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onChange(f);
        }}
      />
      <span className={styles.dropLabel}>{label}</span>
      {file ? (
        <span className={styles.fileName}>{file.name}</span>
      ) : (
        <span className={styles.dropHint}>Click or drag file here</span>
      )}
    </div>
  );
}
