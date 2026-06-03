import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAnalysis } from "../context/AnalysisContext";
import { uploadComtrade, uploadTwsCdb } from "../api/client";
import styles from "./Upload.module.css";

const RELAY_LABELS: Record<string, string> = {
  "21": "21 - Distance",
  "87L": "87L - Differential Line",
  CCP: "CCP / Stub Differential",
  "87T": "87T / REF",
  OCR: "50/51 / GFR",
  REF: "REF",
  SBEF: "SBEF",
  TWS_FL: "TWS FL - Traveling Wave",
};

interface DetectionSuggestion {
  analysisId: string;
  suggestedType: string;
  confidence: number;
}

export default function Upload() {
  const { relayType } = useAnalysis();
  const navigate = useNavigate();

  const comtradeRef = useRef<HTMLInputElement>(null);
  const cdbRef = useRef<HTMLInputElement>(null);

  const [cfgFile, setCfgFile] = useState<File | null>(null);
  const [datFile, setDatFile] = useState<File | null>(null);
  const [cffFile, setCffFile] = useState<File | null>(null);
  const [cdbFile, setCdbFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestion, setSuggestion] = useState<DetectionSuggestion | null>(null);
  const comtradeFiles = [cffFile, cfgFile, datFile].filter((file): file is File => Boolean(file));
  const comtradeReady = Boolean(cffFile || (cfgFile && datFile));

  if (!relayType) {
    navigate("/");
    return null;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (relayType === "TWS_FL" && !cdbFile) {
      setError("Please select a .cdb export file.");
      return;
    }
    if (relayType !== "TWS_FL" && !comtradeReady) {
      setError("Please select one .cff file or a matching .cfg + .dat pair.");
      return;
    }

    setError(null);
    setLoading(true);

    try {
      if (relayType === "TWS_FL") {
        const data = await uploadTwsCdb(cdbFile!);
        navigate(`/tws/${data.analysis_id}`);
        return;
      }

      if (!comtradeReady) {
        setError("Please select one .cff file or a matching .cfg + .dat pair.");
        return;
      }

      if (cffFile && (cfgFile || datFile)) {
        setError("Use either one .cff file or one .cfg + .dat pair, not both.");
        return;
      }
      if (cfgFile && datFile && fileStem(cfgFile) !== fileStem(datFile)) {
        setError("The .cfg and .dat filenames do not match. Select files from the same COMTRADE record.");
        return;
      }

      const data = await uploadComtrade(comtradeFiles);
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
          ? "Upload failed. Make sure the .cff is valid or the .cfg and .dat belong to the same COMTRADE record."
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
        <h1 className={styles.title}>{relayType === "TWS_FL" ? "Upload TWS FL Export" : "Upload COMTRADE Files"}</h1>
        <p className={styles.hint}>
          {relayType === "TWS_FL" ? (
            <>
              Select the Qualitrol Cashel TWS FL <code>.cdb</code> export from the fault locator viewer.
            </>
          ) : (
            <>
              Select one ABB <code>.cff</code> file or the matching <code>.cfg</code> and <code>.dat</code> files from your relay or DFR recorder.
            </>
          )}
        </p>

        <form onSubmit={handleSubmit} className={styles.form}>
          {relayType === "TWS_FL" ? (
            <div className={styles.singleDrop}>
              <DropZone
                label=".cdb"
                accept=".cdb,.CDB"
                file={cdbFile}
                inputRef={cdbRef}
                onChange={setCdbFile}
              />
            </div>
          ) : (
            <div className={styles.singleDrop}>
              <ComtradeDropZone
                files={comtradeFiles}
                inputRef={comtradeRef}
                onChange={(files) => {
                  const next = parseComtradeFiles(files);
                  setCfgFile(next.cfg);
                  setDatFile(next.dat);
                  setCffFile(next.cff);
                  setSuggestion(null);
                  setError(next.error);
                }}
              />
            </div>
          )}

          {error && <div className={styles.error}>{error}</div>}

          <button
            type="submit"
            className={styles.submit}
            disabled={loading || (relayType === "TWS_FL" ? !cdbFile : !comtradeReady)}
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

function fileExt(file: File) {
  return file.name.split(".").pop()?.toLowerCase() ?? "";
}

function fileStem(file: File) {
  return file.name.replace(/\.[^.]+$/, "").toLowerCase();
}

function parseComtradeFiles(files: File[]) {
  const supported = files.filter((file) => ["cff", "cfg", "dat"].includes(fileExt(file)));
  const unsupported = files.filter((file) => !["cff", "cfg", "dat"].includes(fileExt(file)));
  if (unsupported.length) {
    return {
      cfg: null,
      dat: null,
      cff: null,
      error: `Unsupported file type: ${unsupported.map((file) => file.name).join(", ")}`,
    };
  }

  const cffs = supported.filter((file) => fileExt(file) === "cff");
  const cfgs = supported.filter((file) => fileExt(file) === "cfg");
  const dats = supported.filter((file) => fileExt(file) === "dat");

  if (cffs.length > 1) {
    return { cfg: null, dat: null, cff: null, error: "Select only one .cff file." };
  }
  if (cffs.length === 1 && (cfgs.length > 0 || dats.length > 0)) {
    return { cfg: null, dat: null, cff: cffs[0], error: "Use either one .cff file or one .cfg + .dat pair, not both." };
  }
  if (cffs.length === 1) {
    return { cfg: null, dat: null, cff: cffs[0], error: null };
  }

  if (cfgs.length > 1 || dats.length > 1) {
    return { cfg: null, dat: null, cff: null, error: "Select exactly one .cfg file and one .dat file." };
  }

  const cfg = cfgs[0] ?? null;
  const dat = dats[0] ?? null;
  if (cfg && dat && fileStem(cfg) !== fileStem(dat)) {
    return { cfg, dat, cff: null, error: "The .cfg and .dat filenames do not match. Select files from the same COMTRADE record." };
  }

  return { cfg, dat, cff: null, error: null };
}

function ComtradeDropZone({
  files,
  inputRef,
  onChange,
}: {
  files: File[];
  inputRef: React.RefObject<HTMLInputElement | null>;
  onChange: (files: File[]) => void;
}) {
  const [over, setOver] = useState(false);

  function handleFiles(list: FileList | null) {
    if (!list) return;
    onChange(Array.from(list));
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setOver(false);
    handleFiles(e.dataTransfer.files);
  }

  return (
    <div
      className={`${styles.dropzone} ${styles.dropzoneWide} ${over ? styles.dropzoneOver : ""} ${files.length ? styles.dropzoneFilled : ""}`}
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
        accept=".cff,.CFF,.cfg,.CFG,.dat,.DAT"
        multiple
        style={{ display: "none" }}
        onChange={(e) => handleFiles(e.target.files)}
      />
      <span className={styles.dropLabel}>.cff / .cfg + .dat</span>
      {files.length ? (
        <span className={styles.fileName}>{files.map((file) => file.name).join(" + ")}</span>
      ) : (
        <span className={styles.dropHint}>Click or drag one .cff file, or select the .cfg and .dat pair together</span>
      )}
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
