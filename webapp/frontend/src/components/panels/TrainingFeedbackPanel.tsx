import { useEffect, useMemo, useState } from "react";
import {
  clearTrainingArchive,
  downloadTrainingArchive,
  fetchTrainingStatus,
  submitTrainingFeedback,
  type GroundTruthConfidence,
  type GroundTruthSource,
  type TrainingStatus,
} from "../../api/client";
import styles from "./TrainingFeedbackPanel.module.css";

const GROUND_TRUTH_SOURCE_OPTIONS: GroundTruthSource[] = [
  "RELAY_EVENT_REPORT",
  "OPERATOR_SOE",
  "REMOTE_END_COMTRADE",
  "DFR_RECORD",
  "FIELD_INSPECTION",
  "LIGHTNING_DETECTION",
  "PATROL_REPORT",
  "PROTECTION_ENGINEER_REVIEW",
  "UNCONFIRMED_ASSUMPTION",
  "OTHER",
];

const GROUND_TRUTH_CONFIDENCE_OPTIONS: GroundTruthConfidence[] = [
  "CONFIRMED",
  "PROBABLE",
  "POSSIBLE",
  "UNKNOWN",
];

/** Tri-state Correct/Wrong/Unset control for one correction layer. */
function TriState({
  value,
  onChange,
}: {
  value: boolean | null;
  onChange: (next: boolean | null) => void;
}) {
  return (
    <div className={styles.miniSegmented}>
      <button type="button" className={value === true ? styles.active : ""} onClick={() => onChange(true)}>
        OK
      </button>
      <button type="button" className={value === false ? styles.active : ""} onClick={() => onChange(false)}>
        Wrong
      </button>
      <button type="button" className={value === null ? styles.active : ""} onClick={() => onChange(null)}>
        —
      </button>
    </div>
  );
}

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

  // Stage 0: per-layer ground-truth correction fields.
  const [showCorrections, setShowCorrections] = useState(false);
  const [parsingCorrect, setParsingCorrect] = useState<boolean | null>(null);
  const [channelMappingCorrect, setChannelMappingCorrect] = useState<boolean | null>(null);
  const [inceptionCorrect, setInceptionCorrect] = useState<boolean | null>(null);
  const [correctedInceptionMs, setCorrectedInceptionMs] = useState("");
  const [clearingCorrect, setClearingCorrect] = useState<boolean | null>(null);
  const [correctedClearingMs, setCorrectedClearingMs] = useState("");
  const [faultedPhasesCorrect, setFaultedPhasesCorrect] = useState<boolean | null>(null);
  const [actualFaultedPhases, setActualFaultedPhases] = useState("");
  const [faultTypeCorrect, setFaultTypeCorrect] = useState<boolean | null>(null);
  const [actualFaultType, setActualFaultType] = useState("");
  const [zoneCorrect, setZoneCorrect] = useState<boolean | null>(null);
  const [actualZone, setActualZone] = useState("");
  const [tripTypeCorrect, setTripTypeCorrect] = useState<boolean | null>(null);
  const [actualTripType, setActualTripType] = useState("");
  const [recloseCorrect, setRecloseCorrect] = useState<boolean | null>(null);
  const [actualRecloseOutcome, setActualRecloseOutcome] = useState("");
  const [segmentationCorrect, setSegmentationCorrect] = useState<boolean | null>(null);
  const [actualEpisodeCount, setActualEpisodeCount] = useState("");
  const [protectionInterpretationCorrect, setProtectionInterpretationCorrect] = useState<boolean | null>(null);
  const [actualEventClass, setActualEventClass] = useState("");
  const [causeCorrect, setCauseCorrect] = useState<boolean | null>(null);
  const [actualCause, setActualCause] = useState("");
  const [groundTruthSource, setGroundTruthSource] = useState<GroundTruthSource[]>([]);
  const [groundTruthConfidence, setGroundTruthConfidence] = useState<GroundTruthConfidence>("UNKNOWN");

  function toggleGroundTruthSource(src: GroundTruthSource) {
    setGroundTruthSource((prev) => (prev.includes(src) ? prev.filter((s) => s !== src) : [...prev, src]));
  }

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
        parsing_correct: parsingCorrect,
        channel_mapping_correct: channelMappingCorrect,
        inception_correct: inceptionCorrect,
        corrected_inception_time_ms: correctedInceptionMs.trim() ? Number(correctedInceptionMs) : null,
        clearing_correct: clearingCorrect,
        corrected_clearing_time_ms: correctedClearingMs.trim() ? Number(correctedClearingMs) : null,
        faulted_phases_correct: faultedPhasesCorrect,
        actual_faulted_phases: actualFaultedPhases
          .split(/[,+\s]+/)
          .map((p) => p.trim().toUpperCase())
          .filter(Boolean),
        fault_type_correct: faultTypeCorrect,
        actual_fault_type: actualFaultType,
        zone_correct: zoneCorrect,
        actual_zone: actualZone,
        trip_type_correct: tripTypeCorrect,
        actual_trip_type: actualTripType,
        reclose_correct: recloseCorrect,
        actual_reclose_outcome: actualRecloseOutcome,
        event_segmentation_correct: segmentationCorrect,
        actual_episode_count: actualEpisodeCount.trim() ? Number(actualEpisodeCount) : null,
        protection_interpretation_correct: protectionInterpretationCorrect,
        actual_event_class: actualEventClass,
        cause_correct: causeCorrect,
        actual_cause: actualCause,
        ground_truth_source: groundTruthSource,
        ground_truth_confidence: groundTruthConfidence,
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

      <button
        type="button"
        className={styles.sectionToggle}
        onClick={() => setShowCorrections((v) => !v)}
      >
        <span>Per-layer ground-truth corrections (optional)</span>
        <span>{showCorrections ? "Hide" : "Show"}</span>
      </button>

      {showCorrections && (
        <div>
          <div className={styles.correctionRow}>
            <span className={styles.correctionLabel}>Parsing correct</span>
            <TriState value={parsingCorrect} onChange={setParsingCorrect} />
            <span />
          </div>
          <div className={styles.correctionRow}>
            <span className={styles.correctionLabel}>Channel mapping correct</span>
            <TriState value={channelMappingCorrect} onChange={setChannelMappingCorrect} />
            <span />
          </div>
          <div className={styles.correctionRow}>
            <span className={styles.correctionLabel}>Inception correct</span>
            <TriState value={inceptionCorrect} onChange={setInceptionCorrect} />
            <input
              className={styles.miniInput}
              placeholder="corrected inception (ms)"
              value={correctedInceptionMs}
              onChange={(e) => setCorrectedInceptionMs(e.target.value)}
              disabled={inceptionCorrect !== false}
            />
          </div>
          <div className={styles.correctionRow}>
            <span className={styles.correctionLabel}>Clearing correct</span>
            <TriState value={clearingCorrect} onChange={setClearingCorrect} />
            <input
              className={styles.miniInput}
              placeholder="corrected clearing (ms)"
              value={correctedClearingMs}
              onChange={(e) => setCorrectedClearingMs(e.target.value)}
              disabled={clearingCorrect !== false}
            />
          </div>
          <div className={styles.correctionRow}>
            <span className={styles.correctionLabel}>Faulted phases correct</span>
            <TriState value={faultedPhasesCorrect} onChange={setFaultedPhasesCorrect} />
            <input
              className={styles.miniInput}
              placeholder="e.g. A+B"
              value={actualFaultedPhases}
              onChange={(e) => setActualFaultedPhases(e.target.value)}
              disabled={faultedPhasesCorrect !== false}
            />
          </div>
          <div className={styles.correctionRow}>
            <span className={styles.correctionLabel}>Fault type correct</span>
            <TriState value={faultTypeCorrect} onChange={setFaultTypeCorrect} />
            <input
              className={styles.miniInput}
              placeholder="e.g. DLG"
              value={actualFaultType}
              onChange={(e) => setActualFaultType(e.target.value)}
              disabled={faultTypeCorrect !== false}
            />
          </div>
          <div className={styles.correctionRow}>
            <span className={styles.correctionLabel}>Zone correct</span>
            <TriState value={zoneCorrect} onChange={setZoneCorrect} />
            <input
              className={styles.miniInput}
              placeholder="e.g. Z1"
              value={actualZone}
              onChange={(e) => setActualZone(e.target.value)}
              disabled={zoneCorrect !== false}
            />
          </div>
          <div className={styles.correctionRow}>
            <span className={styles.correctionLabel}>Trip type correct</span>
            <TriState value={tripTypeCorrect} onChange={setTripTypeCorrect} />
            <input
              className={styles.miniInput}
              placeholder="single_pole / three_pole"
              value={actualTripType}
              onChange={(e) => setActualTripType(e.target.value)}
              disabled={tripTypeCorrect !== false}
            />
          </div>
          <div className={styles.correctionRow}>
            <span className={styles.correctionLabel}>Reclose outcome correct</span>
            <TriState value={recloseCorrect} onChange={setRecloseCorrect} />
            <input
              className={styles.miniInput}
              placeholder="successful / failed"
              value={actualRecloseOutcome}
              onChange={(e) => setActualRecloseOutcome(e.target.value)}
              disabled={recloseCorrect !== false}
            />
          </div>
          <div className={styles.correctionRow}>
            <span className={styles.correctionLabel}>Event segmentation correct</span>
            <TriState value={segmentationCorrect} onChange={setSegmentationCorrect} />
            <input
              className={styles.miniInput}
              placeholder="actual episode count"
              value={actualEpisodeCount}
              onChange={(e) => setActualEpisodeCount(e.target.value)}
              disabled={segmentationCorrect !== false}
            />
          </div>
          <div className={styles.correctionRow}>
            <span className={styles.correctionLabel}>Protection interpretation correct</span>
            <TriState value={protectionInterpretationCorrect} onChange={setProtectionInterpretationCorrect} />
            <input
              className={styles.miniInput}
              placeholder="e.g. TRANSIENT_LINE_FAULT"
              value={actualEventClass}
              onChange={(e) => setActualEventClass(e.target.value)}
              disabled={protectionInterpretationCorrect !== false}
            />
          </div>
          <div className={styles.correctionRow}>
            <span className={styles.correctionLabel}>Cause correct</span>
            <TriState value={causeCorrect} onChange={setCauseCorrect} />
            <input
              className={styles.miniInput}
              placeholder="e.g. PETIR"
              value={actualCause}
              onChange={(e) => setActualCause(e.target.value)}
              disabled={causeCorrect !== false}
            />
          </div>

          <div className={styles.field} style={{ marginTop: 10 }}>
            <span className={styles.label}>Ground truth source</span>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {GROUND_TRUTH_SOURCE_OPTIONS.map((src) => (
                <button
                  key={src}
                  type="button"
                  className={styles.miniSegmented}
                  onClick={() => toggleGroundTruthSource(src)}
                  style={{
                    border: "1.5px solid",
                    borderColor: groundTruthSource.includes(src) ? "#2563eb" : "#cbd5e1",
                    background: groundTruthSource.includes(src) ? "#eff6ff" : "#fff",
                    color: groundTruthSource.includes(src) ? "#1d4ed8" : "#475569",
                    borderRadius: 6,
                    padding: "4px 8px",
                    fontSize: "0.72rem",
                    fontWeight: 700,
                    cursor: "pointer",
                  }}
                >
                  {src}
                </button>
              ))}
            </div>
          </div>

          <label className={styles.field} style={{ marginTop: 10, maxWidth: 260 }}>
            <span className={styles.label}>Ground truth confidence</span>
            <select
              className={styles.select}
              value={groundTruthConfidence}
              onChange={(e) => setGroundTruthConfidence(e.target.value as GroundTruthConfidence)}
            >
              {GROUND_TRUTH_CONFIDENCE_OPTIONS.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}

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
