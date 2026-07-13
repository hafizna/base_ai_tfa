import type { ReconstructionOut } from "../../api/client";
import styles from "./ReconstructionSummary.module.css";

interface Props {
  reconstruction: ReconstructionOut;
}

const ALIGNMENT_LABELS: Record<string, string> = {
  ALIGNED: "ALIGNED",
  LIKELY_ALIGNED: "LIKELY ALIGNED",
  ORDER_ONLY: "ORDER ONLY",
  MANUAL_ORDER: "MANUAL ORDER",
  UNTRUSTED: "CLOCK UNTRUSTED",
  INSUFFICIENT_DATA: "INSUFFICIENT DATA",
};

const SAME_BAY_LABELS: Record<string, string> = {
  CONFIRMED_SAME_BAY: "CONFIRMED SAME BAY",
  LIKELY_SAME_BAY: "LIKELY SAME BAY",
  MISMATCH_REQUIRES_REVIEW: "REQUIRES REVIEW",
  UNKNOWN: "UNKNOWN",
};

function toneFor(key: string): string {
  if (["ALIGNED", "CONFIRMED_SAME_BAY", "CONSISTENT"].includes(key)) return "good";
  if (["LIKELY_ALIGNED", "LIKELY_SAME_BAY", "MOSTLY_CONSISTENT", "ORDER_ONLY", "MANUAL_ORDER"].includes(key)) return "neutral";
  if (["UNTRUSTED", "MISMATCH_REQUIRES_REVIEW", "MIXED", "CONTRADICTORY", "INSUFFICIENT_DATA", "INSUFFICIENT"].includes(key)) return "warn";
  return "neutral";
}

function formatDuration(ms: number | null): string {
  if (ms == null) return "-";
  const totalSeconds = Math.round(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

export default function ReconstructionSummary({ reconstruction }: Props) {
  const warningCount =
    reconstruction.alignment.warnings.length +
    (reconstruction.same_bay_status === "MISMATCH_REQUIRES_REVIEW" ? 1 : 0);

  const tiles = [
    { label: "Records", value: reconstruction.observed_incident_facts.record_count },
    { label: "Episodes", value: reconstruction.observed_incident_facts.episode_count },
    {
      label: "Alignment",
      value: ALIGNMENT_LABELS[reconstruction.alignment.status] ?? reconstruction.alignment.status,
      tone: toneFor(reconstruction.alignment.status),
    },
    { label: "Alignment confidence", value: `${Math.round(reconstruction.alignment.confidence * 100)}%` },
    { label: "Order source", value: reconstruction.alignment.order_source },
    {
      label: "Same-bay",
      value: SAME_BAY_LABELS[reconstruction.same_bay_status] ?? reconstruction.same_bay_status,
      tone: toneFor(reconstruction.same_bay_status),
    },
    { label: "Incident duration", value: formatDuration(reconstruction.observed_incident_facts.incident_duration_ms) },
    {
      label: "Cause consistency",
      value: reconstruction.physical_cause_evidence.consistency,
      tone: toneFor(reconstruction.physical_cause_evidence.consistency),
    },
    { label: "Reconstruction version", value: reconstruction.reconstruction_id.slice(0, 8) },
    { label: "Warnings", value: warningCount, tone: warningCount > 0 ? "warn" : "good" },
  ];

  return (
    <div className={styles.grid}>
      {tiles.map((tile) => (
        <div key={tile.label} className={`${styles.tile} ${tile.tone ? styles[`tone_${tile.tone}`] : ""}`}>
          <span className={styles.tileLabel}>{tile.label}</span>
          <span className={styles.tileValue}>{tile.value}</span>
        </div>
      ))}
    </div>
  );
}
