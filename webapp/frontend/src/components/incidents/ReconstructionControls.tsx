import { useState } from "react";
import type { ReconstructionOut } from "../../api/client";
import styles from "./ReconstructionControls.module.css";

interface Props {
  reconstruction: ReconstructionOut | null;
  reconstructions: ReconstructionOut[];
  stale: boolean;
  staleReason: string | null;
  loading: boolean;
  recordCount: number;
  onReconstruct: () => Promise<void>;
  onSelectVersion: (reconstructionId: string) => void;
  selectedVersionId: string | null;
}

function formatTime(iso: string | null | undefined) {
  if (!iso) return "-";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function ReconstructionControls({
  reconstruction,
  reconstructions,
  stale,
  staleReason,
  loading,
  recordCount,
  onReconstruct,
  onSelectVersion,
  selectedVersionId,
}: Props) {
  const [triggering, setTriggering] = useState(false);

  async function handleClick() {
    setTriggering(true);
    try {
      await onReconstruct();
    } finally {
      setTriggering(false);
    }
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.row}>
        <button
          type="button"
          className={styles.reconstructButton}
          onClick={handleClick}
          disabled={triggering || loading || recordCount === 0}
          title={recordCount === 0 ? "Attach at least one record before reconstructing" : undefined}
        >
          {triggering ? "Reconstructing…" : reconstruction ? "Re-reconstruct incident" : "Reconstruct incident"}
        </button>

        {reconstructions.length > 1 && (
          <label className={styles.historySelect}>
            Version
            <select
              value={selectedVersionId ?? reconstruction?.reconstruction_id ?? ""}
              onChange={(e) => onSelectVersion(e.target.value)}
            >
              {reconstructions
                .slice()
                .reverse()
                .map((r) => (
                  <option key={r.reconstruction_id} value={r.reconstruction_id}>
                    {formatTime(r.created_at)} {r.is_latest ? "(latest)" : ""}
                  </option>
                ))}
            </select>
          </label>
        )}
      </div>

      {reconstruction && (
        <div className={styles.meta}>
          <span>Engine {reconstruction.engine_version}</span>
          <span>· Schema {reconstruction.schema_version}</span>
          <span>· Created {formatTime(reconstruction.created_at)}</span>
          {!reconstruction.is_latest && <span className={styles.oldBadge}>Viewing historical version</span>}
        </div>
      )}

      {stale && (
        <div className={styles.staleWarning}>
          Reconstruction may be stale{staleReason ? `: ${staleReason}` : ""}. Re-reconstruct to refresh.
        </div>
      )}

      {!reconstruction && !loading && recordCount === 0 && (
        <div className={styles.hint}>Attach at least one record before reconstructing.</div>
      )}
      {!reconstruction && !loading && recordCount === 1 && (
        <div className={styles.hint}>Only one record is attached. Reconstruction will produce a single episode with no relationships to compare.</div>
      )}
      {!reconstruction && !loading && recordCount > 1 && (
        <div className={styles.hint}>This incident has not been reconstructed yet.</div>
      )}
    </div>
  );
}
