import { useEffect, useState } from "react";

import { fetchFaultClassification21 } from "../../../api/client";
import styles from "../../panels/Panel.module.css";

interface Props {
  analysisId: string;
  dataRevision?: number;
}

type Classification = Awaited<ReturnType<typeof fetchFaultClassification21>>;

const FAULT_CODE_COLORS: Record<string, { bg: string; border: string; color: string }> = {
  SLG: { bg: "#fff7ed", border: "#fdba74", color: "#c2410c" },
  DLG: { bg: "#fef2f2", border: "#fca5a5", color: "#b91c1c" },
  LL: { bg: "#fffbeb", border: "#fcd34d", color: "#b45309" },
  "3Ph": { bg: "#fef2f2", border: "#f87171", color: "#991b1b" },
  SL: { bg: "#f0f9ff", border: "#7dd3fc", color: "#0369a1" },
  Unknown: { bg: "#f8fafc", border: "#cbd5e1", color: "#475569" },
};

const FAULT_CODE_LABEL: Record<string, string> = {
  SLG: "1 Fasa ke Tanah",
  DLG: "2 Fasa ke Tanah",
  LL: "2 Fasa (LL)",
  "3Ph": "3 Fasa",
  SL: "1 Fasa",
  Unknown: "Tidak Diketahui",
};

function Metric({ label, value, sub, highlight }: { label: string; value: string; sub?: string; highlight?: boolean }) {
  return (
    <div className={`${styles.faultMetric} ${highlight ? styles.faultMetricHot : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {sub && <small>{sub}</small>}
    </div>
  );
}

function formatMs(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(0) : "-";
}

export default function FaultTypeBadge21({ analysisId, dataRevision = 0 }: Props) {
  const [data, setData] = useState<Classification | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    setData(null);
    fetchFaultClassification21(analysisId)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [analysisId, dataRevision]);

  const colorSet = FAULT_CODE_COLORS[data?.fault_code ?? "Unknown"] ?? FAULT_CODE_COLORS.Unknown;
  const faultLabel = data ? FAULT_CODE_LABEL[data.fault_code] ?? data.fault_code : "";
  const faultSubtext = data?.phases_label && data.phases_label !== "-"
    ? `Fasa: ${data.phases_label}${data.to_ground ? " - ke tanah" : ""}`
    : "";

  return (
    <div className={styles.panel} style={{ padding: "16px 20px" }}>
      <div className={styles.faultSectionLabel}>Jenis Gangguan</div>

      {loading && <div style={{ fontSize: "0.8rem", color: "#94a3b8" }}>Mengklasifikasikan...</div>}

      {!loading && data && (
        <div className={styles.faultSummaryCard}>
          <div className={styles.faultIdentityRow}>
            <span
              className={styles.faultCodePill}
              style={{
                background: colorSet.bg,
                borderColor: colorSet.border,
                color: colorSet.color,
              }}
            >
              {data.fault_code}
            </span>
            <div className={styles.faultIdentityText}>
              <div>{faultLabel}</div>
              {faultSubtext && <p>{faultSubtext}</p>}
              {(data.trip_type || data.zone) && (
                <p>
                  {data.trip_type && (
                    <span>Trip: <strong>{data.trip_type.replace("_", " ")}</strong></span>
                  )}
                  {data.zone && (
                    <span>{data.trip_type ? " / " : ""}Zona: <strong>{data.zone}</strong></span>
                  )}
                </p>
              )}
            </div>
          </div>

          <div className={styles.faultMetricRow}>
            <Metric label="Pre-Fault" value={formatMs(data.prefault_ms)} sub="ms" />
            <Metric
              label="Gangguan"
              value={formatMs(data.fault_ms)}
              sub={data.ar_status === "failed" ? "ms / Permanen" : data.ar_status === "successful" ? "ms / Transien" : "ms"}
              highlight
            />
            <Metric label="Total Rekaman" value={formatMs(data.total_ms)} sub="ms" />
          </div>
        </div>
      )}

      {!loading && !data && (
        <p className={styles.emptyText}>Klasifikasi tidak tersedia.</p>
      )}
    </div>
  );
}
