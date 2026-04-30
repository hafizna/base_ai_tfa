import { useEffect, useState } from "react";

import { fetchFaultClassification21 } from "../../../api/client";
import styles from "../../panels/Panel.module.css";

interface Props {
  analysisId: string;
  dataRevision?: number;
}

type Classification = Awaited<ReturnType<typeof fetchFaultClassification21>>;

const FAULT_CODE_COLORS: Record<string, { bg: string; border: string; color: string }> = {
  SLG:     { bg: "#fff7ed", border: "#fdba74", color: "#c2410c" },
  DLG:     { bg: "#fef2f2", border: "#fca5a5", color: "#b91c1c" },
  LL:      { bg: "#fffbeb", border: "#fcd34d", color: "#b45309" },
  "3Ph":   { bg: "#fef2f2", border: "#f87171", color: "#991b1b" },
  SL:      { bg: "#f0f9ff", border: "#7dd3fc", color: "#0369a1" },
  Unknown: { bg: "#f8fafc", border: "#cbd5e1", color: "#475569" },
};

const FAULT_CODE_LABEL: Record<string, string> = {
  SLG:     "1 Fasa ke Tanah",
  DLG:     "2 Fasa ke Tanah",
  LL:      "2 Fasa (LL)",
  "3Ph":   "3 Fasa",
  SL:      "1 Fasa",
  Unknown: "Tidak Diketahui",
};

function Metric({ label, value, sub, highlight }: { label: string; value: string; sub?: string; highlight?: boolean }) {
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      gap: 2,
      padding: "8px 12px",
      background: highlight ? "#fef2f2" : "#f8fafc",
      border: `1px solid ${highlight ? "#fca5a5" : "#e2e8f0"}`,
      borderRadius: 8,
      flex: 1,
      minWidth: 60,
    }}>
      <span style={{ fontSize: "0.65rem", color: "#94a3b8", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</span>
      <span style={{ fontSize: "1.1rem", fontWeight: 700, color: highlight ? "#dc2626" : "#1e293b", lineHeight: 1 }}>{value}</span>
      {sub && <span style={{ fontSize: "0.65rem", color: "#64748b" }}>{sub}</span>}
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

  return (
    <div className={styles.panel} style={{ padding: "16px 20px" }}>
      <div style={{ fontSize: "0.65rem", fontWeight: 700, color: "#94a3b8", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 10 }}>
        Jenis Gangguan
      </div>

      {loading && <div style={{ fontSize: "0.8rem", color: "#94a3b8" }}>Mengklasifikasikan...</div>}

      {!loading && data && (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12, flexWrap: "wrap" }}>
            <span style={{
              padding: "6px 18px",
              background: colorSet.bg,
              border: `2px solid ${colorSet.border}`,
              borderRadius: 8,
              fontSize: "1.3rem",
              fontWeight: 800,
              color: colorSet.color,
              letterSpacing: "0.04em",
            }}>
              {data.fault_code}
            </span>
            <div>
              <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "#1e293b" }}>
                {FAULT_CODE_LABEL[data.fault_code] ?? data.fault_code}
              </div>
              {data.phases_label && data.phases_label !== "-" && (
                <div style={{ fontSize: "0.78rem", color: "#475569" }}>
                  Fasa: <strong>{data.phases_label}</strong>
                  {data.to_ground ? " — ke tanah" : ""}
                </div>
              )}
              <div style={{ fontSize: "0.75rem", color: "#64748b", marginTop: 2 }}>
                {data.trip_type && (
                  <span>Trip: <strong>{data.trip_type.replace("_", " ")}</strong></span>
                )}
                {data.zone && (
                  <span style={{ marginLeft: data.trip_type ? 10 : 0 }}>
                    • Zona: <strong>{data.zone}</strong>
                  </span>
                )}
              </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 8 }}>
            <Metric label="Pre-Fault" value={formatMs(data.prefault_ms)} sub="ms" />
            <Metric
              label="Gangguan"
              value={formatMs(data.fault_ms)}
              sub={data.ar_status === "failed" ? "ms · Permanen" : data.ar_status === "successful" ? "ms · Transien" : "ms"}
              highlight
            />
            <Metric label="Total Rekaman" value={formatMs(data.total_ms)} sub="ms" />
          </div>
        </>
      )}

      {!loading && !data && (
        <p className={styles.emptyText}>Klasifikasi tidak tersedia.</p>
      )}
    </div>
  );
}
