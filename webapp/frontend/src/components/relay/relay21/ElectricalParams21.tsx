import { useEffect, useState } from "react";

import { fetchElectricalParams21 } from "../../../api/client";
import styles from "../../panels/Panel.module.css";

interface Props {
  analysisId: string;
  dataRevision?: number;
}

type Params = Awaited<ReturnType<typeof fetchElectricalParams21>>;

function Param({ label, value, unit, highlight }: { label: string; value?: number | null; unit: string; highlight?: boolean }) {
  if (value === undefined || value === null) return null;
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      gap: 2,
      padding: "10px 14px",
      background: highlight ? "#fef2f2" : "#f8fafc",
      border: `1.5px solid ${highlight ? "#fca5a5" : "#e2e8f0"}`,
      borderRadius: 10,
    }}>
      <span style={{ fontSize: "0.7rem", color: "#64748b", fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase" }}>{label}</span>
      <span style={{ fontSize: "1.05rem", fontWeight: 700, color: highlight ? "#dc2626" : "#1e293b", fontVariantNumeric: "tabular-nums" }}>
        {value.toFixed(value < 10 ? 2 : 1)}
        <span style={{ fontSize: "0.75rem", fontWeight: 400, color: "#64748b", marginLeft: 4 }}>{unit}</span>
      </span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: "0.7rem", fontWeight: 700, color: "#64748b", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
        {title}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 8 }}>
        {children}
      </div>
    </div>
  );
}

export default function ElectricalParams21({ analysisId, dataRevision = 0 }: Props) {
  const [params, setParams] = useState<Params | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    setParams(null);
    fetchElectricalParams21(analysisId)
      .then(setParams)
      .catch(() => setParams(null))
      .finally(() => setLoading(false));
  }, [analysisId, dataRevision]);

  const hasAnyPhase = params && (params.i_peak_ia_a !== undefined || params.i_peak_ib_a !== undefined || params.i_peak_ic_a !== undefined);
  const hasSeq = params && (params.i_pos_seq_a !== undefined || params.i_neg_seq_a !== undefined || params.i_zero_seq_a !== undefined);
  const hasImpedance = params && (params.z_at_inception_ohm !== undefined || params.r_at_fault_ohm !== undefined || params.z_angle_deg !== undefined);

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Parameter Elektrikal Gangguan</h2>
        {loading && <span className={styles.badge}>Menghitung...</span>}
      </div>

      {!loading && !params && (
        <p className={styles.emptyText}>Data tidak tersedia untuk rekaman ini.</p>
      )}

      {params && (
        <>
          <Section title="Durasi &amp; Arus Puncak">
            <Param label="Durasi Gangguan" value={params.fault_duration_ms} unit="ms" highlight />
            <Param label="I puncak fasa A" value={params.i_peak_ia_a} unit="A" />
            <Param label="I puncak fasa B" value={params.i_peak_ib_a} unit="A" />
            <Param label="I puncak fasa C" value={params.i_peak_ic_a} unit="A" />
            {params.v_sag_pct !== undefined && (
              <Param label="Tegangan Sag" value={params.v_sag_pct} unit="%" highlight={params.v_sag_pct > 30} />
            )}
          </Section>

          {hasSeq && (
            <Section title="Komponen Simetris (Fortescue)">
              <Param label="I₁ Urutan Positif" value={params.i_pos_seq_a} unit="A" />
              <Param label="I₂ Urutan Negatif" value={params.i_neg_seq_a} unit="A" highlight={(params.i_neg_seq_a ?? 0) > (params.i_pos_seq_a ?? 0) * 0.3} />
              <Param label="I₀ Urutan Nol" value={params.i_zero_seq_a} unit="A" highlight={(params.i_zero_seq_a ?? 0) > (params.i_pos_seq_a ?? 0) * 0.2} />
            </Section>
          )}

          {hasImpedance && (
            <Section title="Impedansi Gangguan">
              <Param label="|Z| saat Inception" value={params.z_at_inception_ohm} unit="Ω" highlight />
              <Param label="R (resistif)" value={params.r_at_fault_ohm} unit="Ω" />
              <Param label="X (reaktif)" value={params.x_at_fault_ohm} unit="Ω" />
              <Param label="Rasio R/X" value={params.rx_ratio} unit="" />
              <Param label="Sudut Z" value={params.z_angle_deg} unit="°" />
            </Section>
          )}

          {params.ar_dead_time_ms !== undefined && (
            <Section title="Auto-Reclose">
              <Param label="Dead Time AR" value={params.ar_dead_time_ms} unit="ms" />
            </Section>
          )}

          {!hasAnyPhase && !hasSeq && !hasImpedance && (
            <p className={styles.emptyText}>Tidak ada parameter tambahan yang dapat dihitung dari rekaman ini.</p>
          )}
        </>
      )}
    </div>
  );
}
