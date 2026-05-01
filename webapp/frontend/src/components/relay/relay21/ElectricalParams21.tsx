import { useEffect, useState, type ReactNode } from "react";

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
    <div className={`${styles.paramTile} ${highlight ? styles.paramTileHot : ""}`}>
      <span>{label}</span>
      <strong>
        {value.toFixed(value < 10 ? 2 : 1)}
        <small>{unit}</small>
      </strong>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className={styles.paramSection}>
      <div className={styles.paramSectionTitle}>{title}</div>
      <div className={styles.paramGrid}>{children}</div>
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
          <Section title="Durasi & Arus Puncak">
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
              <Param label="I1 urutan positif" value={params.i_pos_seq_a} unit="A" />
              <Param label="I2 urutan negatif" value={params.i_neg_seq_a} unit="A" highlight={(params.i_neg_seq_a ?? 0) > (params.i_pos_seq_a ?? 0) * 0.3} />
              <Param label="I0 urutan nol" value={params.i_zero_seq_a} unit="A" highlight={(params.i_zero_seq_a ?? 0) > (params.i_pos_seq_a ?? 0) * 0.2} />
            </Section>
          )}

          {hasImpedance && (
            <Section title="Impedansi Gangguan">
              <Param label="|Z| saat Inception" value={params.z_at_inception_ohm} unit="ohm" highlight />
              <Param label="R (resistif)" value={params.r_at_fault_ohm} unit="ohm" />
              <Param label="X (reaktif)" value={params.x_at_fault_ohm} unit="ohm" />
              <Param label="Rasio R/X" value={params.rx_ratio} unit="" />
              <Param label="Sudut Z" value={params.z_angle_deg} unit="deg" />
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
