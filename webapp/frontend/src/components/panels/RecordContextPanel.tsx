import { useEffect, useState } from "react";
import { fetchCanonicalAnalysis, type CanonicalRecordAnalysis } from "../../api/client";
import styles from "./Panel.module.css";

export default function RecordContextPanel({ analysisId, dataRevision = 0 }: { analysisId: string; dataRevision?: number }) {
  const [data, setData] = useState<CanonicalRecordAnalysis | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let active = true;
    setData(null);
    setFailed(false);
    fetchCanonicalAnalysis(analysisId, dataRevision).then((value) => { if (active) setData(value); })
      .catch(() => { if (active) setFailed(true); });
    return () => { active = false; };
  }, [analysisId, dataRevision]);
  const sequence = data?.event_window?.sequence;
  const episodes = data?.fault_episodes ?? [];
  return <section className={styles.panel}>
    <h2>Konteks &amp; urutan kejadian</h2>
    {!data && <p>{failed ? "Konteks belum tersedia." : "Membaca urutan kejadian..."}</p>}
    {data && <>
      <p>{episodes.length} episode gangguan dalam satu rekaman.</p>
      {sequence?.mechanical_close_confirmed && <p>
        PMT menutup kembali. {sequence.restoration_outcome === "failed"
          ? "Pemulihan gagal; penutupan PMT tidak berarti gangguan sudah hilang."
          : sequence.restoration_outcome === "successful" ? "Pemulihan bertahan selama rekaman." : "Hasil pemulihan belum dapat ditentukan."}
      </p>}
      {sequence?.refault_after_reclose && <p>Gangguan muncul kembali setelah reclose.</p>}
      {sequence?.sotf_after_reclose && <p>SOTF/TOR trip setelah penutupan kembali.</p>}
      {episodes.length > 0 && <table style={{ width: "100%", textAlign: "left" }}>
        <thead><tr><th>Episode</th><th>Mulai (ms)</th><th>Padam (ms)</th><th>Durasi (ms)</th></tr></thead>
        <tbody>{episodes.map((episode, index) => <tr key={index}>
          <td>{index + 1}{episode.after_reclose ? " — setelah reclose" : ""}</td>
          {[episode.inception_time_ms, episode.clearing_time_ms, episode.fault_duration_ms].map((value, i) =>
            <td key={i}>{typeof value === "number" ? value.toFixed(1) : "Belum diketahui"}</td>)}
        </tr>)}</tbody>
      </table>}
      {episodes.length > 1 && <p>Waktu episode berasal dari waveform dan dapat bergeser sekitar satu siklus terhadap kontak digital. Dead time tidak dihitung sebagai durasi gangguan.</p>}
      <details>
        <summary>Alasan dan bukti sinyal</summary>
        {data.reasoning?.conclusions.map((row) => <div key={row.key}>
          <h3>{row.label}: {row.title}</h3>
          <ul>{row.evidence.map((evidence, i) => <li key={i}>{evidence}</li>)}</ul>
          {row.conflicts.length > 0 && <p>Perlu dicek: {row.conflicts.join(" ")}</p>}
        </div>)}
      </details>
    </>}
  </section>;
}
