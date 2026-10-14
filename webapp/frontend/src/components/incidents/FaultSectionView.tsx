import { useEffect, useState } from "react";
import { fetchFaultSection, runFaultSection, type FaultSectionEpisode, type FaultSectionOut } from "../../api/client";
import styles from "./FaultSectionView.module.css";

interface Props {
  incidentId: string;
  /** analysis_id of every included record, to tell when the stored result no longer matches them. */
  includedAnalysisIds: string[];
}

const TEMPLATE_LABEL: Record<FaultSectionOut["template"], string> = {
  diameter: "Diameter 1½ CB",
  transformer: "Bay trafo",
};

function signature(ids: string[]): string {
  return [...ids].sort().join(",");
}

function formatComputedAt(iso: string): string {
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return iso;
  return when.toLocaleString("id-ID", { dateStyle: "medium", timeStyle: "short" });
}

function durationMs(episode: FaultSectionEpisode): number {
  return Math.round((episode.end_s - episode.start_s) * 1000);
}

function errorText(err: unknown, fallback: string): string {
  const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
  return detail || (err instanceof Error ? err.message : fallback);
}

/**
 * "Lokasi gangguan": which part of the bay was faulted, tested against the
 * current through every CT and every relay decision in the records. Each
 * location the evidence allows is listed with what supports it; the rest are
 * listed with the observation that rules them out.
 */
export default function FaultSectionView({ incidentId, includedAnalysisIds }: Props) {
  const [result, setResult] = useState<FaultSectionOut | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The page mounts one view per incident (keyed by its id), so the state starts empty for each.
  useEffect(() => {
    let active = true;
    fetchFaultSection(incidentId)
      .then((data) => {
        if (active) setResult(data);
      })
      .catch(() => {
        // A missing or unreadable stored result just leaves the button to compute one.
      })
      .finally(() => {
        if (active) setLoaded(true);
      });
    return () => {
      active = false;
    };
  }, [incidentId]);

  async function handleRun() {
    setRunning(true);
    setError(null);
    try {
      setResult(await runFaultSection(incidentId));
    } catch (err) {
      setError(errorText(err, "Lokasi gangguan gagal dihitung."));
    } finally {
      setRunning(false);
    }
  }

  const stale =
    result !== null &&
    (result.analysis_ids
      ? signature(result.analysis_ids) !== signature(includedAnalysisIds)
      : result.record_count !== includedAnalysisIds.length);

  return (
    <section className={styles.card} aria-labelledby="fault-section-title">
      <div className={styles.header}>
        <div className={styles.headerText}>
          <h2 id="fault-section-title" className={styles.sectionTitle}>
            Lokasi gangguan
          </h2>
          <p className={styles.meta}>
            Bagian bay yang terganggu, diuji terhadap arus di tiap CT dan keputusan tiap rele dari semua rekaman.
          </p>
        </div>
        <button
          type="button"
          className={result ? styles.secondaryButton : styles.primaryButton}
          disabled={running || includedAnalysisIds.length === 0}
          onClick={() => void handleRun()}
        >
          {running ? "Menghitung…" : result ? "Hitung ulang" : "Hitung lokasi gangguan"}
        </button>
      </div>

      {running && <p className={styles.meta}>Rekaman dibaca ulang dari file aslinya; biasanya butuh beberapa detik.</p>}
      {error && <div className={styles.error}>{error}</div>}
      {stale && !running && (
        <div className={styles.warning}>
          Rekaman insiden berubah sejak lokasi ini dihitung. Hitung ulang supaya semua rekaman ikut diuji.
        </div>
      )}

      {loaded && !result && !running && !error && (
        <p className={styles.meta}>
          Belum dihitung. Paling berguna bila rekaman beberapa rele di satu bay sudah lengkap (misalnya CCP, diff, CBF
          di satu diameter, atau diff/REF/OCR di satu trafo).
        </p>
      )}

      {result && <ResultBody result={result} />}
    </section>
  );
}

function ResultBody({ result }: { result: FaultSectionOut }) {
  const multiple = result.episodes.length > 1;
  return (
    <>
      <div className={styles.labelRow}>
        <span className={`${styles.chip} ${styles.toneNeutral}`}>{TEMPLATE_LABEL[result.template] ?? result.template}</span>
        <span className={styles.meta}>
          Dihitung {formatComputedAt(result.computed_at)} dari {result.record_count} rekaman
        </span>
      </div>

      {result.warnings.length > 0 && (
        <ul className={styles.warningList}>
          {result.warnings.map((text) => (
            <li key={text}>{text}</li>
          ))}
        </ul>
      )}

      {result.episodes.length === 0 ? (
        <p className={styles.meta}>Tidak ada arus gangguan yang terbaca di rekaman, jadi tidak ada lokasi yang diuji.</p>
      ) : (
        result.episodes.map((episode, n) => (
          <EpisodeBlock key={`${episode.start_s}-${n}`} episode={episode} heading={multiple ? `Gangguan ${n + 1}` : null} />
        ))
      )}

      <details className={styles.details}>
        <summary>Rekaman dan pemetaan CT</summary>
        <div className={styles.detailsBody}>
          <div>
            <h4 className={styles.subTitle}>Jam tiap rele</h4>
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th scope="col">Rele</th>
                    <th scope="col">Offset</th>
                    <th scope="col">Penyelarasan</th>
                  </tr>
                </thead>
                <tbody>
                  {result.recorders.map((r) => (
                    <tr key={r.id}>
                      <td>{r.label}</td>
                      <td className={styles.mono}>
                        {r.offset_s >= 0 ? "+" : ""}
                        {r.offset_s.toFixed(3)} s
                      </td>
                      <td>{r.alignment}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {result.mapping.length > 0 && (
            <div>
              <h4 className={styles.subTitle}>Pemetaan otomatis dari rekaman</h4>
              <ul className={styles.plainList}>
                {result.mapping.map((d, i) => (
                  <li key={`${d.subject}-${i}`}>
                    <strong>{d.subject}</strong> — {d.evidence} <span className={styles.rule}>[{d.rule}]</span>
                    {!d.certain && (
                      <span className={`${styles.chip} ${styles.toneWarning}`}>belum pasti, diuji sebagai alternatif</span>
                    )}
                  </li>
                ))}
              </ul>
              {result.mapping_choice && <p className={styles.meta}>⇒ {result.mapping_choice}</p>}
            </div>
          )}

          {result.notes.length > 0 && (
            <div>
              <h4 className={styles.subTitle}>Catatan pembacaan</h4>
              <ul className={styles.plainList}>
                {result.notes.map((text) => (
                  <li key={text}>{text}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </details>
    </>
  );
}

function EpisodeBlock({ episode, heading }: { episode: FaultSectionEpisode; heading: string | null }) {
  const s = episode.summary;
  return (
    <div className={styles.episode}>
      <div className={styles.episodeHead}>
        {heading && <h3 className={styles.episodeTitle}>{heading}</h3>}
        <span className={styles.mono}>
          {episode.start_clock} – {episode.end_clock}
        </span>
        <span className={styles.meta}>({durationMs(episode)} ms)</span>
      </div>

      <p className={styles.lead}>
        {s.all_strong_consistent
          ? "Lokasi yang konsisten dengan semua bukti kuat:"
          : "Tidak ada lokasi yang konsisten dengan semua bukti kuat. Lokasi dengan kontradiksi paling sedikit:"}
      </p>
      <ul className={styles.locations}>
        {s.consistent.map((c) => (
          <li key={c.section} className={`${styles.location} ${s.all_strong_consistent ? styles.locationOk : styles.locationWeak}`}>
            <span className={styles.locationLabel}>{c.label}</span>
            <span className={styles.meta}>
              {c.agreements} bukti cocok
              {c.weak_mismatches > 0 && `, ${c.weak_mismatches} bukti lemah tidak cocok`}
              {c.variants > 1 && ` · ${c.variants} varian topologi`}
            </span>
            {c.weak_texts.length > 0 && (
              <ul className={styles.subList}>
                {c.weak_texts.map((text) => (
                  <li key={text}>{text}</li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>

      {s.inferred.length > 0 && (
        <div>
          <h4 className={styles.subTitle}>Disimpulkan dari rekaman</h4>
          <ul className={styles.plainList}>
            {s.inferred.map((text) => (
              <li key={text}>{text}</li>
            ))}
          </ul>
        </div>
      )}

      {s.notes.length > 0 && (
        <div className={styles.noteBox}>
          <h4 className={styles.subTitle}>Catatan — tidak cocok dengan semua lokasi di atas, jadi bukan petunjuk lokasi</h4>
          <ul className={styles.plainList}>
            {s.notes.map((note) => (
              <li key={note.text}>
                {note.text} <span className={styles.hint}>→ {note.hint}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {s.stuck_only.length > 0 && (
        <div>
          <h4 className={styles.subTitle}>Hanya mungkin bila ada CB yang gagal membuka</h4>
          <ul className={styles.plainList}>
            {s.stuck_only.map((label) => (
              <li key={label}>{label}</li>
            ))}
          </ul>
        </div>
      )}

      {s.rejected.length > 0 && (
        <details className={styles.details}>
          <summary>Lokasi lain yang ditolak ({s.rejected.length})</summary>
          <ul className={`${styles.plainList} ${styles.detailsBody}`}>
            {s.rejected.map((r) => (
              <li key={r.section}>
                <strong>{r.label}</strong>
                {r.weak_only && <span className={`${styles.chip} ${styles.toneWarning}`}>bukti lemah</span>}
                {r.reason && <div className={styles.reason}>{r.reason}</div>}
              </li>
            ))}
          </ul>
        </details>
      )}

      {episode.intervals.length > 0 && (
        <details className={styles.details}>
          <summary>Interval yang diuji ({episode.intervals.length})</summary>
          <div className={`${styles.tableWrap} ${styles.detailsBody}`}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th scope="col">Interval</th>
                  <th scope="col">Waktu</th>
                  <th scope="col">Keadaan CB</th>
                </tr>
              </thead>
              <tbody>
                {episode.intervals.map((interval) => (
                  <tr key={interval.id}>
                    <td className={styles.mono}>{interval.id}</td>
                    <td className={styles.mono}>
                      {interval.start_clock}–{interval.end_clock}
                    </td>
                    <td>{interval.label || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}

      <p className={styles.meta}>
        {s.observations} bukti ({s.strong_observations} kuat), {s.hypotheses} hipotesis diuji.
      </p>
    </div>
  );
}
