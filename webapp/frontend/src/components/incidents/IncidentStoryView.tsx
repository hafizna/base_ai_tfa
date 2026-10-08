import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import type { FaultEpisodeOut } from "../../api/client";
import type {
  AiReading,
  CauseStory,
  ChecklistItem,
  Chip,
  IncidentStory,
  RecordRow,
  SequenceEntry,
  Tone,
} from "./incidentStory";
import JoinedWaveformView from "./JoinedWaveformView";
import styles from "./IncidentStoryView.module.css";

const TONE_CLASS: Record<Tone, string> = {
  fault: styles.toneFault,
  reclose: styles.toneReclose,
  neutral: styles.toneNeutral,
  warning: styles.toneWarning,
};

function ChipView({ chip }: { chip: Chip }) {
  return <span className={`${styles.chip} ${TONE_CLASS[chip.tone]}`}>{chip.label}</span>;
}

export function SummarySection({ story, rebuilding }: { story: IncidentStory; rebuilding: boolean }) {
  return (
    <section className={styles.card} aria-labelledby="ringkasan">
      <div className={styles.labelRow}>
        <span className={styles.eyebrow}>Ringkasan</span>
        {story.chips.map((chip) => (
          <ChipView key={chip.label} chip={chip} />
        ))}
        {rebuilding && <span className={`${styles.chip} ${styles.toneNeutral}`}>Menyusun ulang…</span>}
      </div>
      <h1 id="ringkasan" className={styles.headline}>
        {story.headline}
      </h1>
      <p className={styles.narrative}>{story.narrative}</p>
      <div className={styles.tiles}>
        {story.tiles.map((tile) => (
          <div key={tile.label} className={styles.tile}>
            <div className={styles.tileLabel}>{tile.label}</div>
            <div className={tile.mono ? styles.tileValueMono : styles.tileValue}>
              {tile.value}
              {tile.detail && <span className={styles.tileDetail}> ({tile.detail})</span>}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Connector({ entry }: { entry: Extract<SequenceEntry, { type: "connector" }> }) {
  const { connector } = entry;
  const refault = connector.kind === "refault";
  return (
    <div className={styles.connector}>
      <svg width="100%" height="10" viewBox="0 0 120 10" preserveAspectRatio="none" aria-hidden="true">
        <line
          x1="0"
          y1="5"
          x2="120"
          y2="5"
          className={refault ? styles.connectorLineFault : styles.connectorLine}
          strokeDasharray={connector.kind === "dead_time" ? "6 5" : undefined}
        />
      </svg>
      <span className={refault ? styles.connectorLabelFault : styles.connectorLabel}>{connector.label}</span>
      <span>{connector.detail}</span>
    </div>
  );
}

export function SequenceSection({ story }: { story: IncidentStory }) {
  return (
    <section className={styles.card} aria-labelledby="urutan">
      <div className={styles.titleRow}>
        <h2 id="urutan" className={styles.sectionTitle}>
          Urutan kejadian
        </h2>
        <span className={styles.meta}>{story.sequenceMeta}</span>
      </div>
      <div className={styles.sequence}>
        {story.sequence.map((entry, i) => {
          if (entry.type === "connector") return <Connector key={i} entry={entry} />;
          const { card } = entry;
          const kindClass =
            card.kind === "fault" ? styles.eventFault : card.kind === "reclose" ? styles.eventReclose : styles.eventAfter;
          return (
            <article key={i} className={`${styles.event} ${kindClass} ${card.emphasis ? styles.eventEmphasis : ""}`}>
              <div className={styles.eventTitle}>{card.title}</div>
              {card.time && <div className={styles.eventTime}>{card.time}</div>}
              <div className={styles.eventHeadline}>{card.headline}</div>
              {card.bullets.length > 0 &&
                (card.kind === "after" ? (
                  <p className={styles.eventNote}>{card.bullets.join(" ")}</p>
                ) : (
                  <ul className={styles.eventBullets}>
                    {card.bullets.map((b) => (
                      <li key={b}>{b}</li>
                    ))}
                  </ul>
                ))}
              {card.recordId && card.recordName && (
                <a className={styles.eventRecord} href={`#record-${card.recordId}`}>
                  {card.recordName}
                </a>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}

function AiRow({ reading }: { reading: AiReading }) {
  return (
    <>
      <div>
        <div className={styles.aiTitle}>{reading.title}</div>
        <div className={styles.mono}>{reading.recordName}</div>
      </div>
      <div className={styles.aiReading}>
        {reading.kind === "reading" && (
          <>
            <div className={styles.aiLine}>
              <span>{reading.cause}</span>
              <span className={styles.mono}>{reading.percent}%</span>
            </div>
            <div className={styles.bar}>
              <div className={styles.barFill} style={{ width: `${reading.percent ?? 0}%` }} />
            </div>
            {reading.note && <span className={styles.muted}>{reading.note}</span>}
          </>
        )}
        {reading.kind === "no_dominant" && (
          <>
            <div className={styles.aiLine}>
              <span>Tidak ada yang dominan</span>
              <span className={styles.mono}>{reading.candidates?.map((c) => c.percent).join(" / ")}%</span>
            </div>
            <span className={styles.muted}>{reading.candidates?.map((c) => c.cause).join(" · ")}</span>
          </>
        )}
        {(reading.kind === "skipped" || reading.kind === "unavailable") && (
          <span className={styles.muted}>{reading.note}</span>
        )}
      </div>
    </>
  );
}

export function CauseSection({ cause }: { cause: CauseStory }) {
  return (
    <section className={`${styles.card} ${styles.causeCard}`} aria-labelledby="penyebab">
      <div className={styles.labelRow}>
        <span className={styles.eyebrow}>Penyebab</span>
        <ChipView chip={cause.status} />
      </div>
      <h2 id="penyebab" className={styles.causeHeadline}>
        {cause.headline}
      </h2>
      {cause.pattern && (
        <div className={`${styles.patternBox} ${TONE_CLASS[cause.pattern.tone]}`}>
          <div className={styles.titleRow}>
            <span className={styles.strong}>{cause.pattern.title}</span>
            <span className={styles.patternStrength}>{cause.pattern.strength}</span>
          </div>
          <p className={styles.patternText}>{cause.pattern.text}</p>
        </div>
      )}
      {cause.ai.length > 0 && (
        <div className={styles.aiBlock}>
          <div className={styles.titleRow}>
            <span className={styles.strong}>Bacaan AI per rekaman gangguan</span>
            <span className={styles.meta}>Dibaca terpisah, tidak dirata-rata</span>
          </div>
          <div className={styles.aiGrid}>
            {cause.ai.map((reading, i) => (
              <AiRow key={i} reading={reading} />
            ))}
          </div>
        </div>
      )}
      <p className={styles.footnote}>{cause.footnote}</p>
    </section>
  );
}

function storageKey(incidentId: string) {
  return `incident-checklist:${incidentId}`;
}

function loadChecked(incidentId: string): string[] {
  try {
    const raw = window.localStorage.getItem(storageKey(incidentId));
    return raw ? (JSON.parse(raw) as string[]) : [];
  } catch {
    return [];
  }
}

export function ChecklistSection({
  incidentId,
  items,
  onOpenDetails,
}: {
  incidentId: string;
  items: ChecklistItem[];
  onOpenDetails: () => void;
}) {
  // Ticks are a per-viewer convenience kept in this browser only.
  const [checked, setChecked] = useState<string[]>(() => loadChecked(incidentId));
  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey(incidentId), JSON.stringify(checked));
    } catch {
      // storage unavailable (private window): ticks just won't persist
    }
  }, [incidentId, checked]);

  function toggle(id: string) {
    setChecked((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  return (
    <section className={`${styles.card} ${styles.checkCard}`} aria-labelledby="cek">
      <h2 id="cek" className={styles.sectionTitle}>
        Yang perlu dicek
      </h2>
      {items.map((item) => (
        <div key={item.id} className={styles.checkItem}>
          <input
            id={`cek-${item.id}`}
            type="checkbox"
            checked={checked.includes(item.id)}
            onChange={() => toggle(item.id)}
          />
          <div className={styles.checkBody}>
            <label htmlFor={`cek-${item.id}`} className={styles.checkLabel}>
              <span className={styles.strong}>{item.title}</span>
              <span className={styles.muted}>{item.detail}</span>
            </label>
            {item.link && (
              <Link className={styles.checkLink} to={item.link.to}>
                {item.link.label}
              </Link>
            )}
            {item.id === "bay" && (
              <button type="button" className={styles.linkButton} onClick={onOpenDetails}>
                Isi di Detail teknis
              </button>
            )}
          </div>
        </div>
      ))}
    </section>
  );
}

export function RecordsSection({
  incidentId,
  rows,
  episodes,
}: {
  incidentId: string;
  rows: RecordRow[];
  episodes: FaultEpisodeOut[];
}) {
  const [showWaveform, setShowWaveform] = useState(false);
  const [episodeId, setEpisodeId] = useState<string | null>(episodes[0]?.episode_id ?? null);
  return (
    <section id="rekaman" className={styles.card} aria-labelledby="rekaman-judul">
      <div className={styles.titleRow}>
        <h2 id="rekaman-judul" className={styles.sectionTitle}>
          Rekaman ({rows.length})
        </h2>
        {episodes.length > 0 && (
          <button type="button" className={styles.secondaryButton} onClick={() => setShowWaveform((v) => !v)}>
            {showWaveform ? "Tutup waveform gabungan" : "Lihat waveform gabungan"}
          </button>
        )}
      </div>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">Record</th>
              <th scope="col">Role in sequence</th>
              <th scope="col">Start</th>
              <th scope="col">Analysed line</th>
              <th scope="col">Notes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.recordId} id={`record-${row.recordId}`}>
                <td className={styles.mono}>{row.name}</td>
                <td>
                  <span className={`${styles.role} ${TONE_CLASS[row.roleTone]}`}>{row.roleLabel}</span>
                  {row.roleSuffix}
                </td>
                <td className={styles.mono}>{row.start}</td>
                <td>{row.line}</td>
                <td className={styles.muted}>{row.note || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {showWaveform && episodeId && (
        <div className={styles.waveform}>
          {episodes.length > 1 && (
            <div className={styles.episodeTabs} aria-label="Pilih gangguan">
              {episodes.map((ep) => (
                <button
                  key={ep.episode_id}
                  type="button"
                  aria-pressed={ep.episode_id === episodeId}
                  className={ep.episode_id === episodeId ? styles.episodeTabActive : styles.episodeTab}
                  onClick={() => setEpisodeId(ep.episode_id)}
                >
                  Gangguan #{ep.episode_index + 1}
                </button>
              ))}
            </div>
          )}
          <JoinedWaveformView incidentId={incidentId} episodeId={episodeId} />
        </div>
      )}
    </section>
  );
}
