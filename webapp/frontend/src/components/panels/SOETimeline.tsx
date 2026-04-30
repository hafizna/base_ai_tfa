import type { ComtradeData } from "../../context/AnalysisContext";
import styles from "./Panel.module.css";

interface Props {
  comtrade: ComtradeData;
}

type EventCategory = "trip" | "ar" | "pickup" | "";

interface SOEEvent {
  channel: string;
  relMs: number;
  state: 0 | 1;
  category: EventCategory;
}

const TRIP_PATTERNS = ["TRIP", "PMT", "CB OPEN", "OPEN"];
const AR_PATTERNS = ["AR ", "RECLOSE", "RECLOS"];
const PICKUP_PATTERNS = ["PICKUP", "PKP", "OPERATE", "START"];

function detectCategory(name: string): EventCategory {
  const upper = name.toUpperCase();
  if (TRIP_PATTERNS.some((pattern) => upper.includes(pattern))) return "trip";
  if (AR_PATTERNS.some((pattern) => upper.includes(pattern))) return "ar";
  if (PICKUP_PATTERNS.some((pattern) => upper.includes(pattern))) return "pickup";
  return "";
}

function buildSOEEvents(comtrade: ComtradeData) {
  const baseMs = (comtrade.time[0] ?? 0) * 1000;
  const events: SOEEvent[] = [];

  comtrade.status_channels.forEach((channel) => {
    for (let i = 1; i < channel.samples.length; i += 1) {
      if (channel.samples[i] === channel.samples[i - 1]) continue;

      events.push({
        channel: channel.name,
        relMs: comtrade.time[i] * 1000 - baseMs,
        state: channel.samples[i] === 1 ? 1 : 0,
        category: detectCategory(channel.name),
      });
    }
  });

  events.sort((left, right) => left.relMs - right.relMs);
  return events;
}

export default function SOETimeline({ comtrade }: Props) {
  const events = buildSOEEvents(comtrade);

  if (events.length === 0) {
    return (
      <div className={styles.panel}>
        <div className={styles.panelHeader}>
          <h2 className={styles.panelTitle}>Urutan Event Digital (SOE)</h2>
        </div>
        <p className={styles.emptyText}>Tidak ada event digital yang berubah status pada rekaman ini.</p>
      </div>
    );
  }

  return (
    <div className={`${styles.panel} ${styles.soeFull}`}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Urutan Event Digital (SOE)</h2>
        <span className={styles.badge}>{events.length} events</span>
      </div>

      <div className={styles.soeCols}>
        {events.map((event, index) => (
          <div
            key={`${event.channel}-${event.relMs}-${index}`}
            className={styles.soeEventRow}
            data-cat={event.category || undefined}
          >
            <span className={styles.soeNum}>{index + 1}.</span>
            <span className={styles.soeTime}>
              {event.relMs >= 0 ? "+" : ""}
              {event.relMs.toFixed(2)} ms
            </span>
            <span className={styles.soeCh} title={event.channel}>
              {event.channel}
            </span>
            <span className={event.state === 1 ? styles.soeStateOn : styles.soeStateOff}>
              {event.state === 1 ? "ON ▲" : "OFF ▼"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
