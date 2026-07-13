import { useNavigate } from "react-router-dom";
import { useAnalysis } from "../context/AnalysisContext";
import type { RelayType } from "../context/AnalysisContext";
import styles from "./Landing.module.css";

interface RelayOption {
  id: RelayType;
  label: string;
  subtitle: string;
  tooltip: string;
  icon: string;
}

const RELAY_OPTIONS: RelayOption[] = [
  {
    id: "LINE",
    label: "21 / 87L - Line Protection",
    subtitle: "Distance and line differential diagnostics",
    tooltip:
      "Use this for line relays that combine distance and differential functions. If differential is blocked/unavailable, use the distance panels; if 87L data exists, diff/restraint panels are also shown.",
    icon: "21/87L",
  },
  {
    id: "CCP",
    label: "CCP / Stub Differential",
    subtitle: "CT group and stub differential anomaly review",
    tooltip:
      "Use this category for CCP or stub differential records with multiple CT groups, 87STB signals, relay faulty indications, or unstable current readings.",
    icon: "CCP",
  },
  {
    id: "87T",
    label: "87T / REF",
    subtitle: "Transformer differential and restricted earth fault",
    tooltip:
      "Use this category for transformer differential and REF-style review where the operating principle follows the same transformer-zone fault logic.",
    icon: "87T",
  },
  {
    id: "OCR",
    label: "50/51 / GFR",
    subtitle: "Time-delay overcurrent and ground fault review",
    tooltip:
      "Use this category for OCR and GFR relays that operate with time delay and pickup-based timing behavior.",
    icon: "50/51",
  },
  {
    id: "SBEF",
    label: "SBEF",
    subtitle: "Sensitive back earth fault timing review",
    tooltip:
      "Use this category when the earth-fault protection uses a dedicated SBEF timing characteristic and should be reviewed separately from GFR.",
    icon: "SBEF",
  },
  {
    id: "TWS_FL",
    label: "TWS FL",
    subtitle: "Traveling-wave fault locator viewer",
    tooltip:
      "Upload Qualitrol Cashel TWS FL .cdb exports, inspect paired-end waveforms, GPS tags, record numbers, and fault distance results.",
    icon: "TWS",
  },
];

export default function Landing() {
  const { setRelayType } = useAnalysis();
  const navigate = useNavigate();

  function select(type: RelayType) {
    setRelayType(type);
    navigate("/upload");
  }

  return (
    <div className={styles.page}>
      <div className={styles.orbLeft} />
      <div className={styles.orbRight} />

      <header className={styles.header}>
        <div className={styles.eyebrow}>PLN AI-Powered DFR Analytics</div>
        <div className={styles.logo}>DFR Analyser</div>
        <p className={styles.subtitle}>
          Pick the protection family first, then upload a COMTRADE file set or TWS FL export for analysis.
        </p>
        <button className={styles.simulatorLink} onClick={() => navigate("/simulator")} type="button">
          Event Notification Simulator
        </button>
        <button className={styles.simulatorLink} onClick={() => navigate("/incidents")} type="button">
          Incidents
        </button>
      </header>

      <main className={styles.grid}>
        {RELAY_OPTIONS.map((opt) => (
          <button
            key={opt.id}
            className={styles.card}
            onClick={() => select(opt.id)}
            title={opt.tooltip}
            type="button"
          >
            <span className={styles.cardIcon}>{opt.icon}</span>
            <span className={styles.cardLabel}>{opt.label}</span>
            <span className={styles.cardSub}>{opt.subtitle}</span>
            <span className={styles.cardTooltip}>{opt.tooltip}</span>
          </button>
        ))}
      </main>

      <footer className={styles.footer}>
        Upload one ABB <code>.cff</code>, a matching <code>.cfg</code> + <code>.dat</code> pair, or a TWS FL <code>.cdb</code> export.
      </footer>
    </div>
  );
}
