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
    id: "21",
    label: "21 - Distance",
    subtitle: "Distance protection and fault cause triage",
    tooltip:
      "Impedance locus diagram, zone polygon editor, phase and earth loop selector, plus AI-based fault cause ranking.",
    icon: "21",
  },
  {
    id: "87L",
    label: "87L - Differential Line",
    subtitle: "Line differential diagnostics",
    tooltip:
      "Differential vs restraint characteristic plot, SIPROTEC-style parameter editor, and internal vs external fault analysis.",
    icon: "87L",
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
          Pick the protection family first, then upload the COMTRADE pair for analysis.
        </p>
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
        Upload a matching <code>.cfg</code> and <code>.dat</code> pair after selecting the relay type.
      </footer>
    </div>
  );
}
