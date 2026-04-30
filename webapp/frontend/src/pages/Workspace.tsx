import { Component, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { Navigate, useNavigate, useParams } from "react-router-dom";

import { fetchAnalysis } from "../api/client";
import COMTRADEExplorer from "../components/panels/COMTRADEExplorer";
import CTVTRatioCorrection from "../components/panels/CTVTRatioCorrection";
import SOETimeline from "../components/panels/SOETimeline";
import AIFaultAnalysis21 from "../components/relay/relay21/AIFaultAnalysis21";
import ElectricalParams21 from "../components/relay/relay21/ElectricalParams21";
import FaultTypeBadge21 from "../components/relay/relay21/FaultTypeBadge21";
import ImpedanceLocus from "../components/relay/relay21/ImpedanceLocus";
import AIFaultAnalysis87L from "../components/relay/relay87l/AIFaultAnalysis87L";
import DiffRestraintPlot from "../components/relay/relay87l/DiffRestraintPlot";
import FaultRecap87T from "../components/relay/relay87t/FaultRecap87T";
import OvercurrentOverlay from "../components/relay/relay_ocr/OvercurrentOverlay";
import { useAnalysis } from "../context/AnalysisContext";
import type { ComtradeData } from "../context/AnalysisContext";

import styles from "./Workspace.module.css";

class PanelErrorBoundary extends Component<{ label: string; children: ReactNode }, { error: string | null }> {
  constructor(props: { label: string; children: ReactNode }) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { error: error.message };
  }

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            padding: "16px 20px",
            background: "#fff7ed",
            border: "1px solid #fdba74",
            borderRadius: 10,
            color: "#9a3412",
            fontSize: "0.85rem",
            marginBottom: 16,
          }}
        >
          <strong>{this.props.label} error:</strong> {this.state.error}
        </div>
      );
    }

    return this.props.children;
  }
}

const RELAY_LABELS: Record<string, string> = {
  "21": "21 - Distance",
  "87L": "87L - Differential Line",
  "87T": "87T / Transformer Differential",
  OCR: "50/51 - Overcurrent",
  REF: "REF / GFR / SBEF",
  SBEF: "SBEF",
};

function formatDurationMs(time: number[]) {
  if (time.length < 2) return "-";
  return ((time[time.length - 1] - time[0]) * 1000).toFixed(1);
}

export default function Workspace() {
  const { relayType: urlType, analysisId } = useParams<{ relayType: string; analysisId: string }>();
  const { relayType: ctxRelayType, reset } = useAnalysis();
  const navigate = useNavigate();

  const relayType = urlType ?? ctxRelayType ?? "21";
  const [comtrade, setComtrade] = useState<ComtradeData | null>(null);
  const [dataRevision, setDataRevision] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!analysisId) return;

    setLoading(true);
    setError(null);
    fetchAnalysis(analysisId)
      .then(setComtrade)
      .catch(() => setError("Failed to load analysis data. The session may have expired."))
      .finally(() => setLoading(false));
  }, [analysisId]);

  if (!analysisId) {
    return <Navigate to={ctxRelayType ? "/upload" : "/"} replace />;
  }

  const currentAnalysisId = analysisId;

  function handleReset() {
    reset();
    navigate("/");
  }

  function handlePrint() {
    if (typeof window !== "undefined") {
      window.print();
    }
  }

  function renderPrimaryAnalysisPanel() {
    if (relayType === "21") {
      return (
        <>
          <PanelErrorBoundary label="AI Fault Analysis">
            <AIFaultAnalysis21 analysisId={currentAnalysisId} dataRevision={dataRevision} />
          </PanelErrorBoundary>
          <PanelErrorBoundary label="Parameter Elektrikal">
            <ElectricalParams21 analysisId={currentAnalysisId} dataRevision={dataRevision} />
          </PanelErrorBoundary>
        </>
      );
    }

    if (relayType === "87L") {
      return (
        <>
          <PanelErrorBoundary label="Fault Recap 87L">
            <FaultRecap87T comtrade={comtrade!} relayLabel="Line Differential (87L)" />
          </PanelErrorBoundary>
          <PanelErrorBoundary label="AI Fault Analysis 87L">
            <AIFaultAnalysis87L analysisId={currentAnalysisId} />
          </PanelErrorBoundary>
        </>
      );
    }

    if (relayType === "87T" || relayType === "REF") {
      return (
        <PanelErrorBoundary label="Fault Recap 87T">
          <FaultRecap87T comtrade={comtrade!} />
        </PanelErrorBoundary>
      );
    }

    return (
      <>
        <PanelErrorBoundary label="Fault Recap">
          <FaultRecap87T
            comtrade={comtrade!}
            relayLabel={relayType === "SBEF" ? "SBEF / Ground Fault" : "Overcurrent (50/51)"}
          />
        </PanelErrorBoundary>
        <PanelErrorBoundary label="Overcurrent Overlay">
          <OvercurrentOverlay
            analysisId={currentAnalysisId}
            relayType={relayType === "SBEF" ? "SBEF" : "OCR"}
          />
        </PanelErrorBoundary>
      </>
    );
  }

  function renderRelayVisualPanel() {
    if (relayType === "21") {
      return (
        <PanelErrorBoundary label="Impedance Locus">
          <ImpedanceLocus analysisId={currentAnalysisId} dataRevision={dataRevision} />
        </PanelErrorBoundary>
      );
    }

    if (relayType === "87L") {
      return (
        <PanelErrorBoundary label="Diff/Restraint">
          <DiffRestraintPlot analysisId={currentAnalysisId} relayType="87L" />
        </PanelErrorBoundary>
      );
    }

    if (relayType === "87T" || relayType === "REF") {
      return (
        <>
          <PanelErrorBoundary label="Diff/Restraint">
            <DiffRestraintPlot analysisId={currentAnalysisId} relayType="87T" />
          </PanelErrorBoundary>
          <div className={styles.pendingNote}>
            AI fault cause analysis for transformer differential is pending.
          </div>
        </>
      );
    }

    return null;
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <button className={styles.homeBtn} onClick={handleReset} type="button">
            {"<"} New Analysis
          </button>
          <div>
            <span className={styles.relayBadge}>{RELAY_LABELS[relayType] ?? relayType}</span>
            {comtrade && (
              <>
                <span className={styles.stationName}>{comtrade.station_name}</span>
                <span className={styles.deviceId}>{comtrade.rec_dev_id}</span>
              </>
            )}
          </div>
        </div>
        {comtrade && (
          <div className={styles.headerRight}>
            <span className={styles.meta}>{formatDurationMs(comtrade.time)} ms</span>
            <span className={styles.meta}>{comtrade.total_samples} samples</span>
            <span className={styles.meta}>{comtrade.sampling_rates[0]?.[0]} Hz</span>
            <span className={styles.meta}>{comtrade.frequency} Hz nominal</span>
            <span className={styles.meta}>{comtrade.analog_channels.length} analog</span>
            <span className={styles.meta}>{comtrade.status_channels.length} digital</span>
            <button className={styles.printBtn} onClick={handlePrint} type="button">
              Print / PDF
            </button>
          </div>
        )}
      </header>

      <main className={styles.content}>
        {loading && <div className={styles.loadingState}>Loading waveforms...</div>}

        {error && <div className={styles.errorState}>{error}</div>}

        {!loading && !error && comtrade && (
          <div className={styles.workspaceShell}>
            <div className={styles.resultsLayout}>
              <aside className={styles.leftPanel}>
                {relayType === "21" && (
                  <PanelErrorBoundary label="Jenis Gangguan">
                    <FaultTypeBadge21 analysisId={currentAnalysisId} dataRevision={dataRevision} />
                  </PanelErrorBoundary>
                )}

                <PanelErrorBoundary label="CT/VT Ratio Correction">
                  <CTVTRatioCorrection
                  analysisId={currentAnalysisId}
                  comtrade={comtrade}
                  onUpdate={(updated) => { setComtrade(updated); setDataRevision((r) => r + 1); }}
                />
                </PanelErrorBoundary>

                <PanelErrorBoundary label="SOE Timeline">
                  <SOETimeline comtrade={comtrade} />
                </PanelErrorBoundary>
              </aside>

              <section className={styles.rightPanel}>
                {renderPrimaryAnalysisPanel()}

                <PanelErrorBoundary label="COMTRADE Explorer">
                  <COMTRADEExplorer comtrade={comtrade} />
                </PanelErrorBoundary>

                {renderRelayVisualPanel()}
              </section>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
