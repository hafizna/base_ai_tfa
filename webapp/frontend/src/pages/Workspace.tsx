import { Component, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { Navigate, useNavigate, useParams } from "react-router-dom";

type PlotlyToImageFn = (
  gd: HTMLElement,
  opts: { format: "png"; width: number; height: number; scale: number },
) => Promise<string>;

function getPlotlyToImage(): PlotlyToImageFn | null {
  const plotly = (window as unknown as { Plotly?: { toImage?: PlotlyToImageFn } }).Plotly;
  return plotly?.toImage ?? null;
}

import {
  aiFaultAnalysis21,
  aiFaultAnalysis87L,
  extractFeatures21,
  fetchAnalysis,
  fetchFullSoe21,
  generateReport,
  type ReportChart,
  type ReportSoeEvent,
} from "../api/client";
import COMTRADEExplorer from "../components/panels/COMTRADEExplorer";
import CTVTRatioCorrection from "../components/panels/CTVTRatioCorrection";
import AIFaultAnalysis21 from "../components/relay/relay21/AIFaultAnalysis21";
import ElectricalParams21 from "../components/relay/relay21/ElectricalParams21";
import FaultTypeBadge21 from "../components/relay/relay21/FaultTypeBadge21";
import ImpedanceLocus from "../components/relay/relay21/ImpedanceLocus";
import AIFaultAnalysis87L from "../components/relay/relay87l/AIFaultAnalysis87L";
import DiffRestraintPlot from "../components/relay/relay87l/DiffRestraintPlot";
import FaultRecap87T from "../components/relay/relay87t/FaultRecap87T";
import OvercurrentOverlay from "../components/relay/relay_ocr/OvercurrentOverlay";
import type { OCRReportSettings } from "../components/relay/relay_ocr/OvercurrentOverlay";
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
  CCP: "CCP / Stub Differential",
  "87T": "87T / Transformer Differential",
  OCR: "50/51 - Overcurrent",
  REF: "REF / GFR / SBEF",
  SBEF: "SBEF",
};

const DEFAULT_DIFF_PARAMS = {
  device_type: "SP5",
  idiff_pickup: 0.20,
  slope1: 0.30,
  intersection1: 0.30,
  slope2: 0.70,
  intersection2: 2.50,
  idiff_fast: 7.50,
};

function formatDurationMs(time: number[]) {
  if (time.length < 2) return "-";
  return ((time[time.length - 1] - time[0]) * 1000).toFixed(1);
}

function formatPrintDate() {
  return new Intl.DateTimeFormat("id-ID", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date());
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
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const [ocrReportSettings, setOcrReportSettings] = useState<OCRReportSettings | null>(null);

  useEffect(() => {
    if (!analysisId) return;

    setLoading(true);
    setError(null);
    setOcrReportSettings(null);
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

  async function captureWorkspaceCharts(): Promise<ReportChart[]> {
    const toImage = getPlotlyToImage();
    if (!toImage) {
      console.warn("Plotly runtime not available — chart export skipped.");
      return [];
    }
    const nodes = Array.from(
      document.querySelectorAll<HTMLElement>(".js-plotly-plot"),
    );
    const charts: ReportChart[] = [];
    const seenIds = new Set<string>();

    for (const node of nodes) {
      const gd = node as HTMLElement & {
        _fullLayout?: { title?: { text?: string } | string };
        layout?: { title?: { text?: string } | string };
      };

      // Read explicit id/title from the nearest ancestor that carries
      // data-pdf-chart-id (set by panel components). This is the source of
      // truth — title-regex below is only a legacy fallback.
      const tagged = (node.closest("[data-pdf-chart-id]") as HTMLElement | null);
      let id: string | null = tagged?.dataset.pdfChartId ?? null;
      let title: string = tagged?.dataset.pdfChartTitle ?? "";

      if (tagged?.dataset.pdfReady === "false") continue;

      if (!id) {
        const rawTitle =
          (typeof gd._fullLayout?.title === "object" && gd._fullLayout?.title?.text) ||
          (typeof gd._fullLayout?.title === "string" && gd._fullLayout?.title) ||
          (typeof gd.layout?.title === "object" && gd.layout?.title?.text) ||
          (typeof gd.layout?.title === "string" && gd.layout?.title) ||
          "";
        title = String(rawTitle || "").trim();
        const lowerTitle = title.toLowerCase();
        if (/impedance|locus|r-?x|phase-to-/.test(lowerTitle)) {
          id = "impedance_locus";
        } else if (/diff|restraint/.test(lowerTitle)) {
          id = "diff_restraint";
        } else if (/overcurrent|ocr|inverse|tcc/.test(lowerTitle)) {
          id = "overcurrent_overlay";
        }
      }

      if (!id) continue;
      // Digital status is rendered as an SOE table in the PDF, not a chart.
      if (id === "digital_status") continue;
      // Skip duplicates (e.g. when the user is in a multi-tab UI that
      // accidentally double-mounts the same chart).
      if (seenIds.has(id)) continue;

      try {
        const dataUrl = await toImage(gd, {
          format: "png",
          width: 1400,
          height: 900,
          scale: 2,
        });
        const image_b64 = String(dataUrl).replace(/^data:image\/png;base64,/, "");
        charts.push({ id, title, image_b64 });
        seenIds.add(id);
      } catch (err) {
        console.warn(`Failed to export Plotly chart "${title || id}":`, err);
      }
    }

    // Preferred order in the report: locus first, then waveforms, then
    // differential/overcurrent overlays. Anything unknown trails. Digital
    // status is rendered as a SOE table, not a chart.
    const order = [
      "impedance_locus",
      "impedance_locus_ground",
      "impedance_locus_phase",
      "waveform_voltage",
      "waveform_current",
      "waveform_strip",
      "diff_restraint",
      "overcurrent_overlay",
    ];
    const rank = (id: string) => {
      const i = order.indexOf(id);
      return i === -1 ? order.length : i;
    };
    charts.sort((a, b) => rank(a.id) - rank(b.id));
    return charts;
  }

  async function fetchAiAnalysisSafe(): Promise<Record<string, unknown> | null> {
    try {
      if (relayType === "21") {
        const features = await extractFeatures21(currentAnalysisId);
        const result = await aiFaultAnalysis21(currentAnalysisId, features);
        return result as Record<string, unknown>;
      }
      if (relayType === "87L") {
        const result = await aiFaultAnalysis87L(currentAnalysisId, DEFAULT_DIFF_PARAMS);
        return result as Record<string, unknown>;
      }
      return null;
    } catch (err) {
      console.warn("Failed to fetch AI analysis for report:", err);
      return null;
    }
  }

  async function fetchSoeEventsSafe(): Promise<ReportSoeEvent[]> {
    try {
      // Full SOE (every digital transition) — the curated /locus-events
      // endpoint only emits protection-relevant channels, which is too
      // narrow for the report's SOE section.
      const { events } = await fetchFullSoe21(currentAnalysisId);
      return events.map((e) => ({
        time_ms: e.time_ms,
        rel_ms: e.rel_ms,
        channel: e.channel,
        state: e.state,
        category: e.category,
        label: e.label,
      }));
    } catch (err) {
      console.warn("Failed to fetch SOE events for report:", err);
      return [];
    }
  }

  async function handleDownloadPdf() {
    if (isGeneratingPdf) return;
    setIsGeneratingPdf(true);
    try {
      const [charts, aiAnalysis, soeEvents] = await Promise.all([
        captureWorkspaceCharts(),
        fetchAiAnalysisSafe(),
        fetchSoeEventsSafe(),
      ]);
      const relaySettings = (relayType === "OCR" || relayType === "SBEF") && ocrReportSettings
        ? { ocr: ocrReportSettings }
        : null;

      const blob = await generateReport(currentAnalysisId, {
        relay_type: relayType,
        ai_analysis: aiAnalysis,
        charts,
        soe_events: soeEvents,
        relay_settings: relaySettings,
      });

      const stationSlug = (comtrade?.station_name || "report")
        .replace(/\s+/g, "_")
        .replace(/[\\/]/g, "-");
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `laporan_gangguan_${stationSlug}_${currentAnalysisId.slice(0, 8)}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to generate PDF report:", err);
      alert("Gagal membuat laporan PDF. Cek console untuk detail.");
    } finally {
      setIsGeneratingPdf(false);
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

    if (relayType === "87L" || relayType === "CCP") {
      return (
        <>
          <PanelErrorBoundary label={relayType === "CCP" ? "CCP / Stub Differential Recap" : "Fault Recap 87L"}>
            <FaultRecap87T
              comtrade={comtrade!}
              relayLabel={relayType === "CCP" ? "CCP / Stub Differential" : "Line Differential (87L)"}
            />
          </PanelErrorBoundary>
          {relayType === "87L" && (
            <PanelErrorBoundary label="AI Fault Analysis 87L">
              <AIFaultAnalysis87L analysisId={currentAnalysisId} />
            </PanelErrorBoundary>
          )}
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
            onReportSettingsChange={setOcrReportSettings}
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

    if (relayType === "CCP") {
      return (
        <div className={styles.pendingNote}>
          CCP-specific CT group comparison is pending. Use the operation recap and COMTRADE Explorer for now.
        </div>
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
            {comtrade && (
              <button
                className={styles.downloadPdfBtn}
                onClick={handleDownloadPdf}
                disabled={isGeneratingPdf}
                type="button"
              >
                {isGeneratingPdf ? "Menyiapkan PDF…" : "📄 Download PDF Report"}
              </button>
            )}
          </div>
        )}
      </header>

      <main className={styles.content}>
        {loading && <div className={styles.loadingState}>Loading waveforms...</div>}

        {error && <div className={styles.errorState}>{error}</div>}

        {!loading && !error && comtrade && (
          <div className={styles.workspaceShell}>
            <section className={styles.printReportHeader}>
              <div>
                <div className={styles.printKicker}>COMTRADE Fault Analysis Report</div>
                <h1>{comtrade.station_name || "Unknown Station"}</h1>
                <p>{RELAY_LABELS[relayType] ?? relayType} | {comtrade.rec_dev_id || "Unknown Device"}</p>
              </div>
              <dl>
                <div><dt>Analysis ID</dt><dd>{currentAnalysisId}</dd></div>
                <div><dt>Printed</dt><dd>{formatPrintDate()}</dd></div>
                <div><dt>Duration</dt><dd>{formatDurationMs(comtrade.time)} ms</dd></div>
                <div><dt>Samples</dt><dd>{comtrade.total_samples}</dd></div>
                <div><dt>Sampling</dt><dd>{comtrade.sampling_rates[0]?.[0] ?? "-"} Hz</dd></div>
                <div><dt>Channels</dt><dd>{comtrade.analog_channels.length} analog / {comtrade.status_channels.length} digital</dd></div>
              </dl>
            </section>
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
