import { useEffect, useMemo, useState } from "react";
import { Navigate, useParams } from "react-router-dom";

import { fetchTwsAnalysis } from "../api/client";
import type { TwsCdbData, TwsEndpoint, TwsResult } from "../api/client";
import Plot from "../components/plot/PlotlyChart";
import styles from "./TwsViewer.module.css";

const MAX_PLOT_POINTS = 3600;
const WAVEFORM_RANGE_KM: [number, number] = [-40, 300];
const PHASE_CENTER_PERCENT: Record<string, number> = {
  A: 200,
  B: 0,
  C: -200,
};
const PHASE_COLORS: Record<string, string> = {
  A: "#ef1d1d",
  B: "#0a8f24",
  C: "#1438e8",
};

function formatKm(value: number | null | undefined, digits = 2) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value.toFixed(digits).replace(".", ",");
}

function formatDateTime(epochSeconds: number | null | undefined) {
  if (typeof epochSeconds !== "number" || !Number.isFinite(epochSeconds) || epochSeconds <= 0) return "-";
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(epochSeconds * 1000));
}

function sampledWaveform(endpoint: TwsEndpoint, channelSamples: number[], sampleDistanceKm: number) {
  const markerPoint = (endpoint.software_trigger_point || 0) + (endpoint.trigger_delay || 0);
  const visibleIndexes = channelSamples
    .map((_, idx) => idx)
    .filter((idx) => {
      const distanceKm = (idx - markerPoint) * sampleDistanceKm;
      return distanceKm >= WAVEFORM_RANGE_KM[0] && distanceKm <= WAVEFORM_RANGE_KM[1];
    });
  const step = Math.max(1, Math.ceil(visibleIndexes.length / MAX_PLOT_POINTS));
  const x: number[] = [];
  const y: number[] = [];

  for (let cursor = 0; cursor < visibleIndexes.length; cursor += step) {
    const i = visibleIndexes[cursor];
    x.push((i - markerPoint) * sampleDistanceKm);
    y.push(channelSamples[i]);
  }

  const lastVisible = visibleIndexes[visibleIndexes.length - 1];
  if (lastVisible !== undefined && x[x.length - 1] !== (lastVisible - markerPoint) * sampleDistanceKm) {
    const idx = lastVisible;
    x.push((idx - markerPoint) * sampleDistanceKm);
    y.push(channelSamples[idx]);
  }

  return { x, y };
}

function waveformTraces(endpoint: TwsEndpoint, result: TwsResult, amplitudeScale: number): Plotly.Data[] {
  return endpoint.channels.map((channel) => {
    const series = sampledWaveform(endpoint, channel.samples, result.sample_distance_km);
    const maxAbs = Math.max(1, amplitudeScale);
    const center = PHASE_CENTER_PERCENT[channel.phase] ?? 0;
    const y = series.y.map((v) => center + (v / maxAbs) * 50);
    const customdata = series.y.map((v) => [v]);
    return {
      x: series.x,
      y,
      customdata,
      type: "scattergl",
      mode: "lines",
      name: `Phase ${channel.phase}`,
      line: { color: PHASE_COLORS[channel.phase] ?? "#334155", width: 1.3 },
      hovertemplate: `Phase ${channel.phase}<br>%{x:.2f} km from marker<br>%{customdata[0]:.0f} raw<extra></extra>`,
      showlegend: false,
    } as Plotly.Data;
  });
}

function WaveformPane({
  endpoint,
  result,
  amplitudeScale,
}: {
  endpoint: TwsEndpoint;
  result: TwsResult;
  amplitudeScale: number;
}) {
  const traces = useMemo(() => waveformTraces(endpoint, result, amplitudeScale), [amplitudeScale, endpoint, result]);

  const layout: Partial<Plotly.Layout> = {
    height: 390,
    margin: { l: 58, r: 12, t: 8, b: 38 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    dragmode: "pan",
    xaxis: {
      title: { text: "Distance from corrected trigger marker (km)", font: { size: 11 } },
      range: WAVEFORM_RANGE_KM,
      dtick: 40,
      gridcolor: "#e5e7eb",
      zeroline: false,
      tickfont: { size: 11 },
    },
    yaxis: {
      title: { text: "Shared normalized amplitude", font: { size: 11 } },
      range: [-290, 290],
      tickmode: "array",
      tickvals: [-250, -200, -150, -50, 0, 50, 150, 200, 250],
      ticktext: ["-50%", "C", "+50%", "-50%", "B", "+50%", "-50%", "A", "+50%"],
      gridcolor: "#eef2f7",
      zeroline: false,
      fixedrange: true,
      tickfont: { size: 11 },
    },
    shapes: [
      {
        type: "line",
        x0: 0,
        x1: 0,
        y0: -285,
        y1: 285,
        line: { color: "#111827", width: 1, dash: "dot" },
      },
      ...Object.values(PHASE_CENTER_PERCENT).map((y) => ({
        type: "line" as const,
        x0: WAVEFORM_RANGE_KM[0],
        x1: WAVEFORM_RANGE_KM[1],
        y0: y,
        y1: y,
        line: { color: "#cbd5e1", width: 1 },
      })),
    ],
    annotations: [
      {
        x: 0,
        y: 282,
        xref: "x",
        yref: "y",
        text: "M",
        showarrow: false,
        font: { size: 11, color: "#b91c1c" },
      },
    ],
  };

  return (
    <section className={styles.pane}>
      <div className={styles.faultBanner}>
        <span>Fault Occurred at</span>
        <strong>{formatKm(endpoint.fault_distance_km)} km</strong>
        <span>from {endpoint.station_display_name || endpoint.station_name}</span>
      </div>
      <div className={styles.distanceFormula}>
        <span>Distance (km)</span>
        <strong>0</strong>
        <span>-</span>
        <strong>0</strong>
        <span>= 0</span>
      </div>
      <Plot
        data={traces}
        layout={layout}
        config={{ responsive: true, displaylogo: false, scrollZoom: true }}
        className={styles.plot}
        useResizeHandler
        style={{ width: "100%" }}
      />
      <EndpointTable endpoint={endpoint} />
    </section>
  );
}

function EndpointTable({ endpoint }: { endpoint: TwsEndpoint }) {
  const rows = [
    {
      type: "FL",
      record: endpoint.record_number,
      gps: endpoint.gps_time_tag,
      time: endpoint.event_time_us,
    },
  ];

  return (
    <div className={styles.tableShell}>
      <table className={styles.eventTable}>
        <thead>
          <tr>
            <th>Event Time</th>
            <th>Record Number</th>
            <th>GPS Locked</th>
            <th>GPS Time Tag</th>
            <th>Type</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${endpoint.role}-${row.record}`} className={styles.selectedRow}>
              <td>{formatDateTime(row.time)}</td>
              <td>{row.record}</td>
              <td>{endpoint.gps_locked ? "Locked" : "Unlocked"}</td>
              <td>{row.gps}</td>
              <td>{row.type}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResultSummary({ result }: { result: TwsResult }) {
  const endpoints = result.endpoints.filter((endpoint) => endpoint.role === "X" || endpoint.role === "Y");
  return (
    <section className={styles.resultSummary}>
      <div className={styles.resultHeader}>
        <div>
          <span className={styles.kicker}>Computed TWS FL Result</span>
          <h2>{result.circuit_name || result.segment_name || "Line fault"}</h2>
        </div>
        <div className={styles.lineLengthPill}>{formatKm(result.line_length_km, 1)} km line</div>
      </div>
      <div className={styles.resultCards}>
        {endpoints.map((endpoint) => (
          <article key={endpoint.role} className={styles.resultCard}>
            <div className={styles.endpointRole}>{endpoint.role}</div>
            <div>
              <strong>{endpoint.station_display_name || endpoint.station_name}</strong>
              <span>{endpoint.feeder_display_name || endpoint.feeder_name}</span>
            </div>
            <div className={styles.distanceValue}>{formatKm(endpoint.fault_distance_km)} km</div>
            <span className={styles.distanceLabel}>from this terminal</span>
          </article>
        ))}
      </div>
      <p className={styles.resultNote}>
        The distances above are read from the export result metadata. The waveform view is diagnostic context for the
        traveling-wave records and is not used as the primary distance display here.
      </p>
    </section>
  );
}

function LineDiagram({ result }: { result: TwsResult }) {
  const x = result.endpoints.find((endpoint) => endpoint.role === "X") ?? result.endpoints[0];
  const y = result.endpoints.find((endpoint) => endpoint.role === "Y") ?? result.endpoints[1];
  const inferredLength = (x?.fault_distance_km ?? 0) + (y?.fault_distance_km ?? 0);
  const lineLength = result.line_length_km || inferredLength || 0;
  const faultPct = lineLength > 0 ? Math.max(0, Math.min(1, ((x?.fault_distance_km ?? 0) / lineLength))) : 0.5;
  const markerLeftPct = 24 + faultPct * 52;

  return (
    <section className={styles.lineSection}>
      <div className={styles.lineMeta}>
        <strong>Line Length</strong>
        <span>X - Y</span>
        <mark>{formatKm(lineLength, 1)}km</mark>
      </div>
      <div className={styles.lineTrack}>
        <div className={styles.endpoint} style={{ left: "24%" }}>
          <span>X</span>
          <strong>{x?.station_display_name || x?.station_name || "Endpoint X"}</strong>
        </div>
        <div className={styles.endpoint} style={{ left: "76%" }}>
          <span>Y</span>
          <strong>{y?.station_display_name || y?.station_name || "Endpoint Y"}</strong>
        </div>
        <div className={styles.conductor} />
        <div className={styles.faultMarker} style={{ left: `${markerLeftPct}%` }}>
          <span aria-hidden="true">⚡</span>
        </div>
      </div>
    </section>
  );
}

export default function TwsViewer() {
  const { analysisId } = useParams<{ analysisId: string }>();
  const [data, setData] = useState<TwsCdbData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!analysisId) return;
    setLoading(true);
    setError(null);
    fetchTwsAnalysis(analysisId)
      .then(setData)
      .catch(() => setError("Failed to load TWS FL analysis data. The session may have expired."))
      .finally(() => setLoading(false));
  }, [analysisId]);

  if (!analysisId) return <Navigate to="/" replace />;

  const result = data?.results[0];
  const xEndpoint = result?.endpoints.find((endpoint) => endpoint.role === "X") ?? result?.endpoints[0];
  const yEndpoint = result?.endpoints.find((endpoint) => endpoint.role === "Y") ?? result?.endpoints[1];
  const amplitudeScale = Math.max(
    1,
    ...(result?.endpoints.flatMap((endpoint) =>
      endpoint.channels.flatMap((channel) => channel.samples.map((sample) => Math.abs(sample))),
    ) ?? []),
  );

  return (
    <div className={styles.page}>
      <header className={styles.titleBar}>
        <strong>TWS Fault Locator</strong>
        <span>{data?.source_file ?? "Cashel .cdb export"}</span>
      </header>
      {loading && <main className={styles.state}>Loading TWS FL export...</main>}
      {error && <main className={styles.state}>{error}</main>}
      {!loading && !error && data && result && xEndpoint && yEndpoint && (
        <main className={styles.viewer}>
          <LineDiagram result={result} />
          {data.warnings.length > 0 && (
            <div className={styles.warning}>
              {data.warnings.join(" ")}
            </div>
          )}
          <ResultSummary result={result} />
          <details className={styles.waveformDetails}>
            <summary>Diagnostic waveform preview</summary>
            <div className={styles.waveformHint}>
              Waveform alignment and scaling are inferred from the Cashel export fields. Use this preview for context;
              use the computed result distances above for reporting.
            </div>
            <div className={styles.paneGrid}>
              <WaveformPane endpoint={xEndpoint} result={result} amplitudeScale={amplitudeScale} />
              <WaveformPane endpoint={yEndpoint} result={result} amplitudeScale={amplitudeScale} />
            </div>
          </details>
        </main>
      )}
    </div>
  );
}
