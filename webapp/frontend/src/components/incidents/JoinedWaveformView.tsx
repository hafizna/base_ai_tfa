import { useEffect, useState } from "react";

import { fetchJoinedWaveform, type JoinedWaveformOut } from "../../api/client";
import Plot from "../plot/PlotlyChart";
import styles from "./JoinedWaveformView.module.css";

interface Props {
  incidentId: string;
  episodeId: string;
}

// Same phase-color convention as COMTRADEExplorer.tsx's inferChannelColor,
// kept in sync deliberately so a channel reads as the same color whether
// viewed on the single-record page or this joined view.
function channelColor(canonicalName: string): string {
  const upper = canonicalName.toUpperCase();
  if (upper === "VA" || upper === "IA") return "#ef4444";
  if (upper === "VB" || upper === "IB") return "#3b82f6";
  if (upper === "VC" || upper === "IC") return "#22c55e";
  if (upper === "IN" || upper === "I0" || upper === "IE") return "#eab308";
  return "#8b5cf6";
}

const DEFAULT_CHANNELS = ["IA", "IB", "IC", "VA", "VB", "VC"];

export default function JoinedWaveformView({ incidentId, episodeId }: Props) {
  const [data, setData] = useState<JoinedWaveformOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    fetchJoinedWaveform(incidentId, episodeId)
      .then((result) => {
        if (alive) setData(result);
      })
      .catch((cause) => {
        if (alive) setError(cause instanceof Error ? cause.message : "Failed to load joined waveform.");
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [incidentId, episodeId]);

  if (loading) return <div className={styles.status}>Loading joined waveform…</div>;
  if (error) return <div className={styles.statusError}>{error}</div>;
  if (!data) return null;

  if (!data.can_join) {
    return (
      <div className={styles.statusError}>
        Cannot join this episode's records into one waveform: {data.reason ?? "unknown reason"}.
      </div>
    );
  }

  const availableChannels = new Set<string>();
  data.segments.forEach((seg) => Object.keys(seg.channels).forEach((c) => availableChannels.add(c)));
  const channelsToPlot = DEFAULT_CHANNELS.filter((c) => availableChannels.has(c));

  // One trace per (segment, channel) — deliberately NOT one trace spanning
  // both segments, so Plotly never draws a connecting line across the gap.
  // The gap itself is rendered separately below via shaded vrects.
  const traces = data.segments.flatMap((seg) =>
    channelsToPlot
      .filter((canon) => seg.channels[canon])
      .map((canon) => ({
        x: seg.channels[canon].t,
        y: seg.channels[canon].values,
        type: "scatter" as const,
        mode: "lines" as const,
        name: canon,
        legendgroup: canon,
        showlegend: seg.incident_record_id === data.segments[0].incident_record_id,
        line: { color: channelColor(canon), width: 1.2 },
        hovertemplate: `${canon}: %{y:.2f}<br>t=%{x:.3f}s<extra>${seg.source_filename ?? seg.incident_record_id}</extra>`,
      }))
  );

  const shapes = data.gap_ranges.map((gap) => ({
    type: "rect" as const,
    xref: "x" as const,
    yref: "paper" as const,
    x0: gap.start_s,
    x1: gap.end_s,
    y0: 0,
    y1: 1,
    fillcolor: gap.precision === "measured" ? "rgba(148, 163, 184, 0.25)" : "rgba(248, 113, 113, 0.25)",
    line: { width: 0 },
  }));

  const annotations = data.gap_ranges.map((gap) => ({
    x: (gap.start_s + gap.end_s) / 2,
    y: 1,
    yref: "paper" as const,
    yanchor: "bottom" as const,
    showarrow: false,
    text:
      gap.precision === "measured"
        ? `gap ${(gap.end_s - gap.start_s).toFixed(2)}s (measured)`
        : "gap unknown (not measured)",
    font: { size: 10, color: gap.precision === "measured" ? "#64748b" : "#dc2626" },
  }));

  const boundaryWarnings = data.warnings.filter((w) => w.type === "GAP_NOT_MEASURED");

  return (
    <div className={styles.wrap}>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          height: 320,
          margin: { l: 50, r: 20, t: 30, b: 40 },
          shapes,
          annotations,
          xaxis: { title: { text: "Incident-relative time (s)" }, gridcolor: "#e2e8f0" },
          yaxis: { title: { text: "Amplitude (A / kV, primary)" }, gridcolor: "#e2e8f0" },
          legend: { orientation: "h" },
          paper_bgcolor: "#ffffff",
          plot_bgcolor: "#ffffff",
        }}
        config={{ responsive: true, displaylogo: false }}
        style={{ width: "100%" }}
        useResizeHandler
      />
      <p className={styles.caption}>
        Shaded band = the actual gap between records (never interpolated — no samples exist there).
        Gray = gap duration measured from absolute record timestamps; red = gap duration not measured
        (records placed back-to-back with no implied dead-time length).
      </p>
      {boundaryWarnings.length > 0 && (
        <ul className={styles.warningList}>
          {boundaryWarnings.map((w, i) => (
            <li key={i}>{String(w.description ?? w.type)}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
