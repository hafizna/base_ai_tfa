import { useMemo, useState } from "react";

import type { DoubleEndedSingleEndedResult } from "../../../api/client";
import Plot from "../../plot/PlotlyChart";
import styles from "./FaultLocatorHistogram.module.css";

interface TwoEndedSummary {
  distanceKm: number;
  faultCurrentA: number;
  loop: string;
}

interface Props {
  histogram: number[];
  lineLenKm: number;
  twoEnded: TwoEndedSummary;
  singleEndedA: DoubleEndedSingleEndedResult | null;
  singleEndedB: DoubleEndedSingleEndedResult | null;
}

interface ReferenceLine {
  key: string;
  label: string;
  distanceKm: number;
  color: string;
  detail: string;
}

/** Vertical reference-line + checkbox panel for the two-ended answer and
 * each terminal's own single-ended reading, drawn over a window-voting
 * distance histogram (see webapp/api/routers/relay_21_de.py's
 * _compute_distance_histogram docstring for what the histogram represents
 * and why — every bar is a real Kirchhoff solution from a different
 * evaluation window across the fault's own duration, not a fabricated
 * confidence score).
 *
 * Deliberately styled as a single Plotly histogram + shapes/annotations
 * overlay (this app's own existing chart conventions, matching
 * ImpedanceLocus.tsx's zone-overlay pattern) rather than the multi-panel
 * report layout some other tools use for a similar comparison — this is
 * an independent implementation, not a copy of any other unit's report
 * design. */
export default function FaultLocatorHistogram({ histogram, lineLenKm, twoEnded, singleEndedA, singleEndedB }: Props) {
  const [showTwoEnded, setShowTwoEnded] = useState(true);
  const [showA, setShowA] = useState(true);
  const [showB, setShowB] = useState(true);

  const lines = useMemo<ReferenceLine[]>(() => {
    const result: ReferenceLine[] = [];
    if (showTwoEnded) {
      result.push({
        key: "two-ended",
        label: `Two-ended (${twoEnded.loop})`,
        distanceKm: twoEnded.distanceKm,
        color: "#0891b2",
        detail: `${twoEnded.distanceKm.toFixed(2)} km · ${(twoEnded.faultCurrentA / 1000).toFixed(2)} kA`,
      });
    }
    if (showA && singleEndedA) {
      result.push({
        key: "single-a",
        label: "Single-ended (A)",
        distanceKm: singleEndedA.distance_km,
        color: "#d946ef",
        detail: `${singleEndedA.distance_km.toFixed(2)} km · ${(singleEndedA.fault_current_a / 1000).toFixed(2)} kA`,
      });
    }
    if (showB && singleEndedB) {
      result.push({
        key: "single-b",
        label: "Single-ended (B)",
        distanceKm: singleEndedB.distance_km,
        color: "#f59e0b",
        detail: `${singleEndedB.distance_km.toFixed(2)} km · ${(singleEndedB.fault_current_a / 1000).toFixed(2)} kA`,
      });
    }
    return result;
  }, [showTwoEnded, showA, showB, twoEnded, singleEndedA, singleEndedB]);

  // Bound the bin count to the actual sample count — a 41-window sweep
  // rendered into 40+ bins would look like scattered noise rather than a
  // readable distribution.
  const binCount = Math.max(3, Math.min(20, Math.round(histogram.length / 2)));
  const binSize = lineLenKm / binCount;

  const shapes = lines.map((line) => ({
    type: "line" as const,
    xref: "x" as const,
    yref: "paper" as const,
    x0: line.distanceKm,
    x1: line.distanceKm,
    y0: 0,
    y1: 1,
    line: { color: line.color, width: 2, dash: "dash" as const },
  }));

  const annotations = lines.map((line, i) => ({
    x: line.distanceKm,
    y: 1 - i * 0.09,
    yref: "paper" as const,
    yanchor: "top" as const,
    showarrow: false,
    text: `${line.label}<br>${line.detail}`,
    font: { size: 10, color: line.color },
    bgcolor: "rgba(255,255,255,0.85)",
    bordercolor: line.color,
    borderpad: 3,
  }));

  return (
    <div className={styles.wrap}>
      <div className={styles.title}>Distance distribution (window-voting)</div>
      <p className={styles.caption}>
        Each sample is the two-ended distance solved independently at one evaluation window across the fault's
        own detected duration — a tight cluster means the answer is stable across the fault; a wide spread is
        honest evidence it isn't. Not a confidence score.
      </p>

      <div className={styles.toggleRow}>
        <label className={styles.toggleLabel}>
          <input type="checkbox" checked={showTwoEnded} onChange={(e) => setShowTwoEnded(e.target.checked)} />
          <span className={styles.swatch} style={{ background: "#0891b2" }} />
          Two-ended (K1&amp;K2)
        </label>
        <label className={styles.toggleLabel}>
          <input type="checkbox" checked={showA} onChange={(e) => setShowA(e.target.checked)} disabled={!singleEndedA} />
          <span className={styles.swatch} style={{ background: "#d946ef" }} />
          Single-ended (A)
        </label>
        <label className={styles.toggleLabel}>
          <input type="checkbox" checked={showB} onChange={(e) => setShowB(e.target.checked)} disabled={!singleEndedB} />
          <span className={styles.swatch} style={{ background: "#f59e0b" }} />
          Single-ended (B)
        </label>
      </div>

      <Plot
        data={[
          {
            x: histogram,
            type: "histogram",
            xbins: { start: 0, end: lineLenKm, size: binSize },
            marker: { color: "#94a3b8" },
            name: "Distance samples",
            hovertemplate: "%{x:.2f} km<br>%{y} windows<extra></extra>",
          },
        ]}
        layout={{
          autosize: true,
          height: 320,
          margin: { l: 50, r: 20, t: 60, b: 45 },
          xaxis: { title: { text: "Distance (km)" }, range: [0, lineLenKm], gridcolor: "#e2e8f0" },
          yaxis: { title: { text: "Windows" }, gridcolor: "#e2e8f0" },
          shapes,
          annotations,
          paper_bgcolor: "#ffffff",
          plot_bgcolor: "#ffffff",
          showlegend: false,
        }}
        config={{ responsive: true, displaylogo: false }}
        style={{ width: "100%" }}
        useResizeHandler
      />

      <p className={styles.rfNote}>
        Fault transition resistance (Rf) is not computed by this tool — the two-ended solve algebraically
        eliminates Rf as an unknown, so it is never separately recoverable from this method.
      </p>
    </div>
  );
}
