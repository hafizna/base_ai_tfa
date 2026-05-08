import { useEffect, useMemo, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";

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
}: {
  endpoint: TwsEndpoint;
  result: TwsResult;
}) {
  const amplitudeScale = useMemo(
    () =>
      Math.max(
        1,
        ...endpoint.channels.flatMap((channel) => channel.samples.map((sample) => Math.abs(sample))),
      ),
    [endpoint],
  );
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
          <span className={styles.kicker}>Qualitrol TWS FL Generated Result</span>
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

  return (
    <div className={styles.page}>
      <header className={styles.titleBar}>
        <strong>TWS Fault Locator</strong>
        <span>{data?.source_file ?? "Cashel .cdb export"}</span>
        <div className={styles.titleActions}>
          <Link to="/" className={styles.titleButton}>← Back to Home</Link>
          <button type="button" className={styles.titleButton} onClick={() => window.print()}>
            🖨 Print
          </button>
        </div>
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
              <WaveformPane endpoint={xEndpoint} result={result} />
              <WaveformPane endpoint={yEndpoint} result={result} />
            </div>
          </details>
          <SelComparisonPanel result={result} />
          <MethodologyPanel result={result} xEndpoint={xEndpoint} yEndpoint={yEndpoint} />
        </main>
      )}
    </div>
  );
}

function MethodologyPanel({
  result,
  xEndpoint,
  yEndpoint,
}: {
  result: TwsResult;
  xEndpoint: TwsEndpoint;
  yEndpoint: TwsEndpoint;
}) {
  const xName = xEndpoint.station_display_name || xEndpoint.station_name || "X";
  const yName = yEndpoint.station_display_name || yEndpoint.station_name || "Y";
  const xKm = xEndpoint.fault_distance_km;
  const yKm = yEndpoint.fault_distance_km;
  const lineKm = result.line_length_km;
  const velocityFactor = result.velocity_factor;

  return (
    <section className={styles.docPanel}>
      <header className={styles.docHeader}>
        <h3>Cara Membaca Grafik & Menentukan Lokasi Gangguan</h3>
        <span className={styles.docSubtitle}>
          Penjelasan metodologi Travelling Wave (TW) Fault Locator — wajib dikonfirmasi dengan bukti lapangan
        </span>
      </header>

      <div className={styles.docGrid}>
        <article className={styles.docCard}>
          <h4>1. Apa yang ditampilkan grafik</h4>
          <p>
            Saat terjadi gangguan, surja tegangan/arus (<em>travelling wave</em>) merambat dari titik gangguan ke kedua
            ujung saluran dengan kecepatan mendekati cahaya. Kedua relay TWS di {xName} (X) dan {yName} (Y) merekam
            kedatangan gelombang tersebut. Sumbu X grafik adalah <strong>jarak relatif</strong> terhadap penanda
            kedatangan gelombang pertama (titik <code>M</code>, di posisi 0 km), dihitung dari{" "}
            <code>(waktu sampel − waktu marker) × kecepatan rambat</code>. Pantulan dari ujung saluran atau titik
            gangguan akan muncul sebagai puncak pada sumbu X tersebut. Tiga jejak warna adalah fasa A (merah), B
            (hijau), dan C (biru).
          </p>
        </article>

        <article className={styles.docCard}>
          <h4>2. Bagaimana jarak gangguan dihitung</h4>
          <p>
            Metode utama yang dipakai adalah <strong>Type D (dua ujung)</strong> mengikuti referensi Schweitzer et al.
            (IEEE, 2014):
          </p>
          <pre className={styles.docFormula}>m = ½ × (ℓ + (t_X − t_Y) × v)</pre>
          <ul>
            <li>
              <strong>ℓ</strong> = panjang saluran ({formatKm(lineKm, 2)} km) — diambil dari <code>CIRCUIT2.XML</code>{" "}
              pada file CDB.
            </li>
            <li>
              <strong>v</strong> = kecepatan rambat = <code>c × velocity_factor</code> (
              {velocityFactor.toFixed(2)}% × 299.792 km/s).
            </li>
            <li>
              <strong>t_X, t_Y</strong> = waktu kedatangan gelombang di masing-masing ujung, sudah disinkronkan via
              GPS.
            </li>
          </ul>
          <p>
            Hasil: gangguan terjadi <strong>{formatKm(xKm, 2)} km dari {xName}</strong> atau{" "}
            <strong>{formatKm(yKm, 2)} km dari {yName}</strong>. Nilai ini juga di-cross-check dengan nilai DTFX/DTFY
            yang sudah dihitung perangkat Qualitrol (lihat tabel perbandingan di atas).
          </p>
        </article>

        <article className={styles.docCard}>
          <h4>3. Akurasi & sumber kesalahan</h4>
          <ul>
            <li>
              Akurasi inheren TW sekitar <strong>± 0,2 µs</strong> ≈ 60 meter (≈ satu span tower).
            </li>
            <li>
              Faktor pembatas akurasi: ketidakseragaman <em>line sag</em>, perubahan ketinggian medan, beda struktur
              tower antar seksi, dan dispersi gelombang.
            </li>
            <li>
              Untuk saluran &gt; 100 km, beda 0,5–1 km antara hasil TW dan posisi fisik adalah hal wajar dan tidak
              menggugurkan validitas hasil.
            </li>
          </ul>
        </article>

        <article className={`${styles.docCard} ${styles.docCardConfirm}`}>
          <h4>4. Konfirmasi Lapangan (WAJIB DIISI)</h4>
          <p>
            Hasil TW di atas adalah <strong>indikasi awal</strong>. Tim patroli wajib mengisi temuan lapangan agar
            arsip ini berstatus <em>terverifikasi</em>:
          </p>
          <ul className={styles.docChecklist}>
            <li>
              <span>Nomor tower terdampak (mis. T-145):</span>
              <code className={styles.docFillBlank}>__________</code>
            </li>
            <li>
              <span>Span yang terdampak (antara T-… s/d T-…):</span>
              <code className={styles.docFillBlank}>__________ s/d __________</code>
            </li>
            <li>
              <span>Jarak terukur dari tower {xName} (km):</span>
              <code className={styles.docFillBlank}>__________ km</code>
            </li>
            <li>
              <span>Selisih vs hasil TWS ({formatKm(xKm, 2)} km):</span>
              <code className={styles.docFillBlank}>__________ m</code>
            </li>
            <li>
              <span>Jenis kerusakan (insulator/konduktor/aksesoris/lainnya):</span>
              <code className={styles.docFillBlank}>__________</code>
            </li>
            <li>
              <span>Penyebab dugaan (petir / pohon / hewan / vandalisme / lainnya):</span>
              <code className={styles.docFillBlank}>__________</code>
            </li>
            <li>
              <span>Foto / berita acara patroli (lampiran):</span>
              <code className={styles.docFillBlank}>__________</code>
            </li>
            <li>
              <span>Petugas patroli & tanggal verifikasi:</span>
              <code className={styles.docFillBlank}>__________</code>
            </li>
          </ul>
          <p className={styles.docNote}>
            Catatan: bukti lapangan akan dipakai untuk kalibrasi ulang <code>velocity_factor</code> dan koreksi panjang
            saluran (line sag) agar prediksi TWS berikutnya makin akurat.
          </p>
        </article>
      </div>
    </section>
  );
}

function SelComparisonPanel({ result }: { result: TwsResult }) {
  const sel = result.sel_type_d;
  if (!sel) {
    return (
      <section className={styles.selPanel}>
        <h3>TFA Calculation vs Qualitrol</h3>
        <p className={styles.selEmpty}>
          Not enough metadata in this export to recompute the two-end fault location (need both X and Y arrival
          times, line length, and velocity factor).
        </p>
      </section>
    );
  }

  const v_kms = sel.velocity_km_s;
  return (
    <section className={styles.selPanel}>
      <header className={styles.selHeader}>
        <h3>TFA Calculation vs Qualitrol</h3>
        <span>
          m = ½(ℓ + (t_X − t_Y)·v) &nbsp;|&nbsp; ℓ = {formatKm(sel.line_length_km, 2)} km, v ={" "}
          {(v_kms / 1000).toFixed(2)} ×10³ km/s, Δt = {sel.delta_t_us.toFixed(2)} µs
        </span>
      </header>
      <table className={styles.selTable}>
        <thead>
          <tr>
            <th>Terminal</th>
            <th>TFA Calculation</th>
            <th>Qualitrol DTF</th>
            <th>Δ (ours − Qualitrol)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>X</td>
            <td>{formatKm(sel.m_from_x_km, 3)} km</td>
            <td>{formatKm(sel.qualitrol_x_km, 3)} km</td>
            <td className={styles.selDelta}>{(sel.delta_x_km * 1000).toFixed(0)} m</td>
          </tr>
          <tr>
            <td>Y</td>
            <td>{formatKm(sel.m_from_y_km, 3)} km</td>
            <td>{formatKm(sel.qualitrol_y_km, 3)} km</td>
            <td className={styles.selDelta}>{(sel.delta_y_km * 1000).toFixed(0)} m</td>
          </tr>
        </tbody>
      </table>
    </section>
  );
}
