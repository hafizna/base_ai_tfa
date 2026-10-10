import axios from "axios";
import type { ComtradeData } from "../context/AnalysisContext";

const BASE = import.meta.env.VITE_API_URL?.trim() || "";

export const api = axios.create({
  baseURL: BASE,
  timeout: 30000,
});

export interface UploadedAnalysis {
  analysis_id: string;
  station_name: string;
  rec_dev_id: string;
  total_samples: number;
  analog_channel_count: number;
  status_channel_count: number;
  suggested_relay_type?: string | null;
  detection_confidence?: number | null;
}

export interface TwsUploadedAnalysis {
  analysis_id: string;
  source_file: string;
  circuit_name: string;
  line_length_km: number;
  endpoint_count: number;
  total_samples: number;
  warnings: string[];
}

export interface TwsChannel {
  phase: string;
  name: string;
  samples: number[];
  min: number;
  max: number;
}

export interface TwsEndpoint {
  role: "X" | "Y" | "Z";
  index_id: number;
  station_name: string;
  station_display_name: string;
  device_name: string;
  device_display_name: string;
  feeder_name: string;
  feeder_display_name: string;
  record_number: number;
  record_file_name: string;
  trigger_time: string;
  trigger_time_us: number;
  event_time_us: number;
  event_time_local: number;
  gps_time_tag: string;
  corrected_gps: string;
  gps_locked: boolean;
  sample_rate_hz: number;
  total_samples: number;
  decimation: number;
  trigger_phase: string;
  software_trigger_phase: string;
  software_trigger_point: number;
  trigger_delay: number;
  fault_distance_km: number;
  channels: TwsChannel[];
}

export interface TwsSelTypeD {
  tx_seconds: number;
  ty_seconds: number;
  delta_t_us: number;
  velocity_km_s: number;
  line_length_km: number;
  m_from_x_km: number;
  m_from_y_km: number;
  qualitrol_x_km: number;
  qualitrol_y_km: number;
  delta_x_km: number;
  delta_y_km: number;
}

export interface TwsResult {
  result_id: number;
  result_time_us: number;
  result_time_local: number;
  circuit_id: number;
  circuit_name: string;
  segment_id: number;
  segment_name: string;
  line_length_km: number;
  velocity_factor: number;
  velocity_km_s?: number;
  sample_distance_km: number;
  distance_from_segment_end_a: number;
  is_component_fault: boolean;
  endpoints: TwsEndpoint[];
  sel_type_d?: TwsSelTypeD | null;
}

export interface TwsCdbData {
  source_type: "tws_cdb";
  source_file: string;
  station_name: string;
  rec_dev_id: string;
  total_samples: number;
  results: TwsResult[];
  warnings: string[];
}

export interface AnalysisChannelSummary {
  id: string;
  name: string;
  canonical_name: string;
  unit: string;
  phase: string | null;
  measurement: string;
  ct_primary: number;
  ct_secondary: number;
  pors: string;
}

export interface StatusChannelSummary {
  id: string;
  name: string;
  sample_count: number;
  on_count: number;
  transition_count: number;
}

export interface AnalysisSummary {
  analysis_id: string;
  station_name: string;
  rec_dev_id: string;
  rev_year: string;
  sampling_rates: [number, number][];
  trigger_time: number;
  total_samples: number;
  frequency: number;
  duration_ms: number;
  analog_channels: AnalysisChannelSummary[];
  status_channels: StatusChannelSummary[];
  warnings: string[];
}

export async function uploadComtrade(files: File[]) {
  const form = new FormData();
  files.forEach((file) => {
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (ext === "cff") form.append("cff_file", file);
    else if (ext === "cfg") form.append("cfg_file", file);
    else if (ext === "dat") form.append("dat_file", file);
    else form.append("files", file);
  });
  const { data } = await api.post<UploadedAnalysis>("/api/upload", form);
  return data;
}

export async function uploadTwsCdb(cdb: File) {
  const form = new FormData();
  form.append("cdb_file", cdb);
  const { data } = await api.post<TwsUploadedAnalysis>("/api/tws/upload-cdb", form);
  return data;
}

export async function fetchTwsAnalysis(analysisId: string) {
  const { data } = await api.get<TwsCdbData>(`/api/tws/${analysisId}`);
  return data;
}

export async function fetchAnalysisSummary(analysisId: string) {
  const { data } = await api.get<AnalysisSummary>(`/api/analysis/${analysisId}/summary`);
  return data;
}

export async function fetchAnalysis(analysisId: string) {
  const { data } = await api.get<ComtradeData>(`/api/analysis/${analysisId}`);
  return data;
}

export interface CanonicalEventWindow {
  record_start_ms: number;
  trigger_time_ms: number | null;
  inception_idx: number | null;
  inception_time_ms: number | null;
  clearing_idx: number | null;
  clearing_time_ms: number | null;
  fault_duration_ms: number | null;
  method: string;
  timing_source: string;
  confidence: number;
  faulted_phases: string[];
  reclose_events: Array<{
    time: number;
    success: boolean | null;
    cb_open_verified?: boolean;
    dead_time_ms?: number | null;
    source?: string;
  }>;
  warnings: string[];
}

/** Per-phase values keyed "A" | "B" | "C" (null where the window had no data). */
export type PhaseValues = Partial<Record<"A" | "B" | "C", number | null>>;

/** Measured magnitudes around a record's event (webapp/api/record_facts.py). */
export interface ElectricalMeasurements {
  current_unit?: string | null;
  voltage_unit?: string | null;
  prefault?: { current_rms: PhaseValues; voltage_rms: PhaseValues };
  fault?: { current_peak: PhaseValues; current_rms_max: PhaseValues; voltage_rms_min: PhaseValues };
  after_clearing?: { current_rms: PhaseValues; voltage_rms: PhaseValues };
  after_reclose?: { current_rms: PhaseValues; voltage_rms: PhaseValues };
}

/** A status channel that asserted, with edges in ms on the record's time axis. */
export interface ProtectionOperation {
  name: string;
  role: "trip" | "zone" | "teleprotection" | "breaker" | "reclose" | "protection";
  zone: number | null;
  phase: "A" | "B" | "C" | "3P" | null;
  initially_on: boolean;
  on_ms: number[];
  off_ms: number[];
}

/** A change in one phase's analog signals (core/analog_trace.py), in ms on the record's time axis. */
export interface AnalogTraceEvent {
  t_ms: number;
  phase: "A" | "B" | "C";
  kind:
    | "disturbance_start"
    | "energization_start"
    | "current_ceased"
    | "current_back_to_load"
    | "current_return"
    | "current_rise"
    | "voltage_lost"
    | "voltage_return";
  value: number | null;
}

/** A stretch where one phase's current and voltage stay in one class. */
export interface AnalogTraceSpan {
  current: "none" | "load" | "high";
  voltage: "dead" | "sag" | "normal" | null;
  start_ms: number;
  end_ms: number;
  current_rms: number;
  voltage_pu: number | null;
}

export interface AnalogTraceSummary {
  starts_dead: boolean;
  fault_start_ms: number | null;
  /** The last current zero of the fault's phases. */
  fault_clearing_ms: number | null;
  /** Fault clearing time: fault start to the last current zero (Grid Code CCA1 2.2). */
  fct_ms: number | null;
  high_current_phases: string[];
  sagged_phases: string[];
  ceased: Record<string, number>;
  zero_current_pattern: "three_together" | "single_phase" | "two_phase" | "staggered" | null;
  zero_spans: Array<{ phase: string; start_ms: number; end_ms: number | null }>;
  current_returns: Record<string, number>;
  voltage_returns: Record<string, number>;
  dead_time_ms: Record<string, number>;
  energization_ms: number | null;
  refault_ms: number | null;
  disturbances: Array<Record<string, unknown>>;
}

/** What each phase's current and voltage did through a record (core/analog_trace.py). */
export interface AnalogTrace {
  cycle_ms: number;
  reference: Record<string, unknown>;
  phases: Record<string, AnalogTraceSpan[]>;
  events: AnalogTraceEvent[];
  summary: AnalogTraceSummary;
  warnings: string[];
}

export interface CanonicalRecordAnalysis {
  record_id: string;
  source_metadata: Record<string, unknown>;
  data_quality: Record<string, unknown>;
  event_window: CanonicalEventWindow | null;
  fault_episodes: Array<Record<string, unknown>>;
  // Empty in snapshots taken before these facts existed.
  protection_operations: ProtectionOperation[];
  electrical_measurements: ElectricalMeasurements;
  analog_trace?: Partial<AnalogTrace>;
  /** The record's reasoning chain (webapp/api/fault_reasoning.py); absent in older snapshots. */
  reasoning?: RecordReasoning;
  observed_facts: Record<string, unknown>;
  protection_interpretation: Record<string, unknown>;
  cause_hypotheses: Array<Record<string, unknown>>;
  missing_evidence: Array<{ type: string; description: string }>;
  provenance: Record<string, unknown>;
}

/**
 * One conclusion of the reasoning chain: a ledger row. Rule IDs follow
 * docs/fault-reasoning-rules.md. Confidence "flag" marks a row for review;
 * "ai" marks the AI's statistical reading.
 */
export interface ReasoningRow {
  key: string;
  step: number;
  label: string;
  title: string;
  evidence: string[];
  rules: string[];
  confidence: "high" | "medium" | "low" | "flag" | "ai";
  value: Record<string, unknown>;
  conflicts: string[];
}

export interface SignalEvent {
  t_ms: number;
  channel: string;
  role: string;
  change: string;
  /** Contact bounce, shown but not part of the sequence. */
  muted?: boolean;
}

export interface RecordReasoning {
  schema: string;
  has_fault: boolean;
  fault_start_ms: number | null;
  conclusions: ReasoningRow[];
  flag_count: number;
  conflict_count: number;
  signals: {
    /** "fault_start": times are ms after the fault starts; "record_start": after the first sample. */
    reference: "fault_start" | "record_start";
    events: SignalEvent[];
    silent: Array<{ channel: string; role: string | null }>;
    channel_count: number;
  };
}

/** A fault's ledger in a reconstructed incident (episode.interpretation.reasoning). */
export interface EpisodeReasoning {
  fault_record_id: string;
  fault_start_ms: number | null;
  rows: ReasoningRow[];
  flag_count: number;
  conflict_count: number;
}

export async function fetchCanonicalAnalysis(analysisId: string) {
  const { data } = await api.get<CanonicalRecordAnalysis>(`/api/analysis/${analysisId}/canonical`);
  return data;
}

export async function recalculateRatio(analysisId: string, ratios: unknown[]) {
  const { data } = await api.post("/api/recalculate-ratio", { analysis_id: analysisId, ratios });
  return data;
}

export async function computeLocus(
  analysisId: string,
  zones: unknown[],
  loop: string,
  k0 = 0.0,
  k0AngleDeg = 0.0,
  invertI = false,
  ctRatioOverride?: number,
  vtRatioOverride?: number,
) {
  const { data } = await api.post("/api/analyze/21/locus", {
    analysis_id: analysisId,
    zones,
    loop,
    k0,
    k0_angle_deg: k0AngleDeg,
    invert_i: invertI,
    ...(ctRatioOverride != null ? { ct_ratio_override: ctRatioOverride } : {}),
    ...(vtRatioOverride != null ? { vt_ratio_override: vtRatioOverride } : {}),
  });
  return data;
}

export async function computeLocusBatch(
  analysisId: string,
  loops: string[],
  k0 = 0.0,
  k0AngleDeg = 0.0,
  invertI = false,
  ctRatioOverride?: number,
  vtRatioOverride?: number,
) {
  const { data } = await api.post(
    "/api/analyze/21/locus-batch",
    {
      analysis_id: analysisId,
      loops,
      k0,
      k0_angle_deg: k0AngleDeg,
      invert_i: invertI,
      ...(ctRatioOverride != null ? { ct_ratio_override: ctRatioOverride } : {}),
      ...(vtRatioOverride != null ? { vt_ratio_override: vtRatioOverride } : {}),
    },
    // One request does the work of six; allow it more time than the 30s default.
    { timeout: 90000 },
  );
  return data as {
    points_by_loop: Record<string, { t: number; r: number; x: number }[]>;
    diagnostics_by_loop?: Record<string, {
      windows_evaluated: number;
      windows_kept: number;
      retention_pct: number;
      ct_saturation_suspected: boolean;
      ct_saturation_thd_ratio: number;
      dc_offset_corrected: boolean;
    }>;
  };
}

// --- Double Ended FL (two-terminal Kirchhoff fault location) ---

export interface DoubleEndedAlignEstimate {
  inception_time_a_s: number | null;
  inception_time_b_s: number | null;
  timing_source_a: string;
  timing_source_b: string;
  estimated_shift_ms: number | null;
  estimate_available: boolean;
  estimate_reason: string;
  clock_shift_ms?: number | null;
  clock_offset_hours?: number | null;
  line_a?: string | null;
  line_b?: string | null;
  sync_phase?: string | null;
  sync_channel_a?: string | null;
  sync_channel_b?: string | null;
}

export async function fetchDoubleEndedAlignEstimate(analysisIdA: string, analysisIdB: string) {
  const { data } = await api.post<DoubleEndedAlignEstimate>("/api/analyze/21de/align-estimate", {
    analysis_id_a: analysisIdA,
    analysis_id_b: analysisIdB,
  });
  return data;
}

/** The loop each terminal's reasoning chain points to (rule F7.4). */
export interface DoubleEndedLoopSuggestion {
  /** Null when no terminal found the faulted phases. */
  loop: string | null;
  /** PLN phase label, e.g. "S-T". */
  label: string | null;
  source_terminal: "A" | "B" | null;
  agree: boolean | null;
  weak_infeed_terminals: string[];
  reason: string;
}

export async function fetchDoubleEndedLoopSuggestion(analysisIdA: string, analysisIdB: string) {
  const { data } = await api.post<DoubleEndedLoopSuggestion>("/api/analyze/21de/loop-suggestion", {
    analysis_id_a: analysisIdA,
    analysis_id_b: analysisIdB,
  });
  return data;
}

export interface DoubleEndedSuggestShiftRequest {
  analysisIdA: string;
  analysisIdB: string;
  loop: string;
  lineLenKm: number;
  r1OhmPerKm: number;
  x1OhmPerKm: number;
  invertIA?: boolean;
  invertIB?: boolean;
  invertPhaseSequenceA?: boolean;
  invertPhaseSequenceB?: boolean;
}

export interface DoubleEndedSuggestShiftResult {
  shift_ms: number | null;
  residual: number | null;
  searched_range_ms: number;
  reason: string;
}

// Grounded in the same Kirchhoff residual the final calculation itself
// reports (unlike align-estimate, which extrapolates from possibly-wrong
// wall-clock timestamps) — but still only a SUGGESTION to confirm on the
// sync overlay plot, never auto-applied. See relay_21_de.py's
// _find_optimal_shift docstring for why (multiple local minima can exist).
export async function fetchDoubleEndedSuggestShift(req: DoubleEndedSuggestShiftRequest) {
  const { data } = await api.post<DoubleEndedSuggestShiftResult>("/api/analyze/21de/suggest-shift", {
    analysis_id_a: req.analysisIdA,
    analysis_id_b: req.analysisIdB,
    loop: req.loop,
    line_len_km: req.lineLenKm,
    r1_ohm_per_km: req.r1OhmPerKm,
    x1_ohm_per_km: req.x1OhmPerKm,
    invert_i_a: req.invertIA ?? false,
    invert_i_b: req.invertIB ?? false,
    invert_phase_sequence_a: req.invertPhaseSequenceA ?? false,
    invert_phase_sequence_b: req.invertPhaseSequenceB ?? false,
  });
  return data;
}

export interface DoubleEndedComputeRequest {
  analysisIdA: string;
  analysisIdB: string;
  loop: string;
  lineLenKm: number;
  r1OhmPerKm: number;
  x1OhmPerKm: number;
  manualShiftMs: number;
  invertIA?: boolean;
  invertIB?: boolean;
  invertPhaseSequenceA?: boolean;
  invertPhaseSequenceB?: boolean;
}

export interface DoubleEndedSingleEndedResult {
  terminal: "A" | "B";
  distance_km: number;
  distance_pct: number;
  fault_current_a: number;
  r_measured_ohm: number;
  x_measured_ohm: number;
  warnings: string[];
}

export interface DoubleEndedComputeResult {
  loop: string;
  distance_km: number;
  distance_pct: number;
  fault_current_a: number;
  m_residual_imag: number;
  kvl_residual: number;
  distance_spread_km: number;
  selected_window_count: number;
  calculation_basis: "negative_sequence" | "phase_loop";
  inception_time_a_s: number;
  inception_time_b_s: number;
  active_tag_a: string | null;
  active_tag_b: string | null;
  /** F7.4: |I_A + I_B| in the solved quantity over the positive-sequence current drawn at the fault. */
  fault_point_ratio?: number | null;
  /** Per end: fault contribution over load current; under 2× is weak infeed. */
  fault_contribution_ratio_a?: number | null;
  fault_contribution_ratio_b?: number | null;
  /** False when the fault draws no current in the solved quantity: the result is not valid. */
  loop_carries_fault?: boolean | null;
  weak_infeed_terminals?: string[];
  warnings: string[];
  single_ended_a: DoubleEndedSingleEndedResult | null;
  single_ended_b: DoubleEndedSingleEndedResult | null;
  distance_histogram_km: number[];
}

export async function computeDoubleEndedFL(req: DoubleEndedComputeRequest) {
  const { data } = await api.post<DoubleEndedComputeResult>("/api/analyze/21de/compute", {
    analysis_id_a: req.analysisIdA,
    analysis_id_b: req.analysisIdB,
    loop: req.loop,
    line_len_km: req.lineLenKm,
    r1_ohm_per_km: req.r1OhmPerKm,
    x1_ohm_per_km: req.x1OhmPerKm,
    manual_shift_ms: req.manualShiftMs,
    invert_i_a: req.invertIA ?? false,
    invert_i_b: req.invertIB ?? false,
    invert_phase_sequence_a: req.invertPhaseSequenceA ?? false,
    invert_phase_sequence_b: req.invertPhaseSequenceB ?? false,
  });
  return data;
}

export async function fetchFaultClassification21(analysisId: string) {
  const { data } = await api.get(`/api/analyze/21/fault-classification?analysis_id=${analysisId}`);
  return data as {
    fault_code: string;
    phases: string[];
    phases_label: string;
    to_ground: boolean;
    trip_type: string | null;
    zone: string | null;
    prefault_ms: number;
    fault_ms: number;
    total_ms: number;
    ar_status: "successful" | "failed" | null;
    no_fault?: boolean;
    timing_source?: string;
    timing_confidence?: number;
  };
}

export async function fetchElectricalParams21(analysisId: string) {
  const { data } = await api.get(`/api/analyze/21/electrical-params?analysis_id=${analysisId}`);
  return data as {
    fault_duration_ms?: number;
    inception_time_ms?: number;
    trip_time_ms?: number;
    trip_time_source?: "soe" | "status_edge" | "estimated";
    i_peak_ia_a?: number;
    i_peak_ib_a?: number;
    i_peak_ic_a?: number;
    v_sag_pct?: number;
    i_pos_seq_a?: number;
    i_neg_seq_a?: number;
    i_zero_seq_a?: number;
    z_at_inception_ohm?: number;
    z_min_ohm?: number;
    r_at_fault_ohm?: number;
    x_at_fault_ohm?: number;
    rx_ratio?: number;
    z_angle_deg?: number;
    ar_dead_time_ms?: number;
    no_fault?: boolean;
    no_fault_reasons?: string[];
    peak_to_prefault_ratio?: number;
    voltage_sag_pu?: number;
    i0_i1_ratio?: number;
    i2_i1_ratio?: number;
    timing_source?: string;
    timing_confidence?: number;
  };
}

export type LocusEventCategory = "trip" | "zone" | "reclose" | "breaker" | "comms" | "other";

export interface LocusEvent {
  time_ms: number;
  rel_ms: number;
  channel: string;
  state: number;
  category: LocusEventCategory;
  label: string;
}

export async function fetchLocusEvents21(analysisId: string) {
  const { data } = await api.get(`/api/analyze/21/locus-events?analysis_id=${analysisId}`);
  return data as {
    inception_time_ms: number | null;
    events: LocusEvent[];
  };
}

export async function fetchFullSoe21(analysisId: string) {
  const { data } = await api.get(`/api/analyze/21/full-soe?analysis_id=${analysisId}`);
  return data as {
    inception_time_ms: number | null;
    events: LocusEvent[];
  };
}

export async function extractFeatures21(analysisId: string) {
  const { data } = await api.get(`/api/analyze/21/extract-features?analysis_id=${analysisId}`);
  return data as {
    fault_inception_angle_deg: number;
    fault_duration_ms: number;
    prefault_load_a: number;
    impedance_at_trip_ohm: number;
    waveform_asymmetry: number;
    dc_offset: number;
    ar_result: "successful" | "failed" | null;
  };
}

export async function aiFaultAnalysis21(analysisId: string, features: unknown) {
  const { data } = await api.post("/api/analyze/21/ai-analysis", { analysis_id: analysisId, ...(features as object) });
  return data;
}

export async function diffRestraint87L(analysisId: string, params: unknown) {
  const { data } = await api.post("/api/analyze/87l/diff-restraint", { analysis_id: analysisId, params, relay_type: "87L" });
  return data;
}

export async function aiFaultAnalysis87L(analysisId: string, params: unknown) {
  const { data } = await api.post("/api/analyze/87l/ai-analysis", { analysis_id: analysisId, params, relay_type: "87L" });
  return data;
}

/** F5.8: whether a record carries line-differential (87L) evidence. */
export interface LineDiffEvidence {
  has_87l: boolean;
  diff_data_mode: "TWO_TERMINAL" | "TWO_TERMINAL_RAW" | "LOCAL_ONLY";
  /** Differential operate channels recorded, and the ones that asserted. */
  operate_channels: string[];
  operated_channels: string[];
  evidence: string[];
}

export async function fetchLineDiffEvidence(analysisId: string) {
  const { data } = await api.post<LineDiffEvidence>("/api/analyze/87l/evidence", { analysis_id: analysisId });
  return data;
}

export async function diffRestraint87T(analysisId: string, params: unknown) {
  const { data } = await api.post("/api/analyze/87t/diff-restraint", { analysis_id: analysisId, params, relay_type: "87T" });
  return data;
}

export async function overCurrentCharacteristic(analysisId: string, curve_type: string, is_pickup_a: number, tms: number) {
  const { data } = await api.post("/api/analyze/ocr/characteristic", { analysis_id: analysisId, curve_type, is_pickup_a, tms });
  return data;
}

export interface TccStage {
  label: string;
  curve_type: string;       // NI | VI | EI | LTI | DT
  is_pickup_a: number;
  tms: number;
  definite_time_s: number;
}
export interface TccCurveLine {
  label: string;
  curve_type: string;
  is_pickup_a: number;
  currents_a: number[];
  trip_times_s: number[];
}
export interface TccFaultPoint {
  channel_label: string;
  current_a: number;
  winning_stage_label: string | null;
  winning_curve_type: string | null;
  trip_time_s: number | null;
  multiple_of_pickup: number | null;
  is_moment: boolean;
}
export interface TccResult {
  mode: string;
  domain: string;
  curves: TccCurveLine[];
  fault_points: TccFaultPoint[];
  assessment: string;
}
export async function tccMultiStage(
  analysisId: string,
  mode: "phase" | "ef",
  domain: "line" | "trafo",
  stages: TccStage[],
): Promise<TccResult> {
  const { data } = await api.post("/api/analyze/ocr/tcc", { analysis_id: analysisId, mode, domain, stages });
  return data;
}

export interface ReportChart {
  id: string;
  title: string;
  image_b64: string;
}

export interface ReportSoeEvent {
  time_ms: number;
  rel_ms?: number | null;
  channel: string;
  state: number;
  category?: string | null;
  label?: string | null;
}

export interface ReportRequest {
  relay_type: string;
  ai_analysis?: Record<string, unknown> | null;
  charts: ReportChart[];
  soe_events?: ReportSoeEvent[];
  relay_settings?: Record<string, unknown> | null;
  // Opt-in: render the ABB-style binary timing diagram. Off by default since
  // digital channels are already summarized in the SOE table.
  include_binary_diagram?: boolean;
}

export async function generateReport(analysisId: string, body: ReportRequest): Promise<Blob> {
  const { data } = await api.post(`/api/report/${analysisId}`, body, {
    responseType: "blob",
    timeout: 60000,
  });
  return data;
}

/** The DE-FL location report: the server recomputes the result from these
 * inputs; the page adds what only it knows. */
export interface DoubleEndedReportRequest {
  analysisIdA: string;
  analysisIdB: string;
  loop: string;
  lineLenKm: number;
  r1OhmPerKm: number;
  x1OhmPerKm: number;
  manualShiftMs: number;
  invertIA?: boolean;
  invertIB?: boolean;
  invertPhaseSequenceA?: boolean;
  invertPhaseSequenceB?: boolean;
  recordNameA?: string | null;
  recordNameB?: string | null;
  shiftSource: "estimate" | "residual_search" | "manual";
  syncConfirmed: boolean;
  suggestedLoop?: string | null;
  charts: ReportChart[];
}

export async function generateDoubleEndedReport(req: DoubleEndedReportRequest): Promise<Blob> {
  const { data } = await api.post(
    "/api/location-report/de-fl",
    {
      analysis_id_a: req.analysisIdA,
      analysis_id_b: req.analysisIdB,
      loop: req.loop,
      line_len_km: req.lineLenKm,
      r1_ohm_per_km: req.r1OhmPerKm,
      x1_ohm_per_km: req.x1OhmPerKm,
      manual_shift_ms: req.manualShiftMs,
      invert_i_a: req.invertIA ?? false,
      invert_i_b: req.invertIB ?? false,
      invert_phase_sequence_a: req.invertPhaseSequenceA ?? false,
      invert_phase_sequence_b: req.invertPhaseSequenceB ?? false,
      record_name_a: req.recordNameA ?? null,
      record_name_b: req.recordNameB ?? null,
      shift_source: req.shiftSource,
      sync_confirmed: req.syncConfirmed,
      suggested_loop: req.suggestedLoop ?? null,
      charts: req.charts,
    },
    { responseType: "blob", timeout: 120000 },
  );
  return data;
}

export type GroundTruthSource =
  | "RELAY_EVENT_REPORT"
  | "OPERATOR_SOE"
  | "REMOTE_END_COMTRADE"
  | "DFR_RECORD"
  | "FIELD_INSPECTION"
  | "LIGHTNING_DETECTION"
  | "PATROL_REPORT"
  | "PROTECTION_ENGINEER_REVIEW"
  | "UNCONFIRMED_ASSUMPTION"
  | "OTHER";

export type GroundTruthConfidence = "CONFIRMED" | "PROBABLE" | "POSSIBLE" | "UNKNOWN";

export interface TrainingFeedbackRequest {
  analysis_id: string;
  relay_type: string;
  // Legacy (Stage -1) fields — kept for backward compatibility.
  ai_correct: boolean | null;
  actual_label: string;
  fault_type: string;
  include_for_training: boolean;
  operator?: string;
  notes?: string;
  ai_prediction?: Record<string, unknown> | null;
  // Stage 0: per-layer ground-truth correction fields.
  parsing_correct?: boolean | null;
  channel_mapping_correct?: boolean | null;
  inception_correct?: boolean | null;
  corrected_inception_time_ms?: number | null;
  clearing_correct?: boolean | null;
  corrected_clearing_time_ms?: number | null;
  faulted_phases_correct?: boolean | null;
  actual_faulted_phases?: string[];
  fault_type_correct?: boolean | null;
  actual_fault_type?: string;
  zone_correct?: boolean | null;
  actual_zone?: string;
  trip_type_correct?: boolean | null;
  actual_trip_type?: string;
  reclose_correct?: boolean | null;
  actual_reclose_outcome?: string;
  event_segmentation_correct?: boolean | null;
  actual_episode_count?: number | null;
  protection_interpretation_correct?: boolean | null;
  actual_event_class?: string;
  cause_correct?: boolean | null;
  actual_cause?: string;
  ground_truth_source?: GroundTruthSource[];
  ground_truth_confidence?: GroundTruthConfidence;
}

export interface TrainingStatus {
  enabled: boolean;
  admin_token_configured: boolean;
  data_dir: string;
  raw_record_count: number;
  feedback_count: number;
  total_bytes: number;
}

export interface EventSimulatorScenarioSummary {
  id: string;
  title: string;
  subtitle: string;
  station_name: string;
  asset_name: string;
  event_count: number;
}

export interface EventSimulatorNotification {
  id: string;
  incident_id: string;
  tier: number;
  cluster: string;
  label: string;
  color: string;
  title: string;
  message: string;
  first_event_ms: number;
  last_event_ms: number;
  emit_ms: number;
  event_ids: string[];
  timing_note: string;
}

export interface EventSimulatorIncident {
  id: string;
  station_name: string;
  asset_name: string;
  asset_id: string;
  start_ms: number;
  last_event_ms: number;
  status: string;
  primary_tier: number | null;
  title: string;
  summary: string;
  event_count: number;
  events: Record<string, unknown>[];
  measurements: Record<string, unknown>[];
  artifacts: Record<string, unknown>[];
  notifications: EventSimulatorNotification[];
  relay_functions: string[];
  cb_states: Record<string, unknown>[];
  reclose_sequence: Record<string, unknown>[];
}

export interface EventSimulatorTraceStep {
  step: number;
  t_ms: number;
  raw_event: Record<string, unknown> | null;
  mapping: Record<string, unknown> | null;
  classification: Record<string, unknown> | null;
  decision: string;
  emitted_before_this_event: EventSimulatorNotification[];
  emitted_from_this_event: EventSimulatorNotification[];
  incident_snapshot: Record<string, unknown> | null;
}

export interface EventSimulatorRun {
  scenario: {
    id: string;
    title: string;
    subtitle: string;
    description: string;
    station_name: string;
    asset_name: string;
    asset_id: string;
  };
  incidents: EventSimulatorIncident[];
  notifications: EventSimulatorNotification[];
  trace: EventSimulatorTraceStep[];
  artifacts: Record<string, unknown>;
}

function trainingHeaders(token: string) {
  return { "X-Training-Admin-Token": token };
}

export async function fetchTrainingStatus() {
  const { data } = await api.get<TrainingStatus>("/api/training/status");
  return data;
}

export async function submitTrainingFeedback(token: string, body: TrainingFeedbackRequest) {
  const { data } = await api.post("/api/training/feedback", body, {
    headers: trainingHeaders(token),
  });
  return data;
}

export async function downloadTrainingArchive(token: string): Promise<Blob> {
  const { data } = await api.get("/api/training/export", {
    headers: trainingHeaders(token),
    responseType: "blob",
    timeout: 120000,
  });
  return data;
}

export async function clearTrainingArchive(token: string) {
  const { data } = await api.post(
    "/api/training/clear",
    { confirm: "CLEAR" },
    { headers: trainingHeaders(token), timeout: 60000 },
  );
  return data;
}

export async function fetchEventSimulatorScenarios() {
  const { data } = await api.get<EventSimulatorScenarioSummary[]>("/api/event-simulator/scenarios");
  return data;
}

export async function fetchEventSimulatorRun(scenarioId: string) {
  const { data } = await api.get<EventSimulatorRun>(`/api/event-simulator/scenarios/${scenarioId}`);
  return data;
}

// --- Incidents (Stage 1) -----------------------------------------------------

export type IncidentStatus = "DRAFT" | "OPEN" | "UNDER_REVIEW" | "CONFIRMED" | "CLOSED" | "ARCHIVED";
export type AssetType =
  | "TRANSMISSION_LINE"
  | "TRANSFORMER"
  | "BUSBAR"
  | "FEEDER"
  | "REACTOR"
  | "CAPACITOR"
  | "OTHER"
  | "UNKNOWN";
export type ProtectionFamily =
  | "DISTANCE"
  | "LINE_DIFFERENTIAL"
  | "TRANSFORMER_DIFFERENTIAL"
  | "OVERCURRENT"
  | "REF"
  | "SBEF"
  | "MIXED"
  | "UNKNOWN";
export type ClockAssessment = "SYNCHRONIZED" | "LIKELY_SYNCHRONIZED" | "ORDER_ONLY" | "UNTRUSTED" | "UNKNOWN";
export type RecordAttachmentRole = "PRIMARY" | "SUPPORTING" | "REMOTE_END" | "BACKUP_RELAY" | "DFR_EXTERNAL" | "OTHER" | "UNKNOWN";
export type InclusionStatus = "INCLUDED" | "EXCLUDED" | "PENDING_REVIEW";
export type OrderSource = "ABSOLUTE_TIME" | "MANUAL" | "UPLOAD_ORDER" | "UNKNOWN";
export type EvidenceType =
  | "COMTRADE_RECORD"
  | "REMOTE_END_COMTRADE"
  | "RELAY_EVENT_REPORT"
  | "OPERATOR_SOE"
  | "FIELD_INSPECTION"
  | "PATROL_REPORT"
  | "LIGHTNING_DETECTION"
  | "PROTECTION_ENGINEER_NOTE"
  | "PHOTO"
  | "OTHER";
export type EvidenceConfidence = "CONFIRMED" | "PROBABLE" | "POSSIBLE" | "UNKNOWN";

export interface IncidentRecordOut {
  incident_record_id: string;
  incident_id: string;
  analysis_id: string;
  source_filename: string | null;
  station_name: string | null;
  bay_name: string | null;
  relay_id: string | null;
  relay_model: string | null;
  protection_type: string | null;
  record_start_iso: string | null;
  trigger_time_iso: string | null;
  trigger_offset_s: number | null;
  sequence_index: number;
  manual_order: number | null;
  order_source: OrderSource;
  attachment_role: RecordAttachmentRole;
  inclusion_status: InclusionStatus;
  exclusion_reason: string | null;
  canonical_snapshot: CanonicalRecordAnalysis | Record<string, unknown>;
  attachment_warnings: Array<{ type: string; description?: string; requires_review?: boolean; [key: string]: unknown }>;
  created_at: string;
}

export interface IncidentEvidenceOut {
  evidence_id: string;
  incident_id: string;
  evidence_type: EvidenceType;
  source: string;
  description: string;
  value: unknown;
  confidence: EvidenceConfidence;
  observed_at_iso: string | null;
  attachment_name: string | null;
  created_by: string | null;
  created_at: string;
}

export interface IncidentOut {
  incident_id: string;
  title: string;
  status: IncidentStatus;
  station_name: string | null;
  bay_name: string | null;
  asset_id: string | null;
  asset_name: string | null;
  asset_type: AssetType | null;
  voltage_level_kv: number | null;
  protection_family: ProtectionFamily | null;
  incident_start_iso: string | null;
  incident_end_iso: string | null;
  clock_assessment: ClockAssessment;
  clock_assessment_reason: string | null;
  record_ids: string[];
  evidence_ids: string[];
  observed_summary: {
    record_count: number;
    records_with_absolute_time: number;
    records_without_absolute_time: number;
    protection_types: string[];
    faulted_phase_sets: string[][];
    reclose_outcomes: Array<string | null>;
    cause_hypotheses_per_record: Array<{
      analysis_id: string;
      top_hypothesis: string | null;
      confidence: number | null;
      scope: string;
    }>;
  };
  incident_interpretation: { summary: string; scope: string };
  incident_hypotheses: Array<Record<string, unknown>>;
  missing_evidence: Array<{ type: string; description: string }>;
  operator_notes: string | null;
  created_at: string;
  updated_at: string;
  schema_version: string;
  records: IncidentRecordOut[];
  provenance: {
    schema_version: string;
    incident_engine_version: string;
    record_snapshot_versions: Array<Record<string, unknown>>;
    created_at: string;
    updated_at: string;
  };
}

export interface IncidentFeedbackOut {
  feedback_id: string;
  incident_id: string;
  operator: string | null;
  record_grouping_correct: boolean | null;
  actual_record_count: number | null;
  record_order_correct: boolean | null;
  corrected_record_order: string[] | null;
  incident_start_correct: boolean | null;
  corrected_incident_start_iso: string | null;
  incident_end_correct: boolean | null;
  corrected_incident_end_iso: string | null;
  clock_assessment_correct: boolean | null;
  actual_clock_assessment: string | null;
  incident_interpretation_correct: boolean | null;
  actual_incident_class: string | null;
  cause_correct: boolean | null;
  actual_root_cause: string | null;
  ground_truth_sources: string[];
  ground_truth_confidence: string;
  include_for_future_analysis: boolean;
  notes: string | null;
  incident_snapshot: Record<string, unknown>;
  created_at: string;
}

export interface CreateIncidentRequest {
  title: string;
  station_name?: string | null;
  bay_name?: string | null;
  asset_id?: string | null;
  asset_name?: string | null;
  asset_type?: AssetType | null;
  voltage_level_kv?: number | null;
  protection_family?: ProtectionFamily | null;
  operator_notes?: string | null;
}

export async function createIncident(body: CreateIncidentRequest) {
  const { data } = await api.post<IncidentOut>("/api/incidents", body);
  return data;
}

export async function listIncidents(params?: { status?: string; station_name?: string }) {
  const { data } = await api.get<IncidentOut[]>("/api/incidents", { params });
  return data;
}

export async function fetchIncident(incidentId: string) {
  const { data } = await api.get<IncidentOut>(`/api/incidents/${incidentId}`);
  return data;
}

export async function updateIncident(incidentId: string, patch: Partial<CreateIncidentRequest> & {
  status?: IncidentStatus;
  clock_assessment?: ClockAssessment;
  clock_assessment_reason?: string | null;
  incident_start_iso?: string | null;
  incident_end_iso?: string | null;
}) {
  const { data } = await api.patch<IncidentOut>(`/api/incidents/${incidentId}`, patch);
  return data;
}

export async function deleteIncident(incidentId: string) {
  const { data } = await api.delete(`/api/incidents/${incidentId}`);
  return data;
}

export interface AttachRecordRequest {
  analysis_id: string;
  attachment_role?: RecordAttachmentRole;
  bay_name?: string | null;
  relay_id?: string | null;
  relay_model?: string | null;
  protection_type?: string | null;
  source_filename?: string | null;
  override_warnings?: boolean;
  operator_notes?: string | null;
}

export async function attachIncidentRecord(incidentId: string, body: AttachRecordRequest) {
  const { data } = await api.post<IncidentRecordOut>(`/api/incidents/${incidentId}/records`, body);
  return data;
}

export async function listIncidentRecords(incidentId: string) {
  const { data } = await api.get<IncidentRecordOut[]>(`/api/incidents/${incidentId}/records`);
  return data;
}

export async function detachIncidentRecord(incidentId: string, incidentRecordId: string) {
  const { data } = await api.delete(`/api/incidents/${incidentId}/records/${incidentRecordId}`);
  return data;
}

export async function reorderIncidentRecords(incidentId: string, incidentRecordIds: string[]) {
  const { data } = await api.patch<IncidentRecordOut[]>(`/api/incidents/${incidentId}/records/order`, {
    incident_record_ids: incidentRecordIds,
  });
  return data;
}

export async function refreshIncidentSnapshots(incidentId: string) {
  const { data } = await api.post<IncidentRecordOut[]>(`/api/incidents/${incidentId}/refresh-snapshots`);
  return data;
}

export interface AddEvidenceRequest {
  evidence_type: EvidenceType;
  source?: string;
  description?: string;
  value?: unknown;
  confidence?: EvidenceConfidence;
  observed_at_iso?: string | null;
  attachment_name?: string | null;
  created_by?: string | null;
}

export async function addIncidentEvidence(incidentId: string, body: AddEvidenceRequest) {
  const { data } = await api.post<IncidentEvidenceOut>(`/api/incidents/${incidentId}/evidence`, body);
  return data;
}

export async function listIncidentEvidence(incidentId: string) {
  const { data } = await api.get<IncidentEvidenceOut[]>(`/api/incidents/${incidentId}/evidence`);
  return data;
}

export async function removeIncidentEvidence(incidentId: string, evidenceId: string) {
  const { data } = await api.delete(`/api/incidents/${incidentId}/evidence/${evidenceId}`);
  return data;
}

export interface SubmitIncidentFeedbackRequest {
  operator?: string | null;
  record_grouping_correct?: boolean | null;
  actual_record_count?: number | null;
  record_order_correct?: boolean | null;
  corrected_record_order?: string[] | null;
  incident_start_correct?: boolean | null;
  corrected_incident_start_iso?: string | null;
  incident_end_correct?: boolean | null;
  corrected_incident_end_iso?: string | null;
  clock_assessment_correct?: boolean | null;
  actual_clock_assessment?: string | null;
  incident_interpretation_correct?: boolean | null;
  actual_incident_class?: string | null;
  cause_correct?: boolean | null;
  actual_root_cause?: string | null;
  ground_truth_sources?: string[];
  ground_truth_confidence?: string;
  include_for_future_analysis?: boolean;
  notes?: string | null;
  // --- Stage 2 reconstruction correction fields ---
  same_bay_correct?: boolean | null;
  relationships_correct?: boolean | null;
  corrected_relationships?: Array<{ left_record_id: string; right_record_id: string; actual_relationship: string }>;
  episode_grouping_correct?: boolean | null;
  corrected_episode_groups?: string[][];
  evolving_fault_correct?: boolean | null;
  root_cause?: string | null;
}

export async function submitIncidentFeedback(incidentId: string, body: SubmitIncidentFeedbackRequest) {
  const { data } = await api.post<IncidentFeedbackOut>(`/api/incidents/${incidentId}/feedback`, body);
  return data;
}

export async function listIncidentFeedback(incidentId: string) {
  const { data } = await api.get<IncidentFeedbackOut[]>(`/api/incidents/${incidentId}/feedback`);
  return data;
}

// --- Incidents (Stage 2 — same-bay multi-COMTRADE reconstruction) -----------

export interface HealthResponse {
  status: string;
  version: string;
  analysis_storage: string;
  analysis_ttl_hours: number;
  warmup: Record<string, unknown>;
  feature_flags?: {
    multi_comtrade_enabled: boolean;
  };
}

export async function fetchHealth() {
  const { data } = await api.get<HealthResponse>("/api/health");
  return data;
}

export type SameBayStatus = "CONFIRMED_SAME_BAY" | "LIKELY_SAME_BAY" | "MISMATCH_REQUIRES_REVIEW" | "UNKNOWN";
export type AlignmentStatus = "ALIGNED" | "LIKELY_ALIGNED" | "ORDER_ONLY" | "MANUAL_ORDER" | "UNTRUSTED" | "INSUFFICIENT_DATA";
export type TimelineEventType =
  | "RECORD_START"
  | "RECORD_TRIGGER"
  | "FAULT_INCEPTION"
  | "PROTECTION_PICKUP"
  | "ZONE_OPERATE"
  | "TRIP_COMMAND"
  | "BREAKER_OPEN"
  | "FAULT_CLEARING"
  | "RECLOSE_START"
  | "RECLOSE_SUCCESS"
  | "RECLOSE_FAILED"
  | "REFAULT"
  | "RECORD_END"
  | "DATA_GAP"
  | "CLOCK_WARNING"
  | "MANUAL_ANNOTATION";
export type RelationshipType =
  | "DUPLICATE_TRIGGER"
  | "OVERLAPPING_CAPTURE"
  /** The other line end's recording of the same event (another substation's recorder). */
  | "REMOTE_END_CAPTURE"
  | "CONTINUATION"
  | "RECLOSE_SEQUENCE"
  | "REFAULT_AFTER_RECLOSE"
  | "NEW_FAULT_EPISODE"
  | "REPEATED_FAULT"
  | "POSSIBLE_EVOLVING_FAULT"
  | "UNRELATED"
  | "UNCERTAIN";
export type PhysicalCauseConsistency = "CONSISTENT" | "MOSTLY_CONSISTENT" | "MIXED" | "CONTRADICTORY" | "INSUFFICIENT";

export interface AlignmentGap {
  left_incident_record_id: string;
  right_incident_record_id: string;
  /** Trigger to trigger. */
  gap_ms: number;
  /** Left record's last sample to the right record's first; negative when they overlap. */
  data_gap_ms?: number;
  precise: boolean;
}

/** Where one record sits on the incident time axis (webapp/api/incidents/time_axis.py). */
export interface TimeAxisPlacement {
  incident_record_id: string;
  clock_group: string;
  method: "reference_clock" | "fault_aligned" | "own_clock_unverified" | "unplaced";
  /** Seconds added to the recorder clock to read the incident clock. */
  correction_s: number | null;
  /** This recorder's clock minus the reference clock, whole time zones. */
  zone_offset_h: number | null;
  /** This recorder's clock minus the reference clock beyond the zone, at the shared fault. */
  clock_offset_ms: number | null;
  aligned_on: { reference_record_id: string; record_id: string; agreeing_pairs: number; candidate_pairs: number } | null;
  /** First sample on the incident clock. */
  start_iso: string | null;
  offset_s: number | null;
  span_s: number | null;
}

export interface IncidentTimeAxis {
  reference_group: string | null;
  zero_iso: string | null;
  records: TimeAxisPlacement[];
  warnings: Array<{ type: string; description?: string; [key: string]: unknown }>;
}

export interface AlignmentAssessmentOut {
  status: AlignmentStatus;
  confidence: number;
  order_source: OrderSource;
  record_order: string[];
  pairwise_gaps_ms: AlignmentGap[];
  overlap_groups: string[][];
  warnings: Array<{ type: string; description?: string; requires_review?: boolean; [key: string]: unknown }>;
  assumptions: string[];
  /** Absent on reconstructions made before the incident time axis existed. */
  time_axis?: IncidentTimeAxis;
}

export interface IncidentTimelineEventOut {
  timeline_event_id: string;
  incident_id: string;
  incident_record_id: string | null;
  episode_id: string | null;
  event_type: TimelineEventType;
  absolute_time_iso: string | null;
  relative_incident_ms: number | null;
  relative_record_ms: number | null;
  source: string;
  label: string;
  details: Record<string, unknown>;
  confidence: number;
  provenance: Record<string, unknown>;
}

export interface RecordRelationshipOut {
  relationship_id: string;
  incident_id: string;
  left_record_id: string;
  right_record_id: string;
  relationship_type: RelationshipType;
  confidence: number;
  evidence_for: Array<Record<string, unknown>>;
  evidence_against: Array<Record<string, unknown>>;
  assumptions: string[];
  warnings: Array<Record<string, unknown>>;
  metrics: {
    /** Trigger to trigger, on the incident time axis. */
    gap_seconds?: number;
    /** Left record's last sample to the right record's first; negative when they overlap. */
    data_gap_seconds?: number;
    waveform_similarity?: {
      computed: boolean;
      reason?: string | null;
      overlap_seconds?: number;
      channels_compared?: string[];
      mean_correlation?: number | null;
      mean_rms_relative_diff?: number | null;
    };
    digital_sequence_similarity?: number;
    [key: string]: unknown;
  };
  overridden: boolean;
  override_operator: string | null;
  override_reason: string | null;
  override_previous_type: string | null;
  override_at_iso: string | null;
  reconstruction_version: string | null;
}

export interface RecordLocalCauseHypothesis {
  analysis_id: string;
  top_hypothesis: string | null;
  confidence: number | null;
  cause_ranking?: Array<{ cause: string; label: string; confidence: number }>;
  model_version?: string | null;
  timing_source?: string | null;
  scope: "RECORD_LOCAL_SIGNATURE";
  evidence_role: CauseEvidenceRole;
  fault_type?: "transient" | "permanent" | null;
  requires_review?: boolean;
  skip_reason?: string | null;
}

export interface FaultEpisodeOut {
  episode_id: string;
  incident_id: string;
  member_record_ids: string[];
  episode_index: number;
  start_iso: string | null;
  end_iso: string | null;
  duration_ms: number | null;
  faulted_phases: string[];
  fault_type: string | null;
  zone_operations: string[];
  trip_types: string[];
  reclose_outcome: "successful" | "failed" | null;
  electrical_summary: Record<string, unknown>;
  local_cause_hypotheses: RecordLocalCauseHypothesis[];
  relationship_to_previous: RelationshipType | null;
  confidence: number;
  observed_facts: Record<string, unknown>;
  interpretation: { event_classes?: string[]; reasoning?: EpisodeReasoning; [key: string]: unknown };
  missing_evidence: Array<{ type: string; description: string }>;
  provenance: Record<string, unknown>;
}

// "inception" = this record captures an independently faulted waveform, its
// cause reading is real evidence. "aftermath" = this record's only relation
// to its predecessor is a reclose/continuation/duplicate capture — its
// classifier reading is preserved for audit but is not independent cause
// evidence and should not be read as disagreeing with the inception record.
// "remote_end" = the other line end's recording of an event this end
// recorded: another view of the same fault, likewise not separate evidence.
export type CauseEvidenceRole = "inception" | "aftermath" | "remote_end";

/** What another recorder (the far line end, a second device in the bay) saw of an episode. */
export interface EpisodeOtherRecorder {
  recorder: string;
  station: string;
  same_station: boolean;
  member_record_ids: string[];
  relationship_types: RelationshipType[];
  fault_record_id: string | null;
  faulted_phases: string[];
  fct_ms: number | null;
  fault_start_difference_ms: number | null;
  reclose_outcome: "successful" | "failed" | null;
  reclose_dead_time_s: number | null;
  clock: { method: TimeAxisPlacement["method"] | null; zone_offset_h: number | null; clock_offset_ms: number | null };
}

export interface PhysicalCauseRecordEntry {
  analysis_id: string;
  incident_record_id: string;
  top_hypothesis: string | null;
  confidence: number | null;
  cause_ranking: Array<{ cause: string; label: string; confidence: number }>;
  model_version: string | null;
  feature_version: string | null;
  calibration_method: string | null;
  timing_source: string | null;
  timing_confidence?: number | null;
  raw_probabilities: Record<string, number> | null;
  calibrated_probabilities: Record<string, number> | null;
  applied_caps: Array<{ name: string; before: number; after: number; reason: string }>;
  evidence_role: CauseEvidenceRole;
  // "transient" causes (PETIR/LAYANG/HEWAN/BENDA_ASING) are expected to
  // self-clear on reclose; "permanent" causes (KONDUKTOR/PERALATAN) are
  // expected to persist through reclose. Only set for "inception"-role
  // records — see reconstruction.py::_apply_reclose_outcome_cross_validation.
  fault_type?: "transient" | "permanent" | null;
  // True when this record's own aftermath record's reclose outcome
  // physically CONTRADICTS its fault_type (e.g. a transient cause but a
  // failed reclose) — the reclose_outcome_conflict cap already lowered
  // confidence; this flag exists so the UI can surface it without parsing
  // applied_caps.
  requires_review?: boolean;
  // Set (instead of a null top_hypothesis meaning "model/session failed")
  // when this record's protection_type isn't one the line-fault classifier
  // is built for — e.g. "unsupported_protection_type" for an 87T
  // transformer-differential record. No AI cause classifier for
  // transformer events exists anywhere in this app yet (see README), so
  // this is a deliberate, disclosed absence of evidence, not an error.
  skip_reason?: string | null;
}

export interface PhysicalCauseEvidenceOut {
  scope: "RECORD_LOCAL_SIGNATURES";
  records: PhysicalCauseRecordEntry[];
  consistency: PhysicalCauseConsistency;
  incident_root_cause: "UNCONFIRMED" | string;
}

export interface IncidentHypothesis {
  hypothesis: string;
  // null for signals that deliberately state a disagreement between two
  // readings rather than scoring one (e.g. REPEATED_ESCALATING_SIGNATURE_AMBIGUOUS)
  // — see reconstruction.py::_pattern_based_cause_signals.
  confidence: number | null;
  evidence_for: string[];
  evidence_against: string[];
  mechanism_signal?: string;
  episode_indices?: number[];
  description?: string;
}

export interface ReconstructionOut {
  reconstruction_id: string;
  incident_id: string;
  engine_version: string;
  schema_version: string;
  same_bay_status: SameBayStatus;
  same_bay_evidence: Array<{ type: string; description?: string; requires_review?: boolean; [key: string]: unknown }>;
  same_bay_override: { operator?: string | null; reason?: string | null; at_iso?: string | null } | null;
  alignment: AlignmentAssessmentOut;
  timeline_event_ids: string[];
  relationship_ids: string[];
  episode_ids: string[];
  observed_incident_facts: {
    record_count: number;
    episode_count: number;
    incident_duration_ms: number | null;
    phase_sequence: string[][];
    reclose_sequence: Array<"successful" | "failed" | null>;
    [key: string]: unknown;
  };
  protection_sequence_interpretation: { event_class: string; summary: string };
  incident_hypotheses: IncidentHypothesis[];
  physical_cause_evidence: PhysicalCauseEvidenceOut;
  narrative: string;
  record_snapshot_versions: Array<Record<string, unknown>>;
  is_latest: boolean;
  supersedes: string | null;
  created_at: string;
  // Expanded by the /reconstruct, /reconstruction endpoints (not stored on the bare model).
  timeline?: IncidentTimelineEventOut[];
  relationships?: RecordRelationshipOut[];
  episodes?: FaultEpisodeOut[];
}

export interface ReconstructRequest {
  same_bay_override_reason?: string | null;
  same_bay_override_operator?: string | null;
}

export async function reconstructIncident(incidentId: string, body: ReconstructRequest = {}) {
  const { data } = await api.post<ReconstructionOut>(`/api/incidents/${incidentId}/reconstruct`, body);
  return data;
}

export async function fetchReconstruction(incidentId: string, reconstructionId?: string) {
  const { data } = await api.get<ReconstructionOut>(`/api/incidents/${incidentId}/reconstruction`, {
    params: reconstructionId ? { reconstruction_id: reconstructionId } : undefined,
  });
  return data;
}

export async function listReconstructions(incidentId: string) {
  const { data } = await api.get<ReconstructionOut[]>(`/api/incidents/${incidentId}/reconstructions`);
  return data;
}

export async function fetchIncidentTimeline(incidentId: string) {
  const { data } = await api.get<IncidentTimelineEventOut[]>(`/api/incidents/${incidentId}/timeline`);
  return data;
}

export async function fetchIncidentRelationships(incidentId: string) {
  const { data } = await api.get<RecordRelationshipOut[]>(`/api/incidents/${incidentId}/relationships`);
  return data;
}

export async function fetchIncidentEpisodes(incidentId: string) {
  const { data } = await api.get<FaultEpisodeOut[]>(`/api/incidents/${incidentId}/episodes`);
  return data;
}

export interface JoinedWaveformChannel {
  t: number[];
  values: number[];
}

export interface JoinedWaveformSegment {
  incident_record_id: string;
  source_filename: string | null;
  t_offset_s: number;
  // Gap BEFORE this segment; null for the first segment in the episode.
  gap_precision: "measured" | "assumed_back_to_back" | null;
  gap_seconds: number | null;
  channels: Record<string, JoinedWaveformChannel>;
}

export interface JoinedWaveformGapRange {
  start_s: number;
  end_s: number;
  precision: "measured" | "assumed_back_to_back";
  long_gap: boolean;
}

export interface JoinedWaveformOtherLane {
  recorder: string;
  station: string;
  clock_method: TimeAxisPlacement["method"] | null;
  segments: Array<{
    incident_record_id: string;
    source_filename: string | null;
    t_offset_s: number;
    channels: Record<string, JoinedWaveformChannel>;
  }>;
}

export interface JoinedWaveformOut {
  episode_id: string;
  can_join: boolean;
  reason: string | null;
  /** The episode's own recorder, joined in time order. */
  segments: JoinedWaveformSegment[];
  gap_ranges: JoinedWaveformGapRange[];
  /** Other recorders' recordings of the episode (the far line end), on the same time axis. */
  other_lanes?: JoinedWaveformOtherLane[];
  warnings: Array<{ type: string; description?: string; [key: string]: unknown }>;
}

export async function fetchJoinedWaveform(incidentId: string, episodeId: string) {
  const { data } = await api.get<JoinedWaveformOut>(`/api/incidents/${incidentId}/episodes/${episodeId}/joined-waveform`);
  return data;
}

export interface RelationshipOverrideRequest {
  corrected_relationship: RelationshipType;
  operator: string;
  reason: string;
}

export async function overrideRelationship(incidentId: string, relationshipId: string, body: RelationshipOverrideRequest) {
  const { data } = await api.post<RecordRelationshipOut>(
    `/api/incidents/${incidentId}/relationships/${relationshipId}/override`,
    body,
  );
  return data;
}

// --- Batch upload -------------------------------------------------------------

export interface BatchUploadRecordResult {
  analysis_id: string | null;
  incident_record_id: string | null;
  source_files: string[];
  status: "created" | "error";
  error?: string | null;
}

export interface BatchUploadError {
  files: string[];
  reason: string;
}

export interface BatchUploadResponse {
  incident_id: string;
  records_created: BatchUploadRecordResult[];
  errors: BatchUploadError[];
  reconstruction_status: "completed" | "partial" | "failed" | "aborted_atomic" | "not_run";
}

export interface BatchUploadOptions {
  partialSuccess?: boolean;
  attachmentRole?: RecordAttachmentRole;
  bayName?: string | null;
  protectionType?: string | null;
  overrideWarnings?: boolean;
}

export async function uploadIncidentRecords(incidentId: string, files: File[], options: BatchUploadOptions = {}) {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));

  const params: Record<string, string | boolean> = {};
  if (options.partialSuccess) params.partial_success = true;
  if (options.attachmentRole) params.attachment_role = options.attachmentRole;
  if (options.bayName) params.bay_name = options.bayName;
  if (options.protectionType) params.protection_type = options.protectionType;
  if (options.overrideWarnings) params.override_warnings = true;

  const { data } = await api.post<BatchUploadResponse>(`/api/incidents/${incidentId}/upload-records`, form, {
    params,
    timeout: 120000,
  });
  return data;
}

// --- Client-side file-pairing preview (advisory only — backend pairing is authoritative) ---

export interface FilePairPreviewGroup {
  kind: "cff" | "cfg_dat";
  stem: string;
  files: string[];
}

export interface FilePairPreviewError {
  files: string[];
  reason: string;
}

export interface FilePairPreview {
  groups: FilePairPreviewGroup[];
  errors: FilePairPreviewError[];
}

function fileStem(filename: string): string {
  const base = filename.split(/[\\/]/).pop() || filename;
  const dot = base.lastIndexOf(".");
  return (dot > 0 ? base.slice(0, dot) : base).toLowerCase();
}

function fileSuffix(filename: string): string {
  const base = filename.split(/[\\/]/).pop() || filename;
  const dot = base.lastIndexOf(".");
  return dot >= 0 ? base.slice(dot).toLowerCase() : "";
}

/**
 * Client-side preview of how the backend will pair uploaded files, so the
 * batch-upload panel can show groupings/orphans/duplicates before the
 * request is sent. Mirrors webapp/api/incidents/batch_upload.py::pair_files
 * exactly (case-insensitive stem match for .cfg+.dat, standalone .cff),
 * but this preview is advisory only — the backend re-derives pairing from
 * the actual uploaded bytes and is authoritative.
 */
export function previewFilePairing(files: File[]): FilePairPreview {
  const groups: FilePairPreviewGroup[] = [];
  const errors: FilePairPreviewError[] = [];

  const cffFiles = files.filter((f) => fileSuffix(f.name) === ".cff");
  const cfgFiles = files.filter((f) => fileSuffix(f.name) === ".cfg");
  const datFiles = files.filter((f) => fileSuffix(f.name) === ".dat");
  const unsupported = files.filter((f) => ![".cff", ".cfg", ".dat"].includes(fileSuffix(f.name)));

  unsupported.forEach((f) => {
    errors.push({ files: [f.name], reason: `Unsupported file type '${fileSuffix(f.name) || "(none)"}'.` });
  });

  cffFiles.forEach((f) => {
    groups.push({ kind: "cff", stem: fileStem(f.name), files: [f.name] });
  });

  const cfgByStem = new Map<string, File[]>();
  cfgFiles.forEach((f) => {
    const stem = fileStem(f.name);
    cfgByStem.set(stem, [...(cfgByStem.get(stem) || []), f]);
  });
  const datByStem = new Map<string, File[]>();
  datFiles.forEach((f) => {
    const stem = fileStem(f.name);
    datByStem.set(stem, [...(datByStem.get(stem) || []), f]);
  });

  const allStems = Array.from(new Set([...cfgByStem.keys(), ...datByStem.keys()])).sort();
  allStems.forEach((stem) => {
    const cfgs = cfgByStem.get(stem) || [];
    const dats = datByStem.get(stem) || [];

    if (cfgs.length > 1 || dats.length > 1) {
      errors.push({
        files: [...cfgs, ...dats].map((f) => f.name),
        reason: `Duplicate files for stem '${stem}' — expected exactly one .cfg and one .dat.`,
      });
      return;
    }
    if (cfgs.length > 0 && dats.length === 0) {
      errors.push({ files: [cfgs[0].name], reason: "Orphan .cfg file with no matching .dat file." });
      return;
    }
    if (dats.length > 0 && cfgs.length === 0) {
      errors.push({ files: [dats[0].name], reason: "Orphan .dat file with no matching .cfg file." });
      return;
    }
    groups.push({ kind: "cfg_dat", stem, files: [cfgs[0].name, dats[0].name] });
  });

  return { groups, errors };
}
