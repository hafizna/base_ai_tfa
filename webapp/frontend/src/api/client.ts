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
