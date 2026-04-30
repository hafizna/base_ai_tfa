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

export async function uploadComtrade(cfg: File, dat: File) {
  const form = new FormData();
  form.append("cfg_file", cfg);
  form.append("dat_file", dat);
  const { data } = await api.post<UploadedAnalysis>("/api/upload", form);
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
    i_peak_ia_a?: number;
    i_peak_ib_a?: number;
    i_peak_ic_a?: number;
    v_sag_pct?: number;
    i_pos_seq_a?: number;
    i_neg_seq_a?: number;
    i_zero_seq_a?: number;
    z_at_inception_ohm?: number;
    r_at_fault_ohm?: number;
    x_at_fault_ohm?: number;
    rx_ratio?: number;
    z_angle_deg?: number;
    ar_dead_time_ms?: number;
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
