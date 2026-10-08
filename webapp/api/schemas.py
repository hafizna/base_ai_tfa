"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict, Any


class AnalogChannelOut(BaseModel):
    id: str
    name: str
    canonical_name: str
    unit: str
    phase: Optional[str]
    measurement: str
    ct_primary: float
    ct_secondary: float
    pors: str
    samples: List[float]


class StatusChannelOut(BaseModel):
    id: str
    name: str
    samples: List[int]


class ComtradeOut(BaseModel):
    station_name: str
    rec_dev_id: str
    rev_year: str
    sampling_rates: List[Tuple[float, int]]
    trigger_time: float
    # Absolute COMTRADE timing metadata (Stage 0). None when the source file did
    # not carry a usable wall-clock timestamp - never inferred/guessed.
    start_time_iso: Optional[str] = None
    trigger_time_iso: Optional[str] = None
    trigger_offset_s: float = 0.0
    time_code: Optional[str] = None
    local_code: Optional[str] = None
    clock_quality: Optional[str] = None
    total_samples: int
    frequency: float
    time: List[float]
    analog_channels: List[AnalogChannelOut]
    status_channels: List[StatusChannelOut]
    warnings: List[str]


class AnalysisCreatedResponse(BaseModel):
    analysis_id: str
    station_name: str
    rec_dev_id: str
    total_samples: int
    analog_channel_count: int
    status_channel_count: int
    suggested_relay_type: Optional[str] = None   # auto-detected from status channels, None if unknown
    detection_confidence: Optional[float] = None


class AnalysisRequestBase(BaseModel):
    analysis_id: str


class AnalogChannelSummaryOut(BaseModel):
    id: str
    name: str
    canonical_name: str
    unit: str
    phase: Optional[str]
    measurement: str
    ct_primary: float
    ct_secondary: float
    pors: str


class StatusChannelSummaryOut(BaseModel):
    id: str
    name: str
    sample_count: int
    on_count: int
    transition_count: int


class AnalysisSummaryOut(BaseModel):
    analysis_id: str
    station_name: str
    rec_dev_id: str
    rev_year: str
    sampling_rates: List[Tuple[float, int]]
    trigger_time: float
    start_time_iso: Optional[str] = None
    trigger_time_iso: Optional[str] = None
    trigger_offset_s: float = 0.0
    time_code: Optional[str] = None
    local_code: Optional[str] = None
    clock_quality: Optional[str] = None
    total_samples: int
    frequency: float
    duration_ms: float
    analog_channels: List[AnalogChannelSummaryOut]
    status_channels: List[StatusChannelSummaryOut]
    warnings: List[str]


class RatioChannel(BaseModel):
    channel_id: str
    primary: float
    secondary: float


class RecalcRequest(BaseModel):
    comtrade: ComtradeOut
    ratios: List[RatioChannel]


class RecalcByIdRequest(AnalysisRequestBase):
    ratios: List[RatioChannel]


# --- Relay 21 ---

class ZoneConfig(BaseModel):
    """Single protection zone definition (mho or quadrilateral)."""
    label: str           # "Z1", "Z2", "Z3"
    shape: str           # "mho" | "quad"
    # Mho: center_r, center_x, radius
    center_r: float = 0.0
    center_x: float = 0.0
    radius: float = 0.0
    # Quad: rf_fwd, rf_rev, xf, xr, line_angle_deg
    rf_fwd: float = 0.0
    rf_rev: float = 0.0
    xf: float = 0.0
    xr: float = 0.0
    line_angle_deg: float = 75.0
    color: str = "#3b82f6"


class LocusRequest(BaseModel):
    comtrade: ComtradeOut
    zones: List[ZoneConfig] = []
    loop: str = "ZA"     # ZA | ZB | ZC | ZAB | ZBC | ZCA


class LocusAnalysisRequest(AnalysisRequestBase):
    zones: List[ZoneConfig] = []
    loop: str = "ZA"
    k0: float = 0.0
    k0_angle_deg: float = 0.0
    invert_i: bool = False
    ct_ratio_override: Optional[float] = None
    vt_ratio_override: Optional[float] = None


class LocusPoint(BaseModel):
    t: float
    r: float
    x: float


class LocusDiagnostics(BaseModel):
    """Data-quality signals for one loop's computed locus — surfaced instead
    of silently smoothed over, so a credible/defensible locus discloses how
    much of the record its trajectory actually rests on."""
    windows_evaluated: int = 0
    windows_kept: int = 0
    retention_pct: float = 0.0
    ct_saturation_suspected: bool = False
    ct_saturation_thd_ratio: float = 0.0
    dc_offset_corrected: bool = False


class LocusResponse(BaseModel):
    loop: str
    points: List[LocusPoint]
    zones: List[ZoneConfig]
    fault_inception_idx: Optional[int]
    diagnostics: LocusDiagnostics = LocusDiagnostics()


class LocusBatchRequest(AnalysisRequestBase):
    """Compute several loops in one request so the stored COMTRADE payload is
    loaded/parsed once instead of once per loop (6× for the locus panel)."""
    loops: List[str] = ["ZA", "ZB", "ZC", "ZAB", "ZBC", "ZCA"]
    k0: float = 0.0
    k0_angle_deg: float = 0.0
    invert_i: bool = False
    ct_ratio_override: Optional[float] = None
    vt_ratio_override: Optional[float] = None


class LocusBatchResponse(BaseModel):
    points_by_loop: dict[str, List[LocusPoint]]
    diagnostics_by_loop: dict[str, LocusDiagnostics] = {}


class LocusEvent(BaseModel):
    """A single curated digital event placed on the impedance-locus timeline."""
    time_ms: float          # absolute time in the record (ms)
    rel_ms: float           # relative to fault inception (negative = pre-fault)
    channel: str            # raw status channel name
    state: int              # 1 = asserted, 0 = de-asserted
    category: str           # trip | zone | reclose | breaker | comms | other
    label: str              # short human label for the marker


class LocusEventsResponse(BaseModel):
    inception_time_ms: Optional[float]
    events: List[LocusEvent]
    timing_source: Optional[str] = None
    timing_confidence: Optional[float] = None


class AIFaultFeatures(BaseModel):
    analysis_id: Optional[str] = None
    fault_inception_angle_deg: float
    fault_duration_ms: float
    prefault_load_a: float
    impedance_at_trip_ohm: float
    waveform_asymmetry: float
    dc_offset: float
    ar_result: Optional[str]   # "successful" | "failed" | None


class AIFaultResult(BaseModel):
    cause_ranking: List[Dict[str, Any]]   # [{cause, label, confidence}]
    fault_type: str                        # "transient" | "permanent" | "none"
    overall_confidence: float
    # True when the no-fault gate fired: record was triggered (e.g. FD pickup)
    # but no protection operated and no fault signature is present.
    no_fault: bool = False
    # "reclose_capture" when the record starts during breaker dead time and
    # only captures the reclose of an earlier fault (no cause classification).
    record_kind: Optional[str] = None
    reclose_outcome: Optional[str] = None
    skip_reason: Optional[str] = None
    # Evidence can be plain strings (legacy responses) or structured items
    # {text, severity, weight, kind}. UI handles both shapes.
    evidence: List[Any]
    # --- Introspection fields (all optional for backwards compatibility) ---
    tier1: Optional[Dict[str, Any]] = None
    raw_probabilities: Optional[Dict[str, float]] = None
    calibrated_probabilities: Optional[Dict[str, float]] = None
    applied_caps: List[Dict[str, Any]] = []
    feature_vector_used: Optional[Dict[str, float]] = None
    meta: Optional[Dict[str, Any]] = None


# --- Relay 87L / 87T ---

class DiffRestraintSample(BaseModel):
    t: float
    i_diff: float
    i_rest: float
    phase: str   # "L1" | "L2" | "L3"


class CharacteristicParams(BaseModel):
    device_type: str = "SP5"     # "SP5" | "SP4"
    idiff_pickup: float = 0.20
    slope1: float = 0.30
    intersection1: float = 0.30
    slope2: float = 0.70
    intersection2: float = 2.50
    idiff_fast: float = 7.50
    # Base current In (A) for p.u. normalisation of a true two-terminal diff.
    # 0 = auto-estimate from the healthiest phase's prefault through-current.
    in_base_a: float = 0.0


class DiffRestraintRequest(BaseModel):
    comtrade: ComtradeOut
    params: CharacteristicParams = CharacteristicParams()
    relay_type: str = "87L"     # "87L" | "87T"


class DiffRestraintAnalysisRequest(AnalysisRequestBase):
    params: CharacteristicParams = CharacteristicParams()
    relay_type: str = "87L"


class TripMarker(BaseModel):
    """A detected trip event mapped onto the operating-point trajectory."""
    kind: str               # "RELAY_TRIP" | "DIFF" | "DIFF_FAST"
    channel_name: str
    t: float                # timestamp of first assertion (s)
    phase: Optional[str] = None   # "L1"/"L2"/"L3" if the trip channel is phase-segregated
    i_diff: float           # operating point at the trip instant
    i_rest: float


class PhaseClassification(BaseModel):
    """Per-phase summary + verdict, mirroring physical relay event records."""
    phase: str              # "L1" | "L2" | "L3"
    verdict: str            # "Internal Fault" | "Through Fault" | "Not Operated" | "Inrush?"
    confidence: str         # "high" | "medium" | "low"
    max_idiff: float
    max_irest: float
    max_ratio: float        # max(i_diff / threshold) — >1.0 = inside operate region


class DiffRestraintResponse(BaseModel):
    samples: List[DiffRestraintSample]
    params: CharacteristicParams
    operated_status: str    # "NOT_OPERATED" | "IDIFF_OPERATED" | "IDIFF_FAST_OPERATED"
    operated_phases: List[str]
    trip_markers: List[TripMarker] = []
    phase_classification: List[PhaseClassification] = []
    # "TWO_TERMINAL" = relay-computed diff available (true differential);
    # "LOCAL_ONLY" = only local-terminal currents — diff/restraint scatter is
    # local current, NOT a true two-terminal differential. Verdict then comes
    # from the relay's own per-phase diff trip signals (relay_diff_phases).
    diff_data_mode: str = "TWO_TERMINAL"
    relay_diff_phases: List[str] = []
    no_fault: bool = False
    no_fault_reasons: List[str] = []


# --- Relay OCR ---

class OvercurrentRequest(BaseModel):
    comtrade: ComtradeOut
    curve_type: str = "NI"       # NI | VI | EI
    is_pickup_a: float = 1.0
    tms: float = 0.1


class OvercurrentAnalysisRequest(AnalysisRequestBase):
    curve_type: str = "NI"
    is_pickup_a: float = 1.0
    tms: float = 0.1


class OvercurrentPoint(BaseModel):
    current_ratio: float
    trip_time_s: float


class OvercurrentResponse(BaseModel):
    curve_points: List[OvercurrentPoint]
    measured_current_a: float
    measured_trip_time_s: Optional[float]
    intersection_ratio: Optional[float]


# --- Multi-stage TCC (Time-Current Characteristic) ---
# Absolute-Ampere axis, multiple coordinated stages (inverse + definite-time/INST),
# per-phase fault points. Mirrors physical relay setting sheets (elemen waktu + I>> moment).

class TccStage(BaseModel):
    """One protection stage. curve_type 'DT' = definite time / instant (uses definite_time_s)."""
    label: str = "S1"                       # display label, e.g. S1, S2, I>>
    curve_type: str = "NI"                  # NI | VI | EI | LTI | DT  (DT = Definite Time / INST)
    is_pickup_a: float = 100.0              # pickup in ABSOLUTE Amperes
    tms: float = 0.1                        # time multiplier (inverse stages only)
    definite_time_s: float = 0.0            # operating time for DT stages (e.g. 0.3 phase, 0.7 trafo)


class TccCurveLine(BaseModel):
    """Computed line for one stage: list of (current_a, trip_time_s) points to plot."""
    label: str
    curve_type: str
    is_pickup_a: float
    currents_a: List[float]
    trip_times_s: List[float]


class TccFaultPoint(BaseModel):
    """A measured fault current mapped onto the fastest-operating stage."""
    channel_label: str                      # e.g. "A", "B", "C", "N/EF"
    current_a: float
    winning_stage_label: Optional[str]      # stage that trips first; None = below all pickups
    winning_curve_type: Optional[str]
    trip_time_s: Optional[float]
    multiple_of_pickup: Optional[float]     # current / winning stage pickup
    is_moment: bool = False                 # True if winning stage is DT/instant


class TccRequest(AnalysisRequestBase):
    mode: str = "phase"                     # "phase" (A/B/C) | "ef" (IN/I0/IE)
    domain: str = "line"                    # "line" | "trafo" — affects moment-time guidance text
    stages: List[TccStage] = []


class TccResponse(BaseModel):
    mode: str
    domain: str
    curves: List[TccCurveLine]
    fault_points: List[TccFaultPoint]
    assessment: str                         # descriptive evaluation text (kept + enriched)


# --- Double Ended FL (two-terminal Kirchhoff fault location) ---
# Two independently-uploaded COMTRADE records (one per line terminal), manually
# time-synchronized by the user (mirrors the SIGRA workflow this feature is
# modeled on), then combined via Kirchhoff's voltage law. Ground faults use
# negative-sequence quantities instead of an uncompensated phase-current/Z1
# shortcut. See webapp/api/routers/relay_21_de.py module docstring.

class DoubleEndedAlignRequest(BaseModel):
    analysis_id_a: str
    analysis_id_b: str


class DoubleEndedAlignResponse(BaseModel):
    inception_time_a_s: Optional[float]         # record-relative, from A's own time[]
    inception_time_b_s: Optional[float]         # record-relative, from B's own time[]
    timing_source_a: str
    timing_source_b: str
    estimated_shift_ms: Optional[float]         # coarse ISO-based starting point for the sync UI
    estimate_available: bool
    estimate_reason: str                        # why unavailable, or how it was derived


class DoubleEndedSuggestShiftRequest(BaseModel):
    analysis_id_a: str
    analysis_id_b: str
    loop: str = "ZA"
    line_len_km: float
    r1_ohm_per_km: float
    x1_ohm_per_km: float
    invert_i_a: bool = False
    invert_i_b: bool = False
    invert_phase_sequence_a: bool = False
    invert_phase_sequence_b: bool = False


class DoubleEndedSuggestShiftResponse(BaseModel):
    # A residual-minimizing shift, grounded in the same Kirchhoff physics
    # the final calculation uses — but still only a SUGGESTION to visually
    # confirm on the sync overlay before running the calculation, never
    # auto-applied. See _find_optimal_shift's docstring in relay_21_de.py.
    shift_ms: Optional[float]
    residual: Optional[float]
    searched_range_ms: float
    reason: str


class DoubleEndedComputeRequest(BaseModel):
    analysis_id_a: str
    analysis_id_b: str
    loop: str = "ZA"                            # one of LOOP_CHANNELS: ZA/ZB/ZC/ZAB/ZBC/ZCA
    line_len_km: float
    r1_ohm_per_km: float
    x1_ohm_per_km: float
    manual_shift_ms: float = 0.0                # user-confirmed cursor-drag offset, B relative to A
    invert_i_a: bool = False
    invert_i_b: bool = False
    invert_phase_sequence_a: bool = False
    invert_phase_sequence_b: bool = False
    # No CT/VT ratio override here: stored COMTRADE samples are already in
    # primary units (core/comtrade_parser.py), and this calculation works
    # entirely in primary volts/amps vs primary ohm-per-km — unlike the
    # single-ended locus, which additionally scales to relay-secondary ohms
    # for zone-overlay display. If a record's CT/PT ratio is wrong, fix it
    # via the existing CTVTRatioCorrection panel (recalculate-ratio) before
    # running this calculation, not via a local override here.


class DoubleEndedSingleEndedResult(BaseModel):
    """One terminal's OWN single-ended distance reading — shown alongside
    the two-ended answer as a contrast, per the source PPTX's own stated
    motivation (single-ended error grows with fault resistance Rf, which
    the two-ended Kirchhoff solve eliminates but a single-ended reading
    cannot separate from line impedance)."""
    terminal: str                                # "A" | "B"
    distance_km: float
    distance_pct: float
    fault_current_a: float
    r_measured_ohm: float
    x_measured_ohm: float
    warnings: List[str] = []


class DoubleEndedComputeResponse(BaseModel):
    loop: str
    distance_km: float
    distance_pct: float                         # distance_km / line_len_km * 100
    fault_current_a: float                      # |I_A + I_B| at the fault point, primary amps
    m_residual_imag: float                      # imaginary residual of the per-unit-distance solution
    kvl_residual: float = 0.0                   # normalized complex multi-window KVL mismatch
    distance_spread_km: float = 0.0             # weighted spread between selected window solutions
    selected_window_count: int = 1
    calculation_basis: str = "phase_loop"       # negative_sequence for ground loops
    inception_time_a_s: float
    inception_time_b_s: float
    active_tag_a: Optional[str]
    active_tag_b: Optional[str]
    warnings: List[str] = []
    single_ended_a: Optional[DoubleEndedSingleEndedResult] = None
    single_ended_b: Optional[DoubleEndedSingleEndedResult] = None
    # Per-window two-ended distance samples across the fault's own detected
    # duration (window-voting) — each a real Kirchhoff solution, not a
    # fabricated confidence score. See _compute_distance_histogram's
    # docstring in relay_21_de.py for the full rationale.
    distance_histogram_km: List[float] = []
