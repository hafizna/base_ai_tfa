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


class LocusResponse(BaseModel):
    loop: str
    points: List[LocusPoint]
    zones: List[ZoneConfig]
    fault_inception_idx: Optional[int]


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
    cause_ranking: List[Dict[str, Any]]   # [{cause, label_id, confidence}]
    fault_type: str                        # "transient" | "permanent"
    overall_confidence: float
    evidence: List[str]


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


class DiffRestraintRequest(BaseModel):
    comtrade: ComtradeOut
    params: CharacteristicParams = CharacteristicParams()
    relay_type: str = "87L"     # "87L" | "87T"


class DiffRestraintAnalysisRequest(AnalysisRequestBase):
    params: CharacteristicParams = CharacteristicParams()
    relay_type: str = "87L"


class DiffRestraintResponse(BaseModel):
    samples: List[DiffRestraintSample]
    params: CharacteristicParams
    operated_status: str    # "NOT_OPERATED" | "IDIFF_OPERATED" | "IDIFF_FAST_OPERATED"
    operated_phases: List[str]


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
