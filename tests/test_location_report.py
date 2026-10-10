"""Fault-location reports (TWS and DE-FL): what page 1 says, from real
builders on synthetic inputs."""

import io

from pypdf import PdfReader

from tests.test_relay_21_double_ended import _B_C, _fault_payloads
from webapp.api.routers.location_report import (
    DoubleEndedReportRequest,
    _clip,
    _num,
    build_defl_pdf,
    build_tws_pdf,
)


def _text(pdf: bytes) -> str:
    """The report's text with line wraps undone."""
    assert pdf.startswith(b"%PDF")
    return " ".join(" ".join(page.extract_text().split()) for page in PdfReader(io.BytesIO(pdf)).pages)


def _tws_payload() -> dict:
    def endpoint(role: str, station: str, feeder: str, record: int, km: float) -> dict:
        return {
            "role": role, "station_display_name": station, "feeder_display_name": feeder,
            "record_number": record, "gps_locked": True, "fault_distance_km": km, "sample_rate_hz": 1_250_000.0,
        }

    return {
        "source_type": "tws_cdb",
        "source_file": "sample.cdb",
        "station_name": "PMPEK-SMDRA_1",
        "results": [{
            "circuit_name": "PMPEK-SMDRA_1",
            "line_length_km": 30.7,
            "velocity_factor": 99.5,
            "velocity_km_s": 298293.5,
            "sample_distance_km": 0.2386,
            "result_time_local": 1775396788.1922,
            "endpoints": [
                endpoint("X", "GI SUMADRA", "PAMEUNGPEUK 1", 39362, 14.36862),
                endpoint("Y", "GI PAMENGPEUK", "SUMADRA 1", 48734, 16.33138),
            ],
            "sel_type_d": {
                "delta_t_us": -10.0136, "velocity_km_s": 298293.5, "line_length_km": 30.7,
                "m_from_x_km": 13.8565, "m_from_y_km": 16.8435, "qualitrol_x_km": 14.36862,
                "qualitrol_y_km": 16.33138, "delta_x_km": -0.5121, "delta_y_km": 0.5121,
            },
        }],
    }


def test_numbers_print_the_indonesian_way():
    assert _num(-10.0136, 2, "µs") == "−10,01 µs"
    assert _num(298293.5, 0, "km/s") == "298 294 km/s"
    assert _num(512, 0, "m", signed=True) == "+512 m"
    assert _num(None) == "—"


def test_a_comtrade_file_name_shortens_to_its_device():
    assert _clip("230821,081503670,+7h0,GI MOJOSONGO,BRINGIN 1-2,Qualitrol LLC") == "Qualitrol LLC"
    assert _clip("ZQ6D") == "ZQ6D"


def test_the_tws_report_names_its_method_and_process_path():
    text = _text(build_tws_pdf(_tws_payload(), [], "2783b77a" * 4))
    assert "LAPORAN LOKASI GANGGUAN" in text
    assert "BERBEDA DARI ENGINE DISTANCE" in text
    assert "Traveling wave dua ujung (Type D)" in text
    assert "JALUR PROSES" in text
    assert "14,37 km" in text and "16,33 km" in text
    assert "5 Apr 2026" in text
    assert "Konfirmasi lapangan" in text
    # The two computations side by side, with their 512 m difference.
    assert "13,857 km" in text and "−512 m" in text


def _defl_request(**overrides) -> DoubleEndedReportRequest:
    fields = dict(
        analysis_id_a="a" * 32, analysis_id_b="b" * 32, loop="ZBC", line_len_km=20.0,
        r1_ohm_per_km=0.05, x1_ohm_per_km=0.4, manual_shift_ms=0.0, record_name_a="REC-A",
        record_name_b="REC-B", shift_source="estimate", suggested_loop="ZBC",
    )
    fields.update(overrides)
    return DoubleEndedReportRequest(**fields)


def test_a_consistent_confirmed_location_is_reported_as_reliable():
    payload_a, payload_b = _fault_payloads(*_B_C)
    text = _text(build_defl_pdf(payload_a, payload_b, _defl_request(sync_confirmed=True)))
    assert "Impedansi dua ujung" in text
    assert "Hasil belum andal" not in text
    assert "8,00 km" in text  # 40% of 20 km
    assert "Tidak lolos" not in text and "Belum" not in text


def test_an_unconfirmed_sync_keeps_the_location_from_patrol():
    # F7.4: the residuals are fine, but nobody checked the shift on the overlay.
    payload_a, payload_b = _fault_payloads(*_B_C)
    text = _text(build_defl_pdf(payload_a, payload_b, _defl_request(sync_confirmed=False)))
    assert "Hasil belum andal" in text
    assert "LOKASI GANGGUAN (BELUM ANDAL)" in text
    assert "belum dikonfirmasi" in text


def test_a_loop_the_fault_draws_no_current_in_is_refused():
    # An R-N fault solved on the S-T loop: the fault draws no current in it.
    payload_a, payload_b = _fault_payloads({"A": 2000.0}, {"A": 1500.0})
    text = _text(build_defl_pdf(payload_a, payload_b, _defl_request(sync_confirmed=True, suggested_loop="ZA")))
    assert "Hasil belum andal" in text
    assert "Pilih loop fasa yang terganggu" in text
    assert "langkah 4 menyarankan R-N" in text
