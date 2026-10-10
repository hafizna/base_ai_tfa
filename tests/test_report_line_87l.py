"""F5.8 in the PDF: a line record's report prints the 87L summary only when
the record carries 87L evidence, as the line workspace shows its panels."""

from tests.test_relay_87l_diff_restraint import _payload
from webapp.api.routers.report import _build_relay_specific_section, _build_styles


def test_a_line_record_without_87l_evidence_prints_no_87l_summary():
    # Local currents only: a "differential" from them is just the local current.
    payload = _payload(include_diff_trip=False)
    payload["analog_channels"] = payload["analog_channels"][:3]
    assert _build_relay_specific_section(_build_styles(), payload, "LINE", None) == []


def test_a_line_record_with_87l_evidence_keeps_its_summary():
    assert _build_relay_specific_section(_build_styles(), _payload(), "LINE", None)
