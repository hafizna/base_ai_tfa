import pytest

from core.cff_parser import CffParseError, extract_cff


def test_extract_cff_cfg_and_float32_dat_sections():
    cfg = b"KETAUN,RED670_0_F01_0,2013\r\n1,0A,0D\r\n50\r\n"
    dat = b"\x01\x00\x00\x00\x00\x00\x80?"
    payload = (
        b"--- file type: CFG ---\r\n"
        + cfg
        + b"--- file type: HDR ---\r\n"
        + b"<DisturbanceRecording />\r\n"
        + b"--- file type: DAT FLOAT32: 8 ---\r\n"
        + dat
    )

    archive = extract_cff(payload)

    assert archive.cfg == cfg
    assert archive.dat == dat
    assert archive.dat_format == "FLOAT32"
    assert archive.expected_dat_size == 8
    assert archive.warnings == []


def test_extract_cff_reports_dat_size_mismatch():
    payload = (
        b"--- file type: CFG ---\n"
        b"STATION,DEVICE,2013\n"
        b"--- file type: DAT FLOAT32: 4 ---\n"
        b"\x00\x01"
    )

    archive = extract_cff(payload)

    assert archive.dat == b"\x00\x01"
    assert "size mismatch" in archive.warnings[0]


def test_extract_cff_requires_cfg_section():
    with pytest.raises(CffParseError):
        extract_cff(b"--- file type: DAT FLOAT32: 4 ---\n\x00\x00\x00\x00")
