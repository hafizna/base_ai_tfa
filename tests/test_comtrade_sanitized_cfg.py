from core.comtrade_parser import parse_comtrade

_BODY = """2,1A,1D
1,ia,,,A,1.0,0.0,0,-32767,32767,1,1,P
1,trip,,,0
50
1
1000,4
{date},11:34:42.609000
{date},11:34:42.849000
ASCII
1
"""
_DAT = "1,0,10,0\n2,1000,20,1\n3,2000,30,0\n4,3000,40,0\n"


def _write(tmp_path, header, date):
    cfg = tmp_path / "rec.cfg"
    cfg.write_text(header + "\n" + _BODY.format(date=date), encoding="utf-8")
    (tmp_path / "rec.dat").write_text(_DAT, encoding="utf-8")
    return parse_comtrade(str(cfg))


def test_nari_header_with_trailing_comma_keeps_its_day_first_date(tmp_path):
    # "NR,LINE_DISTANCE_RELAY,1999," (NARI PCS-902): the 1999 revision is
    # read day-first, so 26/05/2025 must not be swapped.
    record = _write(tmp_path, "NR,LINE_DISTANCE_RELAY,1999,", "26/05/2025")

    assert record is not None
    assert record.start_time_iso.startswith("2025-05-26T11:34:42.609")
    assert abs(record.trigger_time - 0.240) < 1e-6


def test_month_first_date_in_a_1999_file_is_swapped(tmp_path):
    record = _write(tmp_path, "NR,LINE_DISTANCE_RELAY,1999,", "05/26/2025")

    assert record is not None
    assert record.start_time_iso.startswith("2025-05-26")


def test_day_first_date_in_a_pre_1999_header_is_swapped(tmp_path):
    record = _write(tmp_path, "STATION,DEVICE,1991,extra", "26/05/2025")

    assert record is not None
    assert record.start_time_iso.startswith("2025-05-26")
