from datetime import datetime

import pytest

from core.micom_evt_parser import parse_micom_evt, signal_timelines

_HEADER = b"APP: Courier\r\nTYPE: EventList\r\nFORMAT: 1.0\r\nNEXTEXTENDED: 000003\r\n\r\n\r\n\x1a\x00J\x00"


def _lp(raw: bytes) -> bytes:
    return bytes([len(raw)]) + raw


def _cp56(when: datetime) -> bytes:
    ms = when.second * 1000 + when.microsecond // 1000
    return bytes([ms & 0xFF, ms >> 8, when.minute, when.hour, when.day | (when.isoweekday() << 5),
                  when.month, when.year - 2000])


def _record(when: datetime, text: str, value: int, *, labels=None, size: int = 4) -> bytes:
    raw = value.to_bytes(size, "little")
    record = (b"<\x07" + _cp56(when) + b"\x00\x00\x01\x01"
              + _lp(b"MiCOM") + _lp(b"MiCOM P841 B") + _lp(b"P841916A6M0470K")
              + b"\x37\x0f" + _lp(text.encode()) + b"\x20" + _lp(raw) + b"\x00\x01\x02\x00\x00\x00\x00")
    if labels:
        record += (_lp(f"Event Value      %.{len(labels)}b".encode()) + b"\x20" + _lp(raw)
                   + bytes([len(labels)]) + b"".join(_lp(label.ljust(16).encode()) for label in labels))
    return record


def _file(tmp_path, *records):
    path = tmp_path / "events.evt"
    path.write_bytes(_HEADER + b"".join(records))
    return path


LABELS = ["L1 TEST MODE", "L2 INIT 3PH", "L3 CB OPEN", "L4 INIT PH R"]


def test_word_bits_are_read_by_label_as_first_state_then_changes(tmp_path):
    path = _file(tmp_path,
                 _record(datetime(2025, 5, 26, 11, 34, 42, 916000), "Logic Inputs 1", 0b0111, labels=LABELS, size=2),
                 _record(datetime(2025, 5, 26, 11, 34, 42, 877000), "Logic Inputs 1", 0b0011, labels=LABELS, size=2))

    events = parse_micom_evt(path)
    timelines = signal_timelines(events)

    assert [e.time for e in events] == sorted(e.time for e in events)
    assert events[0].identity == ("MiCOM", "MiCOM P841 B", "P841916A6M0470K")
    assert timelines["L3 CB OPEN"] == [(datetime(2025, 5, 26, 11, 34, 42, 877000), False),
                                       (datetime(2025, 5, 26, 11, 34, 42, 916000), True)]
    assert timelines["L2 INIT 3PH"] == [(datetime(2025, 5, 26, 11, 34, 42, 877000), True)]


def test_on_off_events_become_signal_transitions(tmp_path):
    path = _file(tmp_path,
                 _record(datetime(2025, 5, 26, 11, 34, 43, 31000), "Bfail1 Trip 3ph   ON", 4),
                 _record(datetime(2025, 5, 26, 11, 34, 43, 91000), "Bfail1 Trip 3ph   OFF", 0))

    events = parse_micom_evt(path)

    assert events[0].text == "Bfail1 Trip 3ph ON"
    assert events[0].state() == ("Bfail1 Trip 3ph", True)
    assert signal_timelines(events)["Bfail1 Trip 3ph"][-1] == (datetime(2025, 5, 26, 11, 34, 43, 91000), False)


def test_rejects_a_file_that_is_not_a_courier_event_list(tmp_path):
    path = tmp_path / "other.evt"
    path.write_bytes(b"not an event list")

    with pytest.raises(ValueError):
        parse_micom_evt(path)
