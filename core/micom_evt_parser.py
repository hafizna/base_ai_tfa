"""MiCOM S1 Agile event files (``.evt``: "APP: Courier / TYPE: EventList").

S1 Agile saves a relay's event log as a short text header followed by binary
Courier event records. A record carries:

- its time as IEC 60870-5 CP56Time2a (7 bytes: milliseconds little-endian,
  minute, hour, day of month | day of week << 5, month, year - 2000), right
  behind the two-byte marker ``<\\x07``;
- three length-prefixed identity strings as set in the relay: description,
  plant reference and model number ("MiCOM", "MiCOM P841 B",
  "P841916A6M0470K" on one relay; "BUS PRO B PH S", "GI PEMALANG",
  "P746916K6M0120M" on another);
- a two-byte Courier cell reference, the length-prefixed event text
  ("Any Start        ON"), a type byte and the length-prefixed value;
- for a logic-input or output-contact word, a format string such as
  ``"Event Value      %.32lb"`` followed by the value again, the number of
  bits and the 16-character label of every bit ("L3 CB OPEN"), so a change of
  a binary input can be read by name rather than by bit position.

Only that much is decoded. Times are returned on the relay's own clock: no
time zone or offset is applied, because the relay's clock is exactly what an
incident's time alignment has to correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from core.comtrade_parser import _windows_long_path

_RECORD_MARK = b"<\x07"
_FORMAT_ANCHOR = b"Event Value"
_STATE_SUFFIX = re.compile(r"^(?P<name>.*?)\s+(?P<state>ON|OFF)$")


@dataclass(frozen=True)
class MicomEvent:
    time: datetime
    cell: str                   # Courier cell reference, hex ("370f")
    text: str                   # whitespace-collapsed event text ("Any Start ON")
    value: Optional[int]
    labels: tuple[str, ...] = ()  # bit labels of a logic-input / output word, bit 0 first
    identity: tuple[str, ...] = ()  # description, plant reference, model number

    def state(self) -> Optional[tuple[str, bool]]:
        """``(signal name, state)`` of an ON/OFF event ("Bfail1 Trip 3ph ON"),
        else None."""
        match = _STATE_SUFFIX.match(self.text)
        if not match:
            return None
        return match.group("name"), match.group("state") == "ON"


def _cp56time2a(raw: bytes) -> Optional[datetime]:
    if len(raw) < 7:
        return None
    ms = raw[0] | (raw[1] << 8)
    try:
        return datetime(2000 + (raw[6] & 0x7F), raw[5] & 0x0F, raw[4] & 0x1F,
                        raw[3] & 0x1F, raw[2] & 0x3F) + timedelta(milliseconds=ms)
    except ValueError:
        return None


def _text(raw: bytes) -> str:
    return " ".join(raw.decode("latin-1").split())


def _bit_labels(record: bytes, start: int) -> tuple[str, ...]:
    anchor = record.find(_FORMAT_ANCHOR, start)
    if anchor < 1:
        return ()
    fmt_len = record[anchor - 1]
    fmt = record[anchor:anchor + fmt_len].decode("latin-1").rstrip()
    if not fmt.endswith("b"):
        return ()
    pos = anchor + fmt_len + 1          # skip the type byte
    if pos >= len(record):
        return ()
    pos += 1 + record[pos]              # skip the repeated value
    if pos >= len(record):
        return ()
    count = record[pos]
    pos += 1
    labels = []
    for _ in range(count):
        if pos >= len(record):
            return ()
        size = record[pos]
        raw = record[pos + 1:pos + 1 + size]
        if len(raw) != size or any(b < 0x20 or b > 0x7E for b in raw):
            return ()
        labels.append(_text(raw))
        pos += 1 + size
    return tuple(labels)


def _parse_record(record: bytes) -> Optional[MicomEvent]:
    time = _cp56time2a(record[2:9])
    if time is None:
        return None
    pos = 13                            # marker, CP56Time2a, 4 bytes of flags
    strings = []
    try:
        for _ in range(3):
            size = record[pos]
            strings.append(record[pos + 1:pos + 1 + size].decode("latin-1"))
            pos += 1 + size
        cell = record[pos:pos + 2].hex()
        pos += 2
        size = record[pos]
        text = _text(record[pos + 1:pos + 1 + size])
        pos += 1 + size + 1             # text, then its type byte
        size = record[pos]
        value = int.from_bytes(record[pos + 1:pos + 1 + size], "little") if 0 < size <= 8 else None
        pos += 1 + size
    except IndexError:
        return None
    return MicomEvent(time=time, cell=cell, text=text, value=value,
                      labels=_bit_labels(record, pos), identity=tuple(s.strip() for s in strings))


def parse_micom_evt(path: str | Path) -> list[MicomEvent]:
    """Every decodable event of an S1 Agile ``.evt`` file, oldest first."""
    data = Path(_windows_long_path(path)).read_bytes()
    header = data[:256].decode("latin-1", errors="replace")
    if "Courier" not in header or "EventList" not in header:
        raise ValueError(f"{Path(path).name} is not a MiCOM S1 Courier event list")
    marks = [m.start() for m in re.finditer(re.escape(_RECORD_MARK), data)]
    events = []
    for idx, start in enumerate(marks):
        end = marks[idx + 1] if idx + 1 < len(marks) else len(data)
        event = _parse_record(data[start:end])
        if event is not None:
            events.append(event)
    events.sort(key=lambda e: e.time)
    return events


def signal_timelines(events: Iterable[MicomEvent]) -> dict[str, list[tuple[datetime, bool]]]:
    """State of every named signal over time.

    Two kinds of record feed it: an ON/OFF event ("Bfail1 Trip 3ph ON") is one
    entry for that signal; a logic-input / output word is read bit by bit
    under the bit labels ("L3 CB OPEN"). The first record of a word gives the
    state of all its bits at that time; later records of the word add an entry
    only for the bits that changed, so each list is the signal's state at its
    first appearance followed by its transitions.
    """
    timelines: dict[str, list[tuple[datetime, bool]]] = {}
    last_word: dict[str, int] = {}
    for event in events:
        if event.labels and event.value is not None:
            previous = last_word.get(event.text)
            for bit, label in enumerate(event.labels):
                state = bool(event.value >> bit & 1)
                if previous is None or bool(previous >> bit & 1) != state:
                    timelines.setdefault(label, []).append((event.time, state))
            last_word[event.text] = event.value
            continue
        named = event.state()
        if named is not None:
            name, state = named
            timelines.setdefault(name, []).append((event.time, state))
    return timelines
