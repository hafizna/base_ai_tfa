"""Time-zone offsets between two recorders' clocks.

A recorder whose clock differs from another's by a whole number of quarter
hours plus well under a second was set to another time zone — typically UTC
against WIB, as in Qualitrol files named "...,+7h0,...". Two recorders that
saw one fault never disagree by more than that remainder, so the zone part is
removed before their clocks are compared.
"""

from __future__ import annotations

from typing import Optional

TIME_ZONE_STEP_S = 900.0
MAX_CLOCK_REMAINDER_S = 1.0
# Civil time zones run from UTC-12 to UTC+14, so two clocks set to different
# zones are never more than 26 h apart; a larger whole number of quarter hours
# is a wrong date or another event, not a time zone.
MAX_TIME_ZONE_OFFSET_S = 26 * 3600.0


def split_clock_offset(raw_s: float) -> tuple[float, Optional[float]]:
    """``(remainder_s, zone_offset_s)``. The zone offset is None, and the
    remainder is ``raw_s`` itself, when ``raw_s`` is not a whole number of
    quarter hours (at most ``MAX_TIME_ZONE_OFFSET_S``) plus at most
    ``MAX_CLOCK_REMAINDER_S``."""
    zone_offset_s = round(raw_s / TIME_ZONE_STEP_S) * TIME_ZONE_STEP_S
    remainder_s = raw_s - zone_offset_s
    if 0.0 < abs(zone_offset_s) <= MAX_TIME_ZONE_OFFSET_S and abs(remainder_s) <= MAX_CLOCK_REMAINDER_S:
        return remainder_s, zone_offset_s
    return raw_s, None
