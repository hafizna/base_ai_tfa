"""One chronology for a complete recording and its incident representation.

A contact returning proves closure; a subsequent refault/SOTF/reopen proves
restoration did not hold. Neither observation identifies a physical cause.
"""
import re

import numpy as np


def record_sequence(payload, window):
    time_ms = np.asarray(payload.get("time", []), dtype=float) * 1000
    episodes = window.fault_episodes
    closes = [float(e["time"]) * 1000 for e in window.reclose_events
              if e.get("time") is not None and e.get("cb_open_verified") is not False]
    sotf = []
    for channel in payload.get("status_channels", []):
        name = str(channel.get("name", ""))
        if not re.search(r"\b(?:SOTF|TOR)\b|TRIP\s+ON\s+RECLOSE", name.upper()):
            continue
        samples = np.asarray(channel.get("samples", []), dtype=int)
        for idx in np.flatnonzero(np.diff(samples) > 0) + 1:
            end = next((k for k in range(idx + 1, min(len(samples), len(time_ms))) if samples[k] == 0), len(time_ms) - 1)
            if idx < len(time_ms) and time_ms[end] - time_ms[idx] >= 5:
                at = float(time_ms[idx])
                if any(close - 1 <= at <= close + 1000 for close in closes):
                    sotf.append({"time_ms": at, "channel": name})
    refault = [e for e in episodes[1:] if e.get("after_reclose")]
    failed = bool(sotf or refault or any(e.get("success") is False and e.get("cb_open_verified") is not False for e in window.reclose_events))
    outcome = "failed" if failed else "successful" if closes and any(e.get("success") is True for e in window.reclose_events) else "unknown"
    timeline = []
    for e in episodes:
        timeline.append({"kind": "refault" if e.get("after_reclose") else "fault_inception",
                         "time_ms": e["inception_time_ms"], "episode_index": e["episode_index"]})
        if e.get("clearing_time_ms") is not None:
            timeline.append({"kind": "fault_cleared", "time_ms": e["clearing_time_ms"], "episode_index": e["episode_index"]})
    timeline += [{"kind": "breaker_reclosed", "time_ms": t} for t in closes]
    timeline += [{"kind": "sotf_trip", **e} for e in sotf]
    timeline.sort(key=lambda e: e["time_ms"])
    return {
        "episode_count": len(episodes), "mechanical_close_confirmed": bool(closes),
        "restoration_outcome": outcome, "refault_after_reclose": bool(refault),
        "sotf_after_reclose": bool(sotf), "reclose_times_ms": closes,
        "sotf_trips": sotf, "timeline": timeline,
        "interpretation": "RECLOSE_REFAULT_SOTF" if sotf and refault else "RECLOSE_REFAULT" if refault
            else "RECLOSE_FAILED" if failed else "RECLOSE_SUCCESSFUL" if outcome == "successful" else "FAULT_EVENT",
    }
