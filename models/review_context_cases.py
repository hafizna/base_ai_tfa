"""Export evidence for human context review; never generates training labels.

Legacy CSV columns select candidates only. Conclusions and channel edges are
recomputed from the original recording using the production reader.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from core.comtrade_parser import parse_comtrade
from models.build_dataset import identity, pipeline_fingerprint
from webapp.api.record_analysis import build_record_analysis
from webapp.api.record_payload import record_to_payload


def review(path):
    record = parse_comtrade(str(path))
    if record is None:
        raise ValueError("COMTRADE reader could not parse this recording")
    payload = record_to_payload(record)
    analysis = build_record_analysis(identity(path), payload)
    time = np.asarray(payload["time"]) * 1000
    channels = []
    for ch in payload.get("status_channels", []):
        samples = np.asarray(ch["samples"], dtype=int)
        channels.append({"name": ch["name"], "initial": int(samples[0]),
                         "on_ms": time[np.flatnonzero(np.diff(samples) > 0) + 1].tolist(),
                         "off_ms": time[np.flatnonzero(np.diff(samples) < 0) + 1].tolist()})
    return {"cfg_path": str(path), "record_fingerprint": identity(path),
            "pipeline_fingerprint": pipeline_fingerprint(),
            "record_end_ms": float(time[-1]), "channels": channels,
            "analysis": analysis.to_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("data/features/labeled_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/training-runs/context-review.jsonl"))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--path", type=Path, action="append", help="Review specific recordings instead of candidate selection")
    parser.add_argument("--all", action="store_true", help="Review every row of a deliberately selected inventory")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.inventory.open(encoding="utf-8-sig")))
    candidates = [r for r in rows if "RWALO" in r["cfg_path"].upper() or "RAWALO" in r["cfg_path"].upper()]
    candidates += [r for r in rows if r.get("reclose_successful") == "True" and r.get("teleprotection_rx") == "True"]
    if args.path:
        candidates = [{"cfg_path": str(path)} for path in args.path]
    elif args.all:
        candidates = rows
    seen = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in candidates:
            path = Path(row["cfg_path"])
            if not path.exists():
                continue
            fingerprint = identity(path)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            try:
                result = review(path)
                seq = result["analysis"]["event_window"].get("sequence") or {}
                print(len(seen), path.name, seq.get("interpretation"), flush=True)
            except Exception as exc:
                result = {"cfg_path": str(path), "error": str(exc)}
                print(len(seen), path.name, "ERROR", str(exc), flush=True)
            output.write(json.dumps(result) + "\n")
            output.flush()
            if len(seen) >= args.limit:
                break


if __name__ == "__main__":
    main()
