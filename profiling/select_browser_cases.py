"""Select real local CFG/DAT pairs near 1/5/10 MB without copying raw data."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tmp/browser-validation/cases.json"))
    args = parser.parse_args()
    root = str(args.raw_dir.resolve())
    if os.name == "nt" and not root.startswith("\\\\?\\"):
        root = "\\\\?\\" + root
    candidates = []
    for directory, _, names in os.walk(root):
        lookup = {name.casefold(): name for name in names}
        for name in names:
            if not name.casefold().endswith(".cfg"):
                continue
            dat_name = lookup.get(name[:-4].casefold() + ".dat")
            if dat_name:
                cfg, dat = Path(directory) / name, Path(directory) / dat_name
                candidates.append({"cfg": str(cfg), "dat": str(dat),
                                   "raw_bytes": cfg.stat().st_size + dat.stat().st_size})
    if len(candidates) < 3:
        raise SystemExit("At least three complete CFG/DAT pairs are required")
    selected = []
    used = set()
    for target in (1_000_000, 5_000_000, 10_000_000):
        case = min((item for item in candidates if item["cfg"] not in used),
                   key=lambda item: abs(item["raw_bytes"] - target))
        selected.append({"target_bytes": target, **case})
        used.add(case["cfg"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps({"pair_count": len(candidates), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
