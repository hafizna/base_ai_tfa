"""Transformer bay template (e.g. 150/20 kV).

Lays out one transformer bay from the HV busbar to the LV busbar:

    Bus HV — CB HV / CT HV — HV terminal (GIS, cable, bushing) — HV winding
        ═ magnetic coupling ═
    LV winding — LV terminal (bushing, power cable) — CT LV / CB LV — Bus LV — feeders

with each breaker's CT on the side the case gives (both tried when unknown),
the grounded neutrals each behind a neutral CT, the HV grid as a source, the
LV feeders as load, and — when the case does not know — both with and without
another source on the LV busbar (a parallel transformer through the bus
coupler, embedded generation). That last one matters: an LV earth fault inside
the transformer zone drives fault current through the LV CT only when the LV
busbar can feed it.

Zones, from the case:

- ``transformer_diff`` (87T): between CT HV and CT LV. Its zero-sequence
  filtering is modelled by leaving the neutral CTs out of its boundary, so
  current fed only from a neutral is not differential current.
- ``ref`` (REF HV / REF LV): a winding's phase CT and neutral CT, ending at the
  magnetic coupling; earth faults only.
- ``mechanical`` (Buchholz, sudden pressure): the inside of the tank.

Section and device ids are fixed (``hv_terminal``, ``lv_winding``, ``ct_lv``,
``ct_lv_n`` ...) so a case maps records onto them by id.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping

from .topology import BREAKER, CT, LINK, Topology, Zone

_UNKNOWN = "unknown"


def _labels(spec: Mapping[str, Any]) -> dict[str, str]:
    labels = {"bus_hv": "Bus HV", "bus_lv": "Bus LV", "cb_hv": "CB HV", "cb_lv": "CB LV (incoming)",
              "trafo": "trafo"}
    labels.update(spec.get("labels") or {})
    return labels


def build_transformer(spec: Mapping[str, Any]) -> list[Topology]:
    """Every topology variant the spec leaves open, each with its assumptions
    spelled out."""
    labels = _labels(spec)
    sides = {key: (spec.get("ct_side") or {}).get(key, _UNKNOWN) for key in ("cb_hv", "cb_lv")}
    for key, side in sides.items():
        if side not in ("bus", "trafo", _UNKNOWN):
            raise ValueError(f"ct_side.{key} must be 'bus', 'trafo' or {_UNKNOWN!r}, not {side!r}")
    lv_source = spec.get("lv_source", _UNKNOWN)
    choices = [("bus", "trafo") if side == _UNKNOWN else (side,) for side in sides.values()]
    sources = (True, False) if lv_source == _UNKNOWN else (bool(lv_source),)
    variants = []
    for (hv_side, lv_side), with_source in product(product(*choices), sources):
        variants.append(_build(spec, labels, hv_side, lv_side, with_source, lv_source == _UNKNOWN))
    return variants


def _build(spec: Mapping[str, Any], labels: dict[str, str], hv_side: str, lv_side: str,
           lv_source: bool, lv_source_open: bool) -> Topology:
    topo = Topology()
    trafo = labels["trafo"]
    topo.add_section("bus_hv", labels["bus_hv"])
    topo.add_section("grid_hv", f"luar: sistem di belakang {labels['bus_hv']}", source=True)
    topo.add_device("cb_grid_hv", BREAKER, "grid_hv", "bus_hv", f"bay lain di {labels['bus_hv']} (tidak terekam)")
    topo.add_section("hv_terminal", f"sisi HV {trafo} antara CT HV dan belitan (GIS/kabel/bushing HV)")
    topo.add_section("hv_winding", f"belitan HV {trafo} (di dalam tangki)")
    topo.add_section("lv_winding", f"belitan LV {trafo} (di dalam tangki)")
    topo.add_section("lv_terminal", f"sisi LV {trafo} antara belitan dan CT LV (bushing LV, kabel daya)")
    topo.add_section("bus_lv", labels["bus_lv"])
    topo.add_section("feeders", f"luar: penyulang di {labels['bus_lv']}")
    topo.add_device("cb_feeders", BREAKER, "bus_lv", "feeders", "CB penyulang (tidak terekam)")

    toward_trafo = {}
    for key, bus, terminal, side in (("cb_hv", "bus_hv", "hv_terminal", hv_side), ("cb_lv", "bus_lv", "lv_terminal", lv_side)):
        stub = topo.add_section(f"stub_{key}", f"antara CT dan {labels[key]}")
        ct = "ct_hv" if key == "cb_hv" else "ct_lv"
        ct_label = f"CT {'HV' if key == 'cb_hv' else 'LV'} {trafo}"
        if side == "bus":                   # bus — CT — stub — CB — terminal
            topo.add_device(ct, CT, bus, stub, ct_label)
            topo.add_device(key, BREAKER, stub, terminal, labels[key])
            toward_trafo[ct] = stub
        else:                               # bus — CB — stub — CT — terminal
            topo.add_device(key, BREAKER, bus, stub, labels[key])
            topo.add_device(ct, CT, stub, terminal, ct_label)
            toward_trafo[ct] = terminal

    topo.add_device("bushing_hv", LINK, "hv_terminal", "hv_winding", f"bushing HV {trafo}")
    topo.add_device("coupling", LINK, "hv_winding", "lv_winding", f"kopling magnetik {trafo}")
    topo.add_device("bushing_lv", LINK, "lv_winding", "lv_terminal", f"bushing LV {trafo}")
    for side, winding in (("hv", "hv_winding"), ("lv", "lv_winding")):
        if spec.get(f"{side}_neutral", True):
            ground = topo.add_section(f"ground_{side}", f"tanah netral {side.upper()}", ground=True)
            topo.add_device(f"ct_{side}_n", CT, winding, ground, f"CT netral {side.upper()} {trafo}")
    if lv_source:
        source = topo.add_section("lv_source", f"luar: sumber lain di {labels['bus_lv']} (trafo paralel/kopel)",
                                  source=True)
        topo.add_device("cb_lv_source", BREAKER, "bus_lv", source, "kopel / sumber lain (tidak terekam)")

    for zone in spec.get("zones") or []:
        kind, zid, label = zone.get("kind"), zone["id"], zone.get("label") or zone["id"]
        if kind == "transformer_diff":
            topo.add_zone(Zone(zid, label, (("ct_hv", toward_trafo["ct_hv"]), ("ct_lv", toward_trafo["ct_lv"]))))
        elif kind == "ref":
            side = zone.get("side", "lv")
            ct, neutral = ("ct_hv", "ct_hv_n") if side == "hv" else ("ct_lv", "ct_lv_n")
            if neutral not in topo.devices:
                raise ValueError(f"zone {zid}: REF {side.upper()} needs the {side.upper()} neutral")
            topo.add_zone(Zone(zid, label, ((ct, toward_trafo[ct]), (neutral, f"{side}_winding")),
                               cuts=("coupling",), earth_only=True))
        elif kind == "mechanical":
            topo.add_zone(Zone(zid, label, fixed=("hv_winding", "lv_winding")))
        else:
            raise ValueError(f"zone {zid}: unknown kind {kind!r}")

    where = {"bus": "sisi bus", "trafo": "sisi trafo"}
    assumptions = [f"CT HV di {where[hv_side]} {labels['cb_hv']}", f"CT LV di {where[lv_side]} {labels['cb_lv']}"]
    if lv_source_open:
        assumptions.append(f"{'ada' if lv_source else 'tidak ada'} sumber lain di {labels['bus_lv']}")
    topo.assumptions = tuple(assumptions)
    return topo
