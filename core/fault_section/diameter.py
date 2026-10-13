"""1½-breaker diameter template.

Lays out one diameter — Bus A, CB A, T-node A, CB AB, T-node B, CB B, Bus B —
with each breaker's CT on the side the case gives. Where the side is not known
(no single-line diagram), every side is laid out as its own variant and the
records decide between them. Around the diameter:

- the bay on each T-node: a line, with its far-end CT, station and breaker,
  or an unrecorded feeder taken as a source behind an always-closed breaker;
- the other bays of each busbar, each as CT, stub and breaker in front of a
  source, so the busbar zone has its full boundary and each bay's breaker
  opens on its own; a busbar with no bays listed gets one source;
- the zones: busbar differential (end-zone logic on, off or unknown — unknown
  is laid out both ways), the CCP of a bay with a line (bounded by the CTs of
  its two breakers and its line CT) and the line differential.

Section and device ids are fixed (``bus_b``, ``stub_cb_b``, ``t_b``,
``ct_cb_b``, ``ct_line_b`` ...) so a case maps records onto them by id, and the
same id means the same place in every variant.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping

from .topology import BREAKER, CT, Topology, Zone

_SIDES = {"cb_a": ("bus", "t"), "cb_ab": ("t_a", "t_b"), "cb_b": ("bus", "t")}
_UNKNOWN = "unknown"


def _labels(spec: Mapping[str, Any]) -> dict[str, str]:
    labels = {"bus_a": "Bus A", "bus_b": "Bus B", "cb_a": "CB A", "cb_ab": "CB AB", "cb_b": "CB B"}
    labels.update(spec.get("labels") or {})
    return labels


def _chain(topo: Topology, left: str, right: str, ct_first: bool, breaker: str, labels: dict[str, str]) -> tuple[str, str]:
    """Lay ``left — device — stub — device — right`` for ``breaker`` and its
    CT, the CT next to ``left`` when ``ct_first``. Returns the sections on the
    CT's left and right side."""
    stub = topo.add_section(f"stub_{breaker}", f"antara CT dan {labels[breaker]}")
    ct, ct_label = f"ct_{breaker}", f"CT {labels[breaker]}"
    if ct_first:
        topo.add_device(ct, CT, left, stub, ct_label)
        topo.add_device(breaker, BREAKER, stub, right, labels[breaker])
        return left, stub
    topo.add_device(breaker, BREAKER, left, stub, labels[breaker])
    topo.add_device(ct, CT, stub, right, ct_label)
    return stub, right


def _bay(topo: Topology, key: str, node: str, bay: Mapping[str, Any]) -> None:
    label = bay.get("label") or f"bay {key.upper()}"
    if bay.get("kind") == "line":
        remote = bay.get("remote") or "ujung lawan"
        line = topo.add_section(f"line_{key}", label)
        topo.add_device(f"ct_line_{key}", CT, node, line, f"CT line {label}")
        far = topo.add_section(f"remote_{key}", f"{remote} (di belakang CT line ujung lawan)")
        topo.add_device(f"ct_line_{key}_remote", CT, line, far, f"CT line {label} di {remote}")
        grid = topo.add_section(f"grid_remote_{key}", f"luar: sistem di belakang {remote}", source=True)
        topo.add_device(f"cb_remote_{key}", BREAKER, far, grid, f"CB {label} di {remote}")
        return
    grid = topo.add_section(f"grid_feeder_{key}", f"luar: {label}", source=True)
    topo.add_device(f"cb_feeder_{key}", BREAKER, node, grid, f"CB {label} (tidak terekam)")


def _bus_bays(topo: Topology, key: str, bus: str, bays: list[Mapping[str, Any]], bus_label: str) -> list[tuple[str, str]]:
    """Other bays of a busbar; returns their (CT, breaker) pairs."""
    if not bays:
        grid = topo.add_section(f"grid_bus_{key}", f"luar: bay lain di {bus_label}", source=True)
        topo.add_device(f"cb_grid_bus_{key}", BREAKER, bus, grid, f"CB bay lain di {bus_label} (tidak terekam)")
        return []
    pairs = []
    for bay in bays:
        bay_id, label = bay["id"], bay.get("label") or bay["id"]
        stub = topo.add_section(f"stub_{bay_id}", f"stub {label}", candidate=False)
        topo.add_device(f"ct_{bay_id}", CT, bus, stub, f"CT {label}")
        grid = topo.add_section(f"grid_{bay_id}", f"luar: {label}", source=True)
        topo.add_device(f"cb_{bay_id}", BREAKER, stub, grid, f"CB {label}")
        pairs.append((f"ct_{bay_id}", f"cb_{bay_id}"))
    return pairs


def build_diameter(spec: Mapping[str, Any]) -> list[Topology]:
    """Every topology variant the spec leaves open, each with its assumptions
    spelled out."""
    labels = _labels(spec)
    ct_side = {key: (spec.get("ct_side") or {}).get(key, _UNKNOWN) for key in _SIDES}
    for key, side in ct_side.items():
        if side != _UNKNOWN and side not in _SIDES[key]:
            raise ValueError(f"ct_side.{key} must be one of {_SIDES[key]} or {_UNKNOWN!r}, not {side!r}")
    zones = list(spec.get("zones") or [])
    end_zone_open = [z["id"] for z in zones if z.get("kind") == "busbar" and z.get("end_zone", _UNKNOWN) == _UNKNOWN]

    side_choices = [_SIDES[key] if side == _UNKNOWN else (side,) for key, side in ct_side.items()]
    variants = []
    for sides in product(*side_choices):
        chosen = dict(zip(_SIDES, sides))
        for end_zone_flags in product((True, False), repeat=len(end_zone_open)):
            end_zone = dict(zip(end_zone_open, end_zone_flags))
            variants.append(_build(spec, labels, chosen, end_zone))
    return variants


def _side_text(key: str, side: str, labels: dict[str, str]) -> str:
    where = {"bus": labels["bus_a" if key == "cb_a" else "bus_b"],
             "t": "T-node A" if key == "cb_a" else "T-node B",
             "t_a": "T-node A", "t_b": "T-node B"}[side]
    return f"CT {labels[key]} di sisi {where}"


def _build(spec: Mapping[str, Any], labels: dict[str, str], sides: dict[str, str],
           end_zone: dict[str, bool]) -> Topology:
    topo = Topology()
    bays = {key: spec.get(f"bay_{key}") or {"kind": "feeder"} for key in ("a", "b")}
    for key in ("a", "b"):
        topo.add_section(f"bus_{key}", labels[f"bus_{key}"])
        bay_label = bays[key].get("label")
        topo.add_section(f"t_{key}", f"T-node {key.upper()}" + (f" ({bay_label})" if bay_label else ""))

    ct_a = _chain(topo, "bus_a", "t_a", sides["cb_a"] == "bus", "cb_a", labels)
    ct_ab = _chain(topo, "t_a", "t_b", sides["cb_ab"] == "t_a", "cb_ab", labels)
    ct_b = _chain(topo, "t_b", "bus_b", sides["cb_b"] == "t", "cb_b", labels)

    for key in ("a", "b"):
        _bay(topo, key, f"t_{key}", bays[key])
    bus_pairs = {key: _bus_bays(topo, key, f"bus_{key}", list(spec.get(f"bus_{key}_bays") or []), labels[f"bus_{key}"])
                 for key in ("a", "b")}

    # Zone side of each diameter CT: (towards the bus, towards the T-node).
    toward = {
        "cb_a": (ct_a[0], ct_a[1]),
        "cb_b": (ct_b[1], ct_b[0]),
    }
    assumptions = [_side_text(key, sides[key], labels) for key in _SIDES]
    for zone in spec.get("zones") or []:
        kind, zid, zlabel = zone.get("kind"), zone["id"], zone.get("label") or zone["id"]
        if kind == "busbar":
            key = zone.get("bus", "b")
            cb = f"cb_{key}"
            boundary = [(f"ct_{cb}", toward[cb][0])] + [(ct, f"bus_{key}") for ct, _ in bus_pairs[key]]
            enabled = end_zone.get(zid, zone.get("end_zone") is True)
            pairs = [(f"ct_{cb}", cb)] + list(bus_pairs[key])
            topo.add_zone(Zone(zid, zlabel, tuple(boundary), tuple(pairs) if enabled else ()))
            if zid in end_zone:
                assumptions.append(f"{zlabel}: end-zone {'aktif' if enabled else 'tidak aktif'}")
        elif kind == "ccp":
            key = zone.get("bay", "b")
            if bays[key].get("kind") != "line":
                raise ValueError(f"zone {zid}: CCP needs a line bay on T-node {key.upper()}")
            node = f"t_{key}"
            ab_inside = ct_ab[1] if key == "b" else ct_ab[0]
            boundary = ((f"ct_cb_{key}", toward[f"cb_{key}"][1]), ("ct_cb_ab", ab_inside), (f"ct_line_{key}", node))
            topo.add_zone(Zone(zid, zlabel, boundary))
        elif kind == "line_diff":
            key = zone.get("bay", "b")
            line = f"line_{key}"
            topo.add_zone(Zone(zid, zlabel, ((f"ct_line_{key}", line), (f"ct_line_{key}_remote", line))))
        else:
            raise ValueError(f"zone {zid}: unknown kind {kind!r}")
    topo.assumptions = tuple(assumptions)
    return topo
