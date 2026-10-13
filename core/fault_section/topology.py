"""Station topology for fault section estimation.

Fault section estimation answers "where in the station was the fault" from
which CTs carried fault current, which differential zones operated and which
way directional relays looked, interval by interval as breakers opened. It
needs no hand-drawn single-line diagram and no learned model: the station is a
small graph, every candidate fault location is simulated against every
interval, and the locations that contradict nothing are the answer.

The graph:

- A *section* is a stretch of conductor with no switching or measuring device
  inside it: a busbar, the stub between a CT and its breaker, the T-node of a
  1½-breaker bay, a line. A fault anywhere in a section looks the same to every
  relay, so sections are the fault candidates.
- A *device* joins two sections: a breaker conducts only while closed, a CT
  always conducts and is where current is measured, a link always conducts
  and measures nothing (a bushing, a transformer's magnetic coupling).
- A *source* section stands for the grid behind it (another bay's line or
  transformer, the far end of a line). Sources feed faults; a fault placed in
  one stands for "outside the modelled station, beyond that device". A
  *ground* section is the earth behind a grounded neutral: a source only for
  an earth fault, which is how a neutral CT comes to carry current.
- A differential *zone* is bounded by CTs, each with the section on its zone
  side, and possibly by unmeasured cuts (a REF zone ends at the transformer's
  magnetic coupling). A busbar zone may drop a CT while that bay's breaker
  reads open (end-zone logic), so the stub between them joins the busbar
  zone. A *fixed* zone covers named sections whatever the CTs (Buchholz and
  pressure relays guard the inside of the tank). An earth-only zone (REF)
  sees earth faults only.

Current flow is connectivity only: a CT carries fault current when it lies on
a path from a source to the fault through conducting devices, flowing towards
the fault. Paths stop at source sections, since the grid between two sources
is not modelled.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

BREAKER = "breaker"
CT = "ct"
LINK = "link"


@dataclass(frozen=True)
class Section:
    id: str
    label: str
    source: bool = False
    candidate: bool = True      # False for stubs the case does not model as fault locations
    ground: bool = False        # earth behind a grounded neutral: a source for earth faults only


@dataclass(frozen=True)
class Device:
    id: str
    kind: str                   # BREAKER | CT | LINK
    a: str
    b: str
    label: str

    def other(self, section: str) -> str:
        return self.b if section == self.a else self.a


@dataclass(frozen=True)
class Zone:
    id: str
    label: str
    boundary: tuple[tuple[str, str], ...] = ()      # (CT id, section on the zone side)
    end_zone: tuple[tuple[str, str], ...] = ()      # (CT id, breaker id): CT dropped while the breaker reads open
    cuts: tuple[str, ...] = ()                      # unmeasured devices the zone ends at
    fixed: tuple[str, ...] = ()                     # sections of a zone not bounded by CTs
    earth_only: bool = False                        # sees earth faults only (REF)


@dataclass
class Topology:
    sections: dict[str, Section] = field(default_factory=dict)
    devices: dict[str, Device] = field(default_factory=dict)
    zones: dict[str, Zone] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()               # what this variant takes as given, for the reader

    def add_section(self, id: str, label: str, *, source: bool = False, candidate: bool = True,
                    ground: bool = False) -> str:
        self.sections[id] = Section(id, label, source or ground, candidate and not ground, ground)
        return id

    def add_device(self, id: str, kind: str, a: str, b: str, label: str) -> str:
        if a not in self.sections or b not in self.sections:
            raise KeyError(f"device {id} joins unknown section(s) {a!r}, {b!r}")
        self.devices[id] = Device(id, kind, a, b, label)
        return id

    def add_zone(self, zone: Zone) -> None:
        for ct, inside in zone.boundary:
            device = self.devices[ct]
            if device.kind != CT or inside not in (device.a, device.b):
                raise ValueError(f"zone {zone.id}: {ct} is not a CT next to {inside}")
        unknown = [s for s in zone.fixed if s not in self.sections] + [d for d in zone.cuts if d not in self.devices]
        if unknown or not (zone.boundary or zone.fixed):
            raise ValueError(f"zone {zone.id}: needs a CT boundary or fixed sections (unknown: {unknown})")
        self.zones[zone.id] = zone

    def breakers(self) -> list[str]:
        return [d.id for d in self.devices.values() if d.kind == BREAKER]

    def has_ground(self) -> bool:
        return any(s.ground for s in self.sections.values())

    def label(self, id: str) -> str:
        for table in (self.sections, self.devices, self.zones):
            if id in table:
                return table[id].label
        return id

    def candidates(self) -> list[str]:
        return [s.id for s in self.sections.values() if s.candidate]

    def _links(self, section: str, conducting: Optional[frozenset[str]]) -> Iterable[Device]:
        for device in self.devices.values():
            if section in (device.a, device.b) and (conducting is None or device.id in conducting):
                yield device

    def conducting(self, closed_breakers: Iterable[str]) -> frozenset[str]:
        closed = set(closed_breakers)
        return frozenset(d.id for d in self.devices.values() if d.kind != BREAKER or d.id in closed)

    def fault_paths(self, fault: str, conducting: frozenset[str],
                    earth: bool = True) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
        """Every simple path from a source to ``fault``, as ``(source, steps)``
        where each step is ``(device, section the current flows into)``,
        ordered from the source towards the fault. Ground sections take part
        only in an earth fault."""
        paths: list[tuple[str, tuple[tuple[str, str], ...]]] = []

        def walk(section: str, visited: frozenset[str], steps: tuple[tuple[str, str], ...]) -> None:
            for device in self._links(section, conducting):
                nxt = device.other(section)
                if nxt in visited:
                    continue
                if self.sections[nxt].ground and not earth:
                    continue
                path = ((device.id, section),) + steps
                if self.sections[nxt].source:
                    paths.append((nxt, path))
                    continue
                walk(nxt, visited | {nxt}, path)

        walk(fault, frozenset({fault}), ())
        return paths

    def ct_flows(self, fault: str, conducting: frozenset[str], earth: bool = True) -> dict[str, frozenset[str]]:
        """CT id -> sections its fault current flows into (normally one; two
        when the CT sits on a loop and carries current both ways)."""
        flows: dict[str, set[str]] = {}
        for _, steps in self.fault_paths(fault, conducting, earth):
            for device, into in steps:
                if self.devices[device].kind == CT:
                    flows.setdefault(device, set()).add(into)
        return {ct: frozenset(into) for ct, into in flows.items()}

    def side_of(self, device_id: str, target: str) -> Optional[str]:
        """The section next to ``device_id`` on the side where ``target`` lies
        (None when ``target`` is reachable from both sides)."""
        device = self.devices[device_id]
        if target in (device.a, device.b):
            return target
        reach = []
        for start in (device.a, device.b):
            seen, queue = {start}, deque([start])
            while queue:
                section = queue.popleft()
                for link in self._links(section, None):
                    nxt = link.other(section)
                    if link.id != device_id and nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            reach.append(target in seen)
        if reach[0] == reach[1]:
            return None
        return device.a if reach[0] else device.b

    def zone_region(self, zone_id: str, open_breakers: frozenset[str]) -> frozenset[str]:
        """Sections inside the zone as its relay measures it: bounded by the
        CTs it currently measures. A CT dropped by end-zone logic is not a
        boundary, so the region runs on through it (and through any breaker,
        open or not) until the next measured CT."""
        zone = self.zones[zone_id]
        if zone.fixed:
            return frozenset(zone.fixed)
        dropped = {ct for ct, breaker in zone.end_zone if breaker in open_breakers}
        cut = {ct for ct, _ in zone.boundary if ct not in dropped} | set(zone.cuts)
        region = {inside for _, inside in zone.boundary}
        queue = deque(region)
        while queue:
            section = queue.popleft()
            for device in self._links(section, None):
                if device.id in cut:
                    continue
                nxt = device.other(section)
                if nxt not in region:
                    region.add(nxt)
                    queue.append(nxt)
        return frozenset(region)
