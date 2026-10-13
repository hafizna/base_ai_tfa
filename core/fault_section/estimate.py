"""Consistency check of fault-location hypotheses against observations.

A hypothesis is "the fault sits in section X of topology variant V", optionally
with one breaker that read open (auxiliary contacts) while its main contacts
kept conducting — a stuck breaker — and, where the topology has grounded
neutrals, whether the fault involves earth. For every interval of the episode
(a span with one breaker state) the topology predicts each observation:

- ``ct_current``: the CT lies on a path from a source to the fault;
- ``zone_operate``: the fault is inside the zone as its relay measures it and
  some of the fault current enters through a measured CT (a source inside the
  region, reached only through a dropped CT or an open breaker, feeds no
  differential current); an earth-only zone (REF) needs an earth fault;
- ``direction``: the CT's fault current flows into the section the relay calls
  forward, or out of it.

Hypotheses are ranked by (strong contradictions, stuck breakers assumed, weak
contradictions): a location that contradicts nothing beats one that only
fits with an extra breaker failure, which beats one that a measurement rules
out. Weak observations — transcribed from a screenshot, an estimated clock,
current after the fault cleared, a computed direction — rank but never rule
out on their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from .topology import CT, Topology

CT_CURRENT = "ct_current"
ZONE_OPERATE = "zone_operate"
DIRECTION = "direction"


@dataclass(frozen=True)
class Interval:
    id: str
    start_s: float
    end_s: float
    open: frozenset[str] = frozenset()      # breakers whose auxiliary contacts read open
    label: str = ""
    after_clearing: bool = False


@dataclass(frozen=True)
class Observation:
    interval: str
    kind: str
    subject: str                # CT id, zone id
    value: Any                  # bool, or "forward" / "reverse"
    source: str                 # recorder and channels it came from
    detail: str = ""
    weak: bool = False
    forward: str = ""           # DIRECTION: a section on the side the relay calls forward


@dataclass(frozen=True)
class Hypothesis:
    variant: int
    section: str
    stuck: frozenset[str] = frozenset()
    earth: bool = True          # only tested both ways where the topology has grounded neutrals


@dataclass(frozen=True)
class Check:
    observation: Observation
    predicted: Any

    @property
    def agrees(self) -> Optional[bool]:
        return None if self.predicted is None else self.predicted == self.observation.value


@dataclass
class Verdict:
    hypothesis: Hypothesis
    checks: list[Check]

    def contradictions(self, weak: bool = False) -> list[Check]:
        return [c for c in self.checks if c.agrees is False and c.observation.weak == weak]

    def agreements(self) -> list[Check]:
        return [c for c in self.checks if c.agrees is True]

    @property
    def rank(self) -> tuple[int, int, int]:
        return (len(self.contradictions()), len(self.hypothesis.stuck), len(self.contradictions(weak=True)))


class _Simulator:
    """Predictions for one topology variant, cached per breaker state."""

    def __init__(self, topology: Topology):
        self.topology = topology
        self._breakers = frozenset(topology.breakers())
        self._paths: dict[tuple[str, frozenset[str], bool], list] = {}
        self._regions: dict[tuple[str, frozenset[str]], frozenset[str]] = {}
        self._sides: dict[tuple[str, str], Optional[str]] = {}

    def _fault_paths(self, section: str, physically_open: frozenset[str], earth: bool):
        key = (section, physically_open, earth)
        if key not in self._paths:
            conducting = self.topology.conducting(self._breakers - physically_open)
            self._paths[key] = self.topology.fault_paths(section, conducting, earth)
        return self._paths[key]

    def _region(self, zone: str, open_breakers: frozenset[str]) -> frozenset[str]:
        key = (zone, open_breakers & self._breakers)
        if key not in self._regions:
            self._regions[key] = self.topology.zone_region(zone, key[1])
        return self._regions[key]

    def predict(self, hypothesis: Hypothesis, interval: Interval, observation: Observation) -> Any:
        topo = self.topology
        paths = self._fault_paths(hypothesis.section, (interval.open & self._breakers) - hypothesis.stuck,
                                  hypothesis.earth)
        if observation.kind == CT_CURRENT:
            if observation.subject not in topo.devices:
                return None
            return any(device == observation.subject for _, steps in paths for device, _ in steps)
        if observation.kind == ZONE_OPERATE:
            if observation.subject not in topo.zones:
                return None
            if topo.zones[observation.subject].earth_only and not hypothesis.earth:
                return False
            region = self._region(observation.subject, interval.open)
            return hypothesis.section in region and any(source not in region for source, _ in paths)
        if observation.kind == DIRECTION:
            if observation.subject not in topo.devices:
                return None
            into = {section for _, steps in paths for device, section in steps if device == observation.subject}
            forward_side = self._side(observation.subject, observation.forward)
            if len(into) != 1 or forward_side is None:
                return None
            return "forward" if into == {forward_side} else "reverse"
        return None

    def _side(self, device: str, target: str) -> Optional[str]:
        key = (device, target)
        if key not in self._sides:
            self._sides[key] = self.topology.side_of(device, target) if target in self.topology.sections else None
        return self._sides[key]


@dataclass
class Estimate:
    topologies: list[Topology]
    intervals: list[Interval]
    observations: list[Observation]
    verdicts: list[Verdict] = field(default_factory=list)

    def best(self) -> list[Verdict]:
        if not self.verdicts:
            return []
        top = min(v.rank for v in self.verdicts)
        return [v for v in self.verdicts if v.rank == top]

    def per_section(self, *, allow_stuck: bool = True) -> list[Verdict]:
        """Best verdict for each candidate section, best sections first;
        ``allow_stuck=False`` keeps to hypotheses where every breaker did
        what its auxiliary contacts say."""
        chosen: dict[str, Verdict] = {}
        for verdict in self.verdicts:
            if not allow_stuck and verdict.hypothesis.stuck:
                continue
            current = chosen.get(verdict.hypothesis.section)
            if current is None or verdict.rank < current.rank:
                chosen[verdict.hypothesis.section] = verdict
        return sorted(chosen.values(), key=lambda v: v.rank)

    def shared_weak_mismatches(self) -> list[Observation]:
        """Weak observations every best hypothesis contradicts: they say
        something about the evidence (a polarity, a relay's decision), not
        about where the fault was."""
        best = self.best()
        if not best:
            return []
        common = [c.observation for c in best[0].contradictions(weak=True)]
        return [obs for obs in common
                if all(any(c.observation == obs for c in v.contradictions(weak=True)) for v in best[1:])]

    def inferred_assumptions(self) -> list[str]:
        """Variant choices every best hypothesis shares, where the variants
        themselves differ — what the records settled that the case left open."""
        best = self.best()
        if not best:
            return []
        all_sets = [set(t.assumptions) for t in self.topologies]
        fixed = set.intersection(*all_sets) if all_sets else set()
        shared = set.intersection(*(set(self.topologies[v.hypothesis.variant].assumptions) for v in best))
        order = list(self.topologies[best[0].hypothesis.variant].assumptions)
        return [a for a in order if a in shared - fixed]

    def label(self, verdict: Verdict) -> str:
        topo = self.topologies[verdict.hypothesis.variant]
        text = topo.label(verdict.hypothesis.section)
        if topo.has_ground():
            text += ", gangguan tanah" if verdict.hypothesis.earth else ", gangguan tanpa tanah"
        if verdict.hypothesis.stuck:
            text += " + " + ", ".join(f"{topo.label(b)} gagal membuka" for b in sorted(verdict.hypothesis.stuck))
        return text

    def explain(self, check: Check, verdict: Verdict) -> str:
        topo = self.topologies[verdict.hypothesis.variant]
        obs = check.observation
        interval = next(i for i in self.intervals if i.id == obs.interval)
        when = f"{interval.id}" + (f" ({interval.label})" if interval.label else "")
        return (f"{when}: {_subject(topo, obs)} diprediksi {_value_text(obs.kind, check.predicted)}, "
                f"teramati {_value_text(obs.kind, obs.value)}"
                + (f" [{obs.detail}]" if obs.detail else "") + f" — {obs.source}"
                + (" (bukti lemah)" if obs.weak else ""))

    def report_lines(self) -> list[str]:
        lines = []
        best = self.best()
        strong = sum(1 for o in self.observations if not o.weak)
        lines.append(f"{len(self.observations)} bukti ({strong} kuat) di {len(self.intervals)} interval, "
                     f"{len(self.verdicts)} hipotesis diuji.")
        if not best or best[0].rank[0] > 0:
            lines.append("Tidak ada lokasi yang konsisten dengan semua bukti kuat; "
                         "lokasi dengan kontradiksi paling sedikit:")
        else:
            lines.append("Lokasi yang konsisten dengan semua bukti kuat:")
        sections = []
        for verdict in best:
            if verdict.hypothesis.section not in sections:
                sections.append(verdict.hypothesis.section)
        shared = self.shared_weak_mismatches()
        for section in sections:
            group = [v for v in best if v.hypothesis.section == section]
            weak = group[0].contradictions(weak=True)
            lines.append(f"  ✓ {self.label(group[0])}  ({len(group[0].agreements())} bukti cocok"
                         + (f", {len(weak)} bukti lemah tidak cocok" if weak else "")
                         + f"; {len(group)} varian topologi)")
            for check in weak:
                if check.observation not in shared:
                    lines.append(f"      · {self.explain(check, group[0])}")
        if shared:
            lines.append("Catatan — tidak cocok dengan semua lokasi di atas, jadi bukan petunjuk lokasi:")
            for obs in shared:
                check = next(c for c in best[0].checks if c.observation == obs)
                hint = ("arah terukur berlawanan: cek polaritas CT/VT atau konvensi forward rele ini"
                        if obs.kind == DIRECTION and obs.detail.startswith("ΔV/ΔI")
                        else "keputusan rele tidak sesuai arah arus gangguan" if obs.kind == DIRECTION
                        else "periksa sumber bukti ini")
                lines.append(f"  ! {self.explain(check, best[0])} → {hint}")
        inferred = self.inferred_assumptions()
        if inferred:
            lines.append("Disimpulkan dari rekaman (berlaku untuk semua lokasi di atas):")
            lines.extend(f"  • {text}" for text in inferred)
        stuck_only = [v for v in self.per_section()
                      if v.hypothesis.section not in sections and v.hypothesis.stuck and v.rank[0] == 0]
        if stuck_only:
            lines.append("Hanya konsisten bila ada CB yang gagal membuka:")
            lines.extend(f"  ? {self.label(v)}" for v in stuck_only)
        skip = set(sections) | {v.hypothesis.section for v in stuck_only}
        lines.append("Ditolak:")
        for verdict in self.per_section(allow_stuck=False):
            if verdict.hypothesis.section in skip:
                continue
            reasons = self._ordered(verdict.contradictions()) or self._ordered(verdict.contradictions(weak=True))
            lines.append(f"  ✗ {self.label(verdict)} — {self.explain(reasons[0], verdict)}")
        return lines

    def _ordered(self, checks: list[Check]) -> list[Check]:
        """Earliest interval first, measured current before relay decisions."""
        order = {i.id: n for n, i in enumerate(self.intervals)}
        kinds = {CT_CURRENT: 0, ZONE_OPERATE: 1, DIRECTION: 2}
        return sorted(checks, key=lambda c: (order.get(c.observation.interval, 99), kinds.get(c.observation.kind, 9)))

    def to_dict(self) -> dict[str, Any]:
        def check_dict(check: Check, verdict: Verdict) -> dict[str, Any]:
            obs = check.observation
            return {"interval": obs.interval, "kind": obs.kind, "subject": obs.subject,
                    "observed": obs.value, "predicted": check.predicted, "agrees": check.agrees,
                    "weak": obs.weak, "source": obs.source, "detail": obs.detail,
                    "text": self.explain(check, verdict) if check.agrees is False else None}

        return {
            "intervals": [{"id": i.id, "start_s": i.start_s, "end_s": i.end_s, "open": sorted(i.open),
                           "label": i.label, "after_clearing": i.after_clearing} for i in self.intervals],
            "best": [{"section": v.hypothesis.section, "label": self.label(v), "rank": list(v.rank),
                      "assumptions": list(self.topologies[v.hypothesis.variant].assumptions)} for v in self.best()],
            "inferred_assumptions": self.inferred_assumptions(),
            "sections": [{"section": v.hypothesis.section, "label": self.label(v), "rank": list(v.rank),
                          "checks": [check_dict(c, v) for c in v.checks]} for v in self.per_section()],
        }


def _subject(topo: Topology, obs: Observation) -> str:
    return topo.label(obs.subject)


def _value_text(kind: str, value: Any) -> str:
    if value is None:
        return "tidak terprediksi"
    if kind == CT_CURRENT:
        return "membawa arus gangguan" if value else "tanpa arus gangguan"
    if kind == ZONE_OPERATE:
        return "operate" if value else "tidak operate"
    return str(value)


def expected_observations(topology: Topology, section: str, intervals: Sequence[Interval], *,
                          stuck: Iterable[str] = (), earth: bool = True, cts: Optional[Iterable[str]] = None,
                          zones: Optional[Iterable[str]] = None, source: str = "simulasi") -> list[Observation]:
    """What every CT and zone would show, interval by interval, if the fault
    sat in ``section`` — the forward model the estimate checks against."""
    simulator = _Simulator(topology)
    hypothesis = Hypothesis(0, section, frozenset(stuck), earth)
    if cts is None:
        cts = [d.id for d in topology.devices.values() if d.kind == CT]
    if zones is None:
        zones = list(topology.zones)
    cts, zones = list(cts), list(zones)
    observations = []
    for interval in intervals:
        for kind, subjects in ((CT_CURRENT, cts), (ZONE_OPERATE, zones)):
            for subject in subjects:
                probe = Observation(interval.id, kind, subject, None, source)
                value = simulator.predict(hypothesis, interval, probe)
                if value is not None:
                    observations.append(Observation(interval.id, kind, subject, value, source))
    return observations


def estimate(topologies: Sequence[Topology], intervals: Sequence[Interval], observations: Sequence[Observation],
             *, stuck_candidates: Optional[Iterable[str]] = None) -> Estimate:
    """Test every candidate section of every variant, without and with each
    single stuck breaker, against every observation."""
    if stuck_candidates is None:
        stuck_candidates = sorted({b for i in intervals for b in i.open})
    stuck_candidates = list(stuck_candidates)
    by_id = {i.id: i for i in intervals}
    result = Estimate(list(topologies), list(intervals), list(observations))
    for variant, topology in enumerate(topologies):
        simulator = _Simulator(topology)
        stuck_options = [frozenset()] + [frozenset({b}) for b in stuck_candidates if b in topology.devices]
        earth_options = (True, False) if topology.has_ground() else (True,)
        for section in topology.candidates():
            for stuck in stuck_options:
                for earth in earth_options:
                    hypothesis = Hypothesis(variant, section, stuck, earth)
                    checks = [Check(obs, simulator.predict(hypothesis, by_id[obs.interval], obs))
                              for obs in observations if obs.interval in by_id]
                    result.verdicts.append(Verdict(hypothesis, checks))
    return result
