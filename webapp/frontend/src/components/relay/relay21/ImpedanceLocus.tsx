import { useEffect, useMemo, useRef, useState } from "react";

import { computeLocus, fetchElectricalParams21, fetchFaultClassification21 } from "../../../api/client";
import Plot from "../../plot/PlotlyChart";
import styles from "../../panels/Panel.module.css";

interface Zone {
  label: string;
  shape: "mho" | "quad";
  reach_mode?: "rx" | "z";
  center_r: number;
  center_x: number;
  radius: number;
  rf_fwd: number;
  rf_rev: number;
  xf: number;
  xr: number;
  z_fwd?: number;
  z_rev?: number;
  line_angle_deg: number;
  color: string;
  /** Direct polygon vertices (R, X) — when set, rendered as-is instead of via quadVertices() */
  poly_r?: number[];
  poly_x?: number[];
}

interface LocusPoint {
  t: number;
  r: number;
  x: number;
}

interface ImportedZone {
  label: string;
  shapeType: "circle" | "poly" | "mho" | "xrio";
  centerR?: number;
  centerX?: number;
  radius?: number;
  reach?: number;
  poly?: { r: number; x: number }[];
  X?: number;    // reactance reach (Z reach, Ohm)
  R?: number;    // fault resistance reach (Ohm)
  RF?: number | null;
  lineAngleDeg?: number;  // line impedance angle for tilted zone rendering
}

interface ImportedRelayData {
  kind: "rio" | "xrio";
  phGnd: ImportedZone[];
  phPh: ImportedZone[];
  earthComp?: {
    k0: number;
    angleDeg: number;
    source: string;
  };
  ctRatio?: number;
  vtRatio?: number;
}

type LoopName = "ZA" | "ZB" | "ZC" | "ZAB" | "ZBC" | "ZCA";

type ZoneFamily = "ground" | "phase";
type TimeMode = "fault" | "all";
type PlotFamily = "ground" | "phase";
type PlotRange = { x?: [number, number]; y?: [number, number] };
type DetailMode = "standard" | "detailed";
type ZoneShapeChoice = "mho" | "quad_rx" | "quad_z";
type FaultClassification21 = Awaited<ReturnType<typeof fetchFaultClassification21>>;

const GROUND_LOOPS: LoopName[] = ["ZA", "ZB", "ZC"];
const PHASE_LOOPS: LoopName[] = ["ZAB", "ZBC", "ZCA"];

const LOOP_COLORS: Record<LoopName, string> = {
  ZA: "#16a34a",
  ZB: "#d946ef",
  ZC: "#2563eb",
  ZAB: "#0891b2",
  ZBC: "#b45309",
  ZCA: "#be123c",
};

function loopForGroundPhase(phase: string): LoopName | null {
  const normalized = phase.trim().toUpperCase();
  if (normalized === "A") return "ZA";
  if (normalized === "B") return "ZB";
  if (normalized === "C") return "ZC";
  return null;
}

function loopForPhasePair(phases: string[]): LoopName | null {
  const key = [...new Set(phases.map((phase) => phase.trim().toUpperCase()))].sort().join("");
  if (key === "AB") return "ZAB";
  if (key === "BC") return "ZBC";
  if (key === "AC") return "ZCA";
  return null;
}

function eventLoopsForFamily(family: PlotFamily, classification: FaultClassification21 | null): LoopName[] {
  if (!classification) return family === "ground" ? GROUND_LOOPS : PHASE_LOOPS;
  const phases = classification.phases ?? [];

  if (classification.to_ground) {
    if (family !== "ground") return [];
    const loops = phases.map(loopForGroundPhase).filter((loop): loop is LoopName => loop !== null);
    return loops.length ? loops : GROUND_LOOPS;
  }

  if (family !== "phase") return [];
  if (phases.length >= 3) return PHASE_LOOPS;
  const loop = loopForPhasePair(phases);
  return loop ? [loop] : PHASE_LOOPS;
}

const GROUND_ZONE_TEMPLATES: Zone[] = [
  { label: "Z1", shape: "mho", center_r: 0, center_x: 0, radius: 0, rf_fwd: 0, rf_rev: 0, xf: 0, xr: 0, line_angle_deg: 75, color: "#22c55e" },
  { label: "Z2", shape: "mho", center_r: 0, center_x: 0, radius: 0, rf_fwd: 0, rf_rev: 0, xf: 0, xr: 0, line_angle_deg: 75, color: "#f59e0b" },
  { label: "Z3", shape: "mho", center_r: 0, center_x: 0, radius: 0, rf_fwd: 0, rf_rev: 0, xf: 0, xr: 0, line_angle_deg: 75, color: "#ef4444" },
];

const PHASE_ZONE_TEMPLATES: Zone[] = [
  { label: "Z1", shape: "mho", center_r: 0, center_x: 0, radius: 0, rf_fwd: 0, rf_rev: 0, xf: 0, xr: 0, line_angle_deg: 75, color: "#22c55e" },
  { label: "Z2", shape: "mho", center_r: 0, center_x: 0, radius: 0, rf_fwd: 0, rf_rev: 0, xf: 0, xr: 0, line_angle_deg: 75, color: "#f59e0b" },
  { label: "Z3", shape: "mho", center_r: 0, center_x: 0, radius: 0, rf_fwd: 0, rf_rev: 0, xf: 0, xr: 0, line_angle_deg: 75, color: "#ef4444" },
];

function mhoCircleTrace(zone: Zone): Partial<Plotly.ScatterData> {
  const theta = Array.from({ length: 361 }, (_, i) => (i * Math.PI) / 180);
  return {
    x: theta.map((t) => zone.center_r + zone.radius * Math.cos(t)),
    y: theta.map((t) => zone.center_x + zone.radius * Math.sin(t)),
    type: "scatter",
    mode: "lines",
    name: zone.label,
    line: { color: zone.color, width: 1.6, dash: "dot" },
    showlegend: true,
  };
}

function reachFromReactance(xReach: number, angleDeg: number) {
  const sinA = Math.sin((angleDeg * Math.PI) / 180);
  return Math.abs(sinA) > 0.01 ? xReach / sinA : xReach;
}

function reactanceFromReach(zReach: number, angleDeg: number) {
  return zReach * Math.sin((angleDeg * Math.PI) / 180);
}

function zReachQuadVertices(zone: Zone) {
  const angleRad = (zone.line_angle_deg * Math.PI) / 180;
  const sinA = Math.sin(angleRad);
  const cosA = Math.cos(angleRad);
  const slope = Math.abs(sinA) > 0.01 ? cosA / sinA : 0;
  const zFwd = zone.z_fwd ?? reachFromReactance(zone.xf, zone.line_angle_deg);
  const zRev = zone.z_rev ?? Math.max(0, reachFromReactance(zone.xr, zone.line_angle_deg));
  const rReachFwd = zFwd * cosA;
  const xReachFwd = zFwd * sinA;
  const rReachRev = zRev * cosA;
  const xReachRev = zRev * sinA;
  const leftR = -Math.max(0, zone.rf_rev);
  const rightR = Math.max(0, zone.rf_fwd);
  const topXAt = (r: number) => xReachFwd + (r - rReachFwd) * slope;
  const bottomXAt = (r: number) => -xReachRev + (r + rReachRev) * slope;

  return [
    [leftR, bottomXAt(leftR)],
    [rightR, bottomXAt(rightR)],
    [rightR, topXAt(rightR)],
    [leftR, topXAt(leftR)],
    [leftR, bottomXAt(leftR)],
  ];
}

function quadVertices(zone: Zone) {
  if (zone.reach_mode === "z") return zReachQuadVertices(zone);

  const ang = (zone.line_angle_deg * Math.PI) / 180;
  const cos = Math.cos(ang);
  const sin = Math.sin(ang);
  return [
    [zone.rf_fwd * cos, zone.rf_fwd * sin],
    [-zone.rf_rev * cos, -zone.rf_rev * sin],
    [-zone.rf_rev * cos + zone.xr * sin, zone.xr * cos - zone.rf_rev * sin],
    [zone.rf_fwd * cos + zone.xf * sin, zone.xf * cos + zone.rf_fwd * sin],
    [zone.rf_fwd * cos, zone.rf_fwd * sin],
  ];
}

function quadTrace(zone: Zone): Partial<Plotly.ScatterData> {
  let xs: number[];
  let ys: number[];
  if (zone.poly_r && zone.poly_x && zone.poly_r.length >= 3) {
    xs = zone.poly_r;
    ys = zone.poly_x;
  } else {
    const pts = quadVertices(zone);
    xs = pts.map((p) => p[0]);
    ys = pts.map((p) => p[1]);
  }
  return {
    x: xs,
    y: ys,
    type: "scatter",
    mode: "lines",
    name: zone.label,
    line: { color: zone.color, width: 1.6, dash: "dot" },
    showlegend: true,
  };
}

function parseTripCharCircle(block: string) {
  const arc = block.match(/\bARC\s+([-0-9.]+)\s*,\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*,\s*(CW|CCW)/i);
  if (!arc) return null;

  const radius = Number.parseFloat(arc[1]);
  // arc[2] is the angle (degrees) of the START point measured from the circle centre.
  // Centre = START − radius×(cos(angle), sin(angle)).
  const arcAngleDeg = Number.parseFloat(arc[2]);
  const sweep = Number.parseFloat(arc[3]);
  if (!Number.isFinite(radius) || radius <= 0 || Math.abs(sweep) < 359) {
    return null;
  }

  const startMatch = block.match(/\bSTART\s+([-0-9.]+)\s*,\s*([-0-9.]+)/i);
  if (!startMatch) return null;
  const startR = Number.parseFloat(startMatch[1]);
  const startX = Number.parseFloat(startMatch[2]);
  const angleRad = (arcAngleDeg * Math.PI) / 180;
  const centerR = startR - radius * Math.cos(angleRad);
  const centerX = startX - radius * Math.sin(angleRad);

  return { centerR, centerX, radius };
}

function parseTripCharPolygon(block: string) {
  const points: { r: number; x: number }[] = [];
  const start = block.match(/\bSTART\s+([-0-9.]+)\s*,\s*([-0-9.]+)/i);
  if (start) {
    points.push({ r: Number.parseFloat(start[1]), x: Number.parseFloat(start[2]) });
  }
  const lineMatches = block.matchAll(/\bLINE\s+([-0-9.]+)\s*,\s*([-0-9.]+)/gi);
  for (const match of lineMatches) {
    points.push({ r: Number.parseFloat(match[1]), x: Number.parseFloat(match[2]) });
  }
  return points.length >= 3 ? points : null;
}

function clipShapeByRioLines(shapeText: string) {
  const lines = Array.from(
    shapeText.matchAll(
      /^\s*LINE\s+([+-]?\d[\d.eE+-]*),\s*([+-]?\d[\d.eE+-]*),\s*([+-]?\d[\d.eE+-]*),\s*(LEFT|RIGHT)\b/gim
    )
  ).map((match) => ({
    r: Number.parseFloat(match[1]),
    x: Number.parseFloat(match[2]),
    angleDeg: Number.parseFloat(match[3]),
    side: match[4].toUpperCase() as "LEFT" | "RIGHT",
  }));

  if (lines.length < 2) return null;

  const anchors = lines.flatMap((line) => [Math.abs(line.r), Math.abs(line.x)]).filter(Number.isFinite);
  const extent = Math.max(100, ...anchors) * 4;
  let poly = [
    { r: -extent, x: -extent },
    { r: extent, x: -extent },
    { r: extent, x: extent },
    { r: -extent, x: extent },
  ];

  const signedDistance = (point: { r: number; x: number }, line: (typeof lines)[number]) => {
    const rad = (line.angleDeg * Math.PI) / 180;
    const dirR = Math.cos(rad);
    const dirX = Math.sin(rad);
    return dirR * (point.x - line.x) - dirX * (point.r - line.r);
  };

  const isInside = (point: { r: number; x: number }, line: (typeof lines)[number]) => {
    const value = signedDistance(point, line);
    return line.side === "LEFT" ? value >= -1e-9 : value <= 1e-9;
  };

  for (const line of lines) {
    const next: { r: number; x: number }[] = [];
    for (let idx = 0; idx < poly.length; idx += 1) {
      const current = poly[idx];
      const previous = poly[(idx + poly.length - 1) % poly.length];
      const currentInside = isInside(current, line);
      const previousInside = isInside(previous, line);

      if (currentInside !== previousInside) {
        const prevDistance = signedDistance(previous, line);
        const currentDistance = signedDistance(current, line);
        const denom = prevDistance - currentDistance;
        if (Math.abs(denom) > 1e-12) {
          const t = prevDistance / denom;
          next.push({
            r: previous.r + (current.r - previous.r) * t,
            x: previous.x + (current.x - previous.x) * t,
          });
        }
      }

      if (currentInside) next.push(current);
    }
    poly = next;
    if (poly.length < 3) return null;
  }

  const deduped = poly.filter((point, idx) => {
    const prev = poly[(idx + poly.length - 1) % poly.length];
    return Math.hypot(point.r - prev.r, point.x - prev.x) > 1e-6;
  });

  return deduped.length >= 3 ? deduped : null;
}

function parseSifangRIO(text: string): ImportedRelayData | null {
  if (!/BEGIN\s+TESTOBJECT/i.test(text)) return null;

  const distMatch = text.match(/BEGIN\s+DISTANCE([\s\S]*?)END\s+DISTANCE/i);
  if (!distMatch) return null;
  const distText = distMatch[1];

  const phGnd: ImportedZone[] = [];
  const phPh: ImportedZone[] = [];

  const zonePattern = /BEGIN\s+ZONE([\s\S]*?)END\s+ZONE/gi;
  let zoneMatch: RegExpExecArray | null;

  while ((zoneMatch = zonePattern.exec(distText)) !== null) {
    const block = zoneMatch[1];
    const indexMatch = block.match(/^\s*INDEX\s+(\d+)/m);
    const loopMatch = block.match(/^\s*FAULTLOOP\s+(\w+)/m);
    const shapeMatch = block.match(/BEGIN\s+SHAPE([\s\S]*?)END\s+SHAPE/i);
    if (!indexMatch || !loopMatch || !shapeMatch) continue;

    const label = `Z${indexMatch[1]}`;
    const faultloop = loopMatch[1].toUpperCase();
    const shapeText = shapeMatch[1];

    const poly = clipShapeByRioLines(shapeText);
    if (!poly || poly.length < 3) continue;

    const zone: ImportedZone = { label, shapeType: "poly", poly };
    if (faultloop === "LN") {
      if (!phGnd.find((z) => z.label === label)) phGnd.push(zone);
    } else if (faultloop === "LL") {
      if (!phPh.find((z) => z.label === label)) phPh.push(zone);
    }
  }

  if (phGnd.length === 0 && phPh.length === 0) return null;

  // KM mag, angle = residual compensation (K0) in Sifang/Alstom .rio format
  const kmMatch = text.match(/\bKM\s+([-0-9.]+)\s*,\s*([-0-9.]+)/i);
  let earthComp: ImportedRelayData["earthComp"];
  if (kmMatch) {
    const k0 = Number.parseFloat(kmMatch[1]);
    const angleDeg = Number.parseFloat(kmMatch[2]);
    if (Number.isFinite(k0) && Number.isFinite(angleDeg) && k0 > 0) {
      earthComp = { k0, angleDeg, source: `RIO KM=${k0.toFixed(4)}, ∠=${angleDeg.toFixed(1)}°` };
    }
  }

  return {
    kind: "rio",
    phGnd: phGnd.slice(0, 3),
    phPh: phPh.slice(0, 3),
    ...(earthComp ? { earthComp } : {}),
    ...parseRioRatios(text),
  };
}

function parseRioEarthComp(text: string): ImportedRelayData["earthComp"] {
  const reRl = text.match(/\bRE\/RL\s+([-0-9.]+)\s*,\s*([-0-9.]+)/i);
  const xeXl = text.match(/\bXE\/XL\s+([-0-9.]+)\s*,\s*([-0-9.]+)/i);
  if (!reRl || !xeXl) return undefined;
  const re = Number.parseFloat(reRl[1]);
  const xe = Number.parseFloat(xeXl[1]);
  if (!Number.isFinite(re) || !Number.isFinite(xe)) return undefined;
  return {
    k0: Math.hypot(re, xe),
    angleDeg: (Math.atan2(xe, re) * 180) / Math.PI,
    source: `RIO RE/RL=${re.toFixed(3)}, XE/XL=${xe.toFixed(3)}`,
  };
}

function readRioNumber(text: string, labels: string[]): number | null {
  for (const label of labels) {
    const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
    const match = text.match(new RegExp(`(?:^|\\n)\\s*${escaped}\\s*(?:=|:)?\\s*([-+]?\\d+(?:\\.\\d+)?)`, "i"));
    if (!match) continue;
    const value = Number.parseFloat(match[1]);
    if (Number.isFinite(value)) return value;
  }
  return null;
}

function parseRioRatios(text: string): Pick<ImportedRelayData, "ctRatio" | "vtRatio"> {
  const vtPrimary = readRioNumber(text, ["VPRIM-LL", "VPRIM", "VPRIMARY", "VT PRIMARY", "MAIN VT PRIMARY"]);
  const vtSecondary = readRioNumber(text, ["VNOM", "VSEC", "VSECONDARY", "VT SECONDARY", "MAIN VT SEC'Y"]);
  const ctPrimary = readRioNumber(text, ["IPRIM", "IPRIMARY", "CT PRIMARY", "CTPRIMARY", "PHASE CT PRIMARY"]);
  const ctSecondary = readRioNumber(text, ["INOM", "ISEC", "ISECONDARY", "CT SECONDARY", "CTSECONDARY", "PHASE CT SEC'Y"]);
  const vtRatio = vtPrimary != null && vtSecondary != null && vtSecondary > 0 ? vtPrimary / vtSecondary : null;
  const ctRatio = ctPrimary != null && ctSecondary != null && ctSecondary > 0 ? ctPrimary / ctSecondary : null;
  return {
    ...(ctRatio != null ? { ctRatio } : {}),
    ...(vtRatio != null ? { vtRatio } : {}),
  };
}

function parseRIO(text: string): ImportedRelayData | null {
  if (!/BEGIN\s+PROTECTIONDEVICE/i.test(text)) return parseSifangRIO(text);

  const phGnd: ImportedZone[] = [];
  const phPh: ImportedZone[] = [];
  const zonePattern = /BEGIN\s+(ZONE(?:-OVERREACH)?)\b([\s\S]*?)END\s+\1/gi;
  let zoneMatch: RegExpExecArray | null = zonePattern.exec(text);

  while (zoneMatch) {
    const block = zoneMatch[0];
    const labelMatch = block.match(/\bNAME\s+(.+?)(?=\s+(?:TIME1|TIMEM|BEGIN|ACTIVE|INDEX|FAULTLOOP)\b|$)/i);
    const label = labelMatch?.[1]?.trim() || `Z${Math.max(phGnd.length, phPh.length) + 1}`;
    const tripPhase = block.match(/BEGIN\s+TRIPCHAR([\s\S]*?)END\s+TRIPCHAR/i);
    const tripEarth = block.match(/BEGIN\s+TRIPCHAR-EARTH([\s\S]*?)END\s+TRIPCHAR-EARTH/i);

    if (tripPhase) {
      const circle = parseTripCharCircle(tripPhase[1]);
      const poly = parseTripCharPolygon(tripPhase[1]);
      if (circle) {
        phPh.push({
          label,
          shapeType: "circle",
          centerR: circle.centerR,
          centerX: circle.centerX,
          radius: circle.radius,
        });
      } else if (poly) {
        phPh.push({ label, shapeType: "poly", poly });
      }
    }

    if (tripEarth) {
      const circle = parseTripCharCircle(tripEarth[1]);
      const poly = parseTripCharPolygon(tripEarth[1]);
      if (circle) {
        phGnd.push({
          label,
          shapeType: "circle",
          centerR: circle.centerR,
          centerX: circle.centerX,
          radius: circle.radius,
        });
      } else if (poly) {
        phGnd.push({ label, shapeType: "poly", poly });
      }
    }

    zoneMatch = zonePattern.exec(text);
  }

  return {
    kind: "rio",
    phGnd: phGnd.slice(0, 3),
    phPh: phPh.slice(0, 3),
    earthComp: parseRioEarthComp(text),
    ...parseRioRatios(text),
  };
}

function parseXRIO(text: string): ImportedRelayData | null {
  const parser = new DOMParser();
  const xml = parser.parseFromString(text, "application/xml");
  if (xml.querySelector("parsererror")) return null;

  const params = Array.from(xml.querySelectorAll("Parameter"));

  // Return FIRST occurrence of a parameter by Name text — avoids picking up duplicate
  // calibration blocks (e.g. GE P442 repeats many params with default values).
  function getFirst(name: string): string | null {
    const p = params.find((node) => node.querySelector("Name")?.textContent?.trim() === name);
    return p?.querySelector("Value")?.textContent?.trim() ?? null;
  }

  const lineAngleDeg = Number.parseFloat(getFirst("Line Angle") ?? "75");
  const angDeg = Number.isFinite(lineAngleDeg) ? lineAngleDeg : 75;

  // GE/Alstom P442 xrio naming: Z1/Z2/Z3 = X reach (along line angle, Ohm secondary)
  //   R1G/R2G/R3G-R4G = earth fault resistance coverage (Ohm secondary)
  //   R1Ph/R2Ph/R3Ph-R4Ph = phase fault resistance coverage (Ohm secondary)
  const EARTH_ZONES = [
    { label: "Z1", xKey: "Z1",  rKey: "R1G" },
    { label: "Z2", xKey: "Z2",  rKey: "R2G" },
    { label: "Z3", xKey: "Z3",  rKey: "R3G-R4G" },
  ];
  const PHASE_ZONES = [
    { label: "Z1", xKey: "Z1",  rKey: "R1Ph" },
    { label: "Z2", xKey: "Z2",  rKey: "R2Ph" },
    { label: "Z3", xKey: "Z3",  rKey: "R3Ph-R4Ph" },
  ];

  function buildP442Zones(defs: typeof EARTH_ZONES): ImportedZone[] {
    return defs.flatMap(({ label, xKey, rKey }) => {
      const xStr = getFirst(xKey);
      const rStr = getFirst(rKey);
      if (!xStr || !rStr) return [];
      const x = Number.parseFloat(xStr);
      const r = Number.parseFloat(rStr);
      if (!Number.isFinite(x) || x <= 0 || !Number.isFinite(r) || r <= 0) return [];
      return [{ label, shapeType: "xrio" as const, X: x, R: r, lineAngleDeg: angDeg }];
    });
  }

  // Earth compensation: kZ1 Res Comp + kZ1 Angle (GE P442)
  const k0MagStr = getFirst("kZ1 Res Comp");
  const k0AngStr = getFirst("kZ1 Angle");
  let earthComp: ImportedRelayData["earthComp"];
  if (k0MagStr && k0AngStr) {
    const k0 = Number.parseFloat(k0MagStr);
    const angleDeg = Number.parseFloat(k0AngStr);
    if (Number.isFinite(k0) && Number.isFinite(angleDeg)) {
      earthComp = { k0, angleDeg, source: `xrio kZ1=${k0.toFixed(4)}, ∠=${angleDeg.toFixed(1)}°` };
    }
  }

  // CT/VT ratios from the "CT AND VT RATIOS" block
  const vtPrimary = Number.parseFloat(getFirst("Main VT Primary") ?? "");
  const vtSecondary = Number.parseFloat(getFirst("Main VT Sec'y") ?? "");
  const ctPrimary = Number.parseFloat(getFirst("Phase CT Primary") ?? "");
  const ctSecondary = Number.parseFloat(getFirst("Phase CT Sec'y") ?? "");
  const vtRatio = Number.isFinite(vtPrimary) && vtSecondary > 0 ? vtPrimary / vtSecondary : null;
  const ctRatio = Number.isFinite(ctPrimary) && ctSecondary > 0 ? ctPrimary / ctSecondary : null;

  const phGnd = buildP442Zones(EARTH_ZONES);
  const phPh = buildP442Zones(PHASE_ZONES);
  if (phGnd.length === 0 && phPh.length === 0) return null;

  return {
    kind: "xrio",
    phGnd,
    phPh,
    ...(earthComp ? { earthComp } : {}),
    ...(ctRatio !== null && vtRatio !== null ? { ctRatio, vtRatio } : {}),
  };
}

function importedZoneToConfig(zone: ImportedZone, fallback: Zone): Zone {
  if (zone.shapeType === "circle" || zone.shapeType === "mho") {
    return {
      ...fallback,
      label: zone.label || fallback.label,
      shape: "mho",
      center_r: zone.centerR ?? fallback.center_r,
      center_x: zone.centerX ?? fallback.center_x,
      radius: zone.radius ?? zone.reach ?? fallback.radius,
    };
  }

  if (zone.shapeType === "xrio" && zone.X != null && zone.R != null) {
    // P442-style quadrilateral: X = reactance reach (along line angle), R = fault resistance reach.
    // Build polygon vertices using the tilted-top boundary formula:
    //   X_top(R) = xReach + (R − xReach·cot) · cot    where cot = cos/sin
    const ang = ((zone.lineAngleDeg ?? fallback.line_angle_deg) * Math.PI) / 180;
    const sinA = Math.sin(ang);
    const cosA = Math.cos(ang);
    const cotA = sinA > 0.01 ? cosA / sinA : 0;
    const xf = zone.X;
    const rfFwd = zone.R;
    const rfRev = Math.max(1, rfFwd * 0.15);
    const xr = Math.max(1, xf * 0.08);
    const xReachR = xf * cotA;                             // R at the line-reach point
    const xTopRight = xf + (rfFwd - xReachR) * cotA;
    const xTopLeft  = xf + (-rfRev - xReachR) * cotA;
    const polyR = [-rfRev, rfFwd, rfFwd, -rfRev, -rfRev];
    const polyX = [-xr, -xr, xTopRight, Math.max(xTopLeft, xTopRight * 0.6), -xr];
    return {
      ...fallback,
      label: zone.label || fallback.label,
      shape: "quad",
      rf_fwd: rfFwd,
      rf_rev: rfRev,
      xf,
      xr,
      line_angle_deg: zone.lineAngleDeg ?? fallback.line_angle_deg,
      poly_r: polyR,
      poly_x: polyX,
    };
  }

  if (zone.shapeType === "poly" && zone.poly && zone.poly.length >= 3) {
    const rs = zone.poly.map((p) => p.r);
    const xs = zone.poly.map((p) => p.x);
    const maxR = Math.max(...rs);
    const minR = Math.min(...rs);
    const maxX = Math.max(...xs);
    const minX = Math.min(...xs);
    const polyR = [...rs, rs[0]];
    const polyX = [...xs, xs[0]];
    return {
      ...fallback,
      label: zone.label || fallback.label,
      shape: "quad",
      rf_fwd: maxR,
      rf_rev: Math.abs(minR),
      xf: maxX,
      xr: Math.abs(minX),
      poly_r: polyR,
      poly_x: polyX,
    };
  }

  return fallback;
}

function isRenderableImportedZone(zone: ImportedZone) {
  if (zone.shapeType === "xrio") return zone.X != null && zone.X > 0 && zone.R != null && zone.R > 0;
  if (zone.shapeType === "poly") return !!zone.poly && zone.poly.length >= 3;
  return zone.shapeType === "circle" || zone.shapeType === "mho";
}

function loopTrace(loop: LoopName, points: LocusPoint[], detailMode: DetailMode): Partial<Plotly.ScatterData> {
  const xVals: Array<number | null> = [];
  const yVals: Array<number | null> = [];
  const tVals: Array<number | null> = [];
  const gaps = points
    .slice(1)
    .map((point, idx) => point.t - points[idx].t)
    .filter((gap) => Number.isFinite(gap) && gap > 0);
  const medianGap = gaps.length ? [...gaps].sort((a, b) => a - b)[Math.floor(gaps.length / 2)] : 0;
  // Only insert a visual break when the time gap is substantially larger than normal sample spacing.
  // medianGap*1.8 is too tight — it fragments locus during inception settling (10ms gap at 1kHz).
  // Use max(medianGap*8, 0.04s) so we only break on genuine multi-cycle discontinuities.
  const gapThreshold = Math.max(medianGap * 8, 0.04);
  points.forEach((point, idx) => {
    const prev = points[idx - 1];
    const timeGap = idx > 0 && medianGap > 0 && point.t - prev.t > gapThreshold;
    if (timeGap) {
      xVals.push(null);
      yVals.push(null);
      tVals.push(null);
    }
    xVals.push(point.r);
    yVals.push(point.x);
    tVals.push(point.t * 1000);
  });
  return {
    x: xVals,
    y: yVals,
    customdata: tVals,
    type: "scatter",
    mode: "lines+markers",
    name: loop,
    line: {
      color: LOOP_COLORS[loop],
      width: detailMode === "detailed" ? 1.35 : 1.5,
      shape: detailMode === "detailed" ? "linear" : "spline",
      smoothing: detailMode === "detailed" ? 0 : 0.8,
    },
    marker: { color: LOOP_COLORS[loop], size: detailMode === "detailed" ? 3 : 5, symbol: "square" },
    connectgaps: false,
    hovertemplate: `${loop}<br>t=%{customdata:.2f} ms<br>R=%{x:.2f} Ω<br>X=%{y:.2f} Ω<extra></extra>`,
  };
}

function directionTrace(loop: LoopName, points: LocusPoint[]): Partial<Plotly.ScatterData> | null {
  if (points.length < 6) return null;
  const step = Math.max(3, Math.floor(points.length / 9));
  const picked = points.filter((_, idx) => idx > 0 && idx < points.length - 1 && idx % step === 0).slice(0, 10);
  if (!picked.length) return null;
  return {
    x: picked.map((point) => point.r),
    y: picked.map((point) => point.x),
    customdata: picked.map((point) => point.t * 1000),
    type: "scatter",
    mode: "markers",
    name: `${loop} direction`,
    marker: {
      color: LOOP_COLORS[loop],
      size: 8,
      symbol: "triangle-right",
      line: { color: "#ffffff", width: 0.8 },
    },
    showlegend: false,
    hovertemplate: `${loop} direction<br>t=%{customdata:.2f} ms<br>R=%{x:.2f} ohm<br>X=%{y:.2f} ohm<extra></extra>`,
  };
}

function measuredTrace(loops: LoopName[], pointsByLoop: Partial<Record<LoopName, LocusPoint[]>>): Partial<Plotly.ScatterData> | null {
  const hits = loops.flatMap((loop) => {
    const points = pointsByLoop[loop] ?? [];
    const point = points[points.length - 1];
    return point ? [{ loop, point }] : [];
  });
  if (!hits.length) return null;
  return {
    x: hits.map((hit) => hit.point.r),
    y: hits.map((hit) => hit.point.x),
    customdata: hits.map((hit) => [hit.loop, hit.point.t * 1000]),
    type: "scatter",
    mode: "text+markers",
    name: "Measured impedance",
    text: hits.map((hit) => hit.loop),
    textposition: "top center",
    marker: { color: "#111827", size: 9, symbol: "diamond", line: { color: "#ffffff", width: 1.2 } },
    showlegend: true,
    hovertemplate: `%{customdata[0]} measured<br>t=%{customdata[1]:.2f} ms<br>R=%{x:.2f} ohm<br>X=%{y:.2f} ohm<extra></extra>`,
  };
}

function lineImpedanceTrace(zones: Zone[]): Partial<Plotly.ScatterData> | null {
  const source = zones.find((zone) => Number.isFinite(zone.line_angle_deg)) ?? zones[0];
  if (!source) return null;
  const angle = (source.line_angle_deg * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  if (Math.abs(cos) < 1e-6 && Math.abs(sin) < 1e-6) return null;
  const bounds = boundsFromZones(zones);
  const reach = Math.max(
    10,
    Math.abs(bounds.xMin),
    Math.abs(bounds.xMax),
    Math.abs(bounds.yMin),
    Math.abs(bounds.yMax),
  ) * 1.5;
  if (!Number.isFinite(reach) || reach <= 0) return null;
  return {
    x: [0, reach * cos],
    y: [0, reach * sin],
    type: "scatter",
    mode: "lines",
    name: "Line impedance",
    line: { color: "#334155", width: 1.2, dash: "dashdot" },
    hovertemplate: `Line impedance<br>angle=${source.line_angle_deg.toFixed(1)} deg<extra></extra>`,
  };
}

function closestPointAtTime(allPoints: LocusPoint[], eventMs: number): LocusPoint | null {
  if (!allPoints.length) return null;
  let closest: LocusPoint | null = null;
  let minDiff = Infinity;
  for (const p of allPoints) {
    const diff = Math.abs(p.t * 1000 - eventMs);
    if (diff < minDiff) { minDiff = diff; closest = p; }
  }
  return closest;
}

function eventTrace(
  label: string,
  loops: LoopName[],
  pointsByLoop: Partial<Record<LoopName, LocusPoint[]>>,
  cursorMs: number,
  color: string,
): Partial<Plotly.ScatterData> | null {
  const hits = loops.flatMap((loop) => {
    const point = closestPointAtTime(pointsByLoop[loop] ?? [], cursorMs);
    return point ? [{ loop, point }] : [];
  });
  if (!hits.length) return null;

  return {
    x: hits.map((hit) => hit.point.r),
    y: hits.map((hit) => hit.point.x),
    customdata: hits.map((hit) => [hit.loop, cursorMs]),
    type: "scatter",
    mode: "markers",
    name: label,
    marker: { color, size: 13, symbol: "cross", line: { color: "#111827", width: 1.2 } },
    showlegend: true,
    hovertemplate: `%{customdata[0]} ${label}<br>t=%{customdata[1]:.1f} ms<br>R=%{x:.2f} Ω<br>X=%{y:.2f} Ω<extra></extra>`,
  };
}

function headTrace(loop: LoopName, points: LocusPoint[], currentMs: number): Partial<Plotly.ScatterData> | null {
  const eligible = points.filter((point) => point.t * 1000 <= currentMs);
  const point = eligible[eligible.length - 1];
  if (!point) return null;
  return {
    x: [point.r],
    y: [point.x],
    customdata: [point.t * 1000],
    type: "scatter",
    mode: "markers",
    name: `${loop} @ t`,
    marker: {
      color: LOOP_COLORS[loop],
      size: 8,
      symbol: "circle",
      line: { color: "#ffffff", width: 1.5 },
    },
    showlegend: false,
    hovertemplate: `${loop}<br>t=%{customdata:.2f} ms<br>R=%{x:.2f} ohm<br>X=%{y:.2f} ohm<extra></extra>`,
  };
}

function boundsFromZones(zones: Zone[]) {
  const bounds = { xMin: Infinity, xMax: -Infinity, yMin: Infinity, yMax: -Infinity };

  zones.forEach((zone) => {
    if (zone.shape === "mho") {
      bounds.xMin = Math.min(bounds.xMin, zone.center_r - zone.radius);
      bounds.xMax = Math.max(bounds.xMax, zone.center_r + zone.radius);
      bounds.yMin = Math.min(bounds.yMin, zone.center_x - zone.radius);
      bounds.yMax = Math.max(bounds.yMax, zone.center_x + zone.radius);
      return;
    }

    if (zone.poly_r && zone.poly_x) {
      zone.poly_r.forEach((r, i) => {
        bounds.xMin = Math.min(bounds.xMin, r);
        bounds.xMax = Math.max(bounds.xMax, r);
        bounds.yMin = Math.min(bounds.yMin, zone.poly_x![i]);
        bounds.yMax = Math.max(bounds.yMax, zone.poly_x![i]);
      });
      return;
    }

    quadVertices(zone).forEach(([r, x]) => {
      bounds.xMin = Math.min(bounds.xMin, r);
      bounds.xMax = Math.max(bounds.xMax, r);
      bounds.yMin = Math.min(bounds.yMin, x);
      bounds.yMax = Math.max(bounds.yMax, x);
    });
  });

  return bounds;
}

function boundsFromPoints(pointsByLoop: Partial<Record<LoopName, LocusPoint[]>>, loops: LoopName[]) {
  const bounds = { xMin: Infinity, xMax: -Infinity, yMin: Infinity, yMax: -Infinity };

  loops.forEach((loop) => {
    (pointsByLoop[loop] ?? []).forEach((point) => {
      bounds.xMin = Math.min(bounds.xMin, point.r);
      bounds.xMax = Math.max(bounds.xMax, point.r);
      bounds.yMin = Math.min(bounds.yMin, point.x);
      bounds.yMax = Math.max(bounds.yMax, point.x);
    });
  });

  return bounds;
}

function mergeBounds(a: { xMin: number; xMax: number; yMin: number; yMax: number }, b: { xMin: number; xMax: number; yMin: number; yMax: number }) {
  return {
    xMin: Math.min(a.xMin, b.xMin),
    xMax: Math.max(a.xMax, b.xMax),
    yMin: Math.min(a.yMin, b.yMin),
    yMax: Math.max(a.yMax, b.yMax),
  };
}

function finalizeRange(bounds: { xMin: number; xMax: number; yMin: number; yMax: number }): { x: [number, number]; y: [number, number] } {
  if (!Number.isFinite(bounds.xMin) || !Number.isFinite(bounds.xMax) || !Number.isFinite(bounds.yMin) || !Number.isFinite(bounds.yMax)) {
    return { x: [-40, 40], y: [-40, 40] as [number, number] };
  }

  const xSpan = Math.max(10, bounds.xMax - bounds.xMin);
  const ySpan = Math.max(10, bounds.yMax - bounds.yMin);
  const padX = Math.max(4, xSpan * 0.15);
  const padY = Math.max(4, ySpan * 0.15);
  return {
    x: [bounds.xMin - padX, bounds.xMax + padX] as [number, number],
    y: [bounds.yMin - padY, bounds.yMax + padY] as [number, number],
  };
}

function rangeFromRelayout(event: Readonly<Record<string, unknown>>): PlotRange | null {
  if (event["xaxis.autorange"] || event["yaxis.autorange"]) return { x: undefined, y: undefined };

  const x0 = event["xaxis.range[0]"];
  const x1 = event["xaxis.range[1]"];
  const y0 = event["yaxis.range[0]"];
  const y1 = event["yaxis.range[1]"];
  const next: PlotRange = {};

  if (typeof x0 === "number" && typeof x1 === "number") next.x = [x0, x1];
  if (typeof y0 === "number" && typeof y1 === "number") next.y = [y0, y1];

  return next.x || next.y ? next : null;
}

function zoneLabelAnnotations(zones: Zone[]): Partial<Plotly.Annotations>[] {
  return zones.flatMap((zone) => {
    let xs: number[] = [];
    let ys: number[] = [];
    if (zone.shape === "mho") {
      xs = [zone.center_r];
      ys = [zone.center_x + zone.radius * 0.55];
    } else if (zone.poly_r && zone.poly_x) {
      xs = zone.poly_r;
      ys = zone.poly_x;
    } else {
      const points = quadVertices(zone);
      xs = points.map(([r]) => r);
      ys = points.map(([, x]) => x);
    }
    const finite = xs.map((r, idx) => ({ r, x: ys[idx] })).filter((point) => Number.isFinite(point.r) && Number.isFinite(point.x));
    if (!finite.length) return [];
    const r = finite.reduce((sum, point) => sum + point.r, 0) / finite.length;
    const x = finite.reduce((sum, point) => sum + point.x, 0) / finite.length;
    return [{
      x: r,
      y: x,
      text: zone.label,
      showarrow: false,
      font: { size: 11, color: zone.color },
      bgcolor: "rgba(255,255,255,0.82)",
      bordercolor: zone.color,
      borderpad: 2,
    }];
  });
}

function familyLayout(
  title: string,
  xRange: [number, number],
  yRange: [number, number],
  currentMs?: number,
  detailMode: DetailMode = "standard",
  zones: Zone[] = [],
): Partial<Plotly.Layout> {
  const detailed = detailMode === "detailed";
  const spanX = Math.max(1, xRange[1] - xRange[0]);
  const spanY = Math.max(1, yRange[1] - yRange[0]);
  const majorDtick = Math.max(1, Math.round(Math.max(spanX, spanY) / 10 / 2) * 2);
  return {
    uirevision: title,
    height: detailed ? 560 : 460,
    margin: { t: 36, b: 56, l: 64, r: 20 },
    autosize: true,
    xaxis: {
      title: { text: "R (secondary ohm)" },
      range: xRange,
      zeroline: false,
      tickfont: { size: detailed ? 11 : 10 },
      showgrid: detailed,
      gridcolor: "#e2e8f0",
      dtick: detailed ? majorDtick : undefined,
      ticks: detailed ? "outside" : undefined,
      mirror: detailed,
      showline: detailed,
      linecolor: detailed ? "#475569" : undefined,
    },
    yaxis: {
      title: { text: "X (secondary ohm)" },
      range: yRange,
      zeroline: false,
      tickfont: { size: detailed ? 11 : 10 },
      showgrid: detailed,
      gridcolor: "#e2e8f0",
      dtick: detailed ? majorDtick : undefined,
      ticks: detailed ? "outside" : undefined,
      mirror: detailed,
      showline: detailed,
      linecolor: detailed ? "#475569" : undefined,
      scaleanchor: "x",
      scaleratio: 1,
      constrain: "domain",
    },
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
    hovermode: "closest",
    title: { text: title, font: { size: 12 } },
    legend: { orientation: "h", y: -0.14, font: { size: 10 } },
    shapes: [
      { type: "line", x0: xRange[0], x1: xRange[1], y0: 0, y1: 0, line: { color: detailed ? "#334155" : "#cbd5e1", width: detailed ? 1.3 : 1 } },
      { type: "line", x0: 0, x1: 0, y0: yRange[0], y1: yRange[1], line: { color: detailed ? "#334155" : "#cbd5e1", width: detailed ? 1.3 : 1 } },
    ] as Plotly.Shape[],
    annotations: [
      ...(detailed ? zoneLabelAnnotations(zones) : []),
      ...(currentMs !== undefined ? [{
      xref: "paper" as const,
      yref: "paper" as const,
      x: 1,
      y: 1.08,
      xanchor: "right" as const,
      text: `t <= ${currentMs.toFixed(2)} ms`,
      showarrow: false,
      font: { size: 10, color: "#475569" },
      bgcolor: "rgba(255,255,255,0.9)",
      bordercolor: "#cbd5e1",
      borderpad: 3,
    }] : []),
    ],
  };
}

export default function ImpedanceLocus({ analysisId, dataRevision = 0 }: { analysisId: string; dataRevision?: number }) {
  const [groundZones, setGroundZones] = useState<Zone[]>([]);
  const [phaseZones, setPhaseZones] = useState<Zone[]>([]);
  const [pointsByLoop, setPointsByLoop] = useState<Partial<Record<LoopName, LocusPoint[]>>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [relayStatus, setRelayStatus] = useState("Belum ada file relay dimuat; zona proteksi tidak digambar.");
  const [rioOver, setRioOver] = useState(false);
  const [timeMode, setTimeMode] = useState<TimeMode>("all");
  const [timing, setTiming] = useState<{
    inceptionMs: number | null;
    durationMs: number | null;
    tripMs: number | null;
    tripSource: "soe" | "status_edge" | "estimated" | null;
  }>({
    inceptionMs: null,
    durationMs: null,
    tripMs: null,
    tripSource: null,
  });
  const [faultClassification, setFaultClassification] = useState<FaultClassification21 | null>(null);
  const [playMs, setPlayMs] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [replaySpeed, setReplaySpeed] = useState(0.5);
  const [detailMode, setDetailMode] = useState<DetailMode>("standard");
  const [plotRanges, setPlotRanges] = useState<Record<PlotFamily, PlotRange>>({ ground: {}, phase: {} });
  const [ctRatioOverride, setCtRatioOverride] = useState<number | null>(null);
  const [vtRatioOverride, setVtRatioOverride] = useState<number | null>(null);
  const [ratioNotice, setRatioNotice] = useState("CT/VT locus memakai rasio dari CFG/parser sampai nilai RIO/XRIO atau manual diisi.");
  const rioInputRef = useRef<HTMLInputElement>(null);

  async function fetchAllLoci(
    nextGroundZones = groundZones,
    nextPhaseZones = phaseZones,
    nextCtRatio = ctRatioOverride,
    nextVtRatio = vtRatioOverride,
  ) {
    setLoading(true);
    setError(null);
    try {
      const results = await Promise.all(
        [...GROUND_LOOPS, ...PHASE_LOOPS].map(async (loop) => {
          const zones = GROUND_LOOPS.includes(loop) ? nextGroundZones : nextPhaseZones;
          const response = await computeLocus(
            analysisId, zones, loop, 0, 0, false,
            nextCtRatio ?? undefined, nextVtRatio ?? undefined,
          );
          return [loop, response.points ?? []] as const;
        })
      );

      const nextPoints: Partial<Record<LoopName, LocusPoint[]>> = {};
      results.forEach(([loop, points]) => {
        nextPoints[loop] = points;
      });
      setPointsByLoop(nextPoints);

      const totalPoints = Object.values(nextPoints).reduce((sum, points) => sum + (points?.length ?? 0), 0);
      if (totalPoints === 0) {
        setError("Tidak ada trajectory impedansi yang terbentuk. Pastikan kanal tegangan dan arus tersedia.");
      }
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "Failed to compute impedance locus.";
      setError(`${message} Check that usable voltage and current channels are present.`);
      setPointsByLoop({});
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void fetchAllLoci();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataRevision, detailMode]);

  useEffect(() => {
    let alive = true;
    Promise.allSettled([fetchElectricalParams21(analysisId), fetchFaultClassification21(analysisId)])
      .then(([paramsResult, classificationResult]) => {
        if (!alive) return;
        if (paramsResult.status === "fulfilled") {
          const params = paramsResult.value;
          setTiming({
            inceptionMs: typeof params.inception_time_ms === "number" ? params.inception_time_ms : null,
            durationMs: typeof params.fault_duration_ms === "number" ? params.fault_duration_ms : null,
            tripMs: typeof params.trip_time_ms === "number" ? params.trip_time_ms : null,
            tripSource: typeof params.trip_time_ms === "number" ? (params.trip_time_source ?? "status_edge") : null,
          });
        } else {
          setTiming({ inceptionMs: null, durationMs: null, tripMs: null, tripSource: null });
        }
        setFaultClassification(classificationResult.status === "fulfilled" ? classificationResult.value : null);
      });
    return () => { alive = false; };
  }, [analysisId, dataRevision]);

  function updateZoneSet(family: ZoneFamily, updater: (zones: Zone[]) => Zone[]) {
    if (family === "ground") {
      setGroundZones((prev) => updater(prev));
      return;
    }
    setPhaseZones((prev) => updater(prev));
  }

  function updateZone(family: ZoneFamily, idx: number, field: keyof Zone, value: string | number) {
    updateZoneSet(family, (zones) =>
      zones.map((zone, zoneIdx) => (zoneIdx === idx ? { ...zone, [field]: value } : zone))
    );
  }

  function updateRatioOverride(kind: "ct" | "vt", raw: string) {
    const value = raw.trim() === "" ? null : Number.parseFloat(raw);
    if (value !== null && (!Number.isFinite(value) || value <= 0)) return;
    if (kind === "ct") setCtRatioOverride(value);
    else setVtRatioOverride(value);
    setRatioNotice("CT/VT locus diubah manual. Tekan Apply CT/VT untuk menghitung ulang locus dengan rasio ini.");
  }

  function polygonVertexRows(zone: Zone) {
    if (!zone.poly_r || !zone.poly_x) return [];
    const count = Math.min(zone.poly_r.length, zone.poly_x.length);
    if (!count) return [];
    const closesToStart =
      count > 2 &&
      Math.abs(zone.poly_r[0] - zone.poly_r[count - 1]) < 1e-9 &&
      Math.abs(zone.poly_x[0] - zone.poly_x[count - 1]) < 1e-9;
    const visibleCount = closesToStart ? count - 1 : count;
    return Array.from({ length: visibleCount }, (_, vertexIdx) => ({
      vertexIdx,
      r: zone.poly_r?.[vertexIdx] ?? 0,
      x: zone.poly_x?.[vertexIdx] ?? 0,
    }));
  }

  function updatePolygonVertex(
    family: ZoneFamily,
    zoneIndex: number,
    vertexIndex: number,
    axis: "r" | "x",
    value: number
  ) {
    if (!Number.isFinite(value)) return;
    updateZoneSet(family, (zones) =>
      zones.map((zone, idx) => {
        if (idx !== zoneIndex || !zone.poly_r || !zone.poly_x) return zone;
        const nextR = [...zone.poly_r];
        const nextX = [...zone.poly_x];
        const count = Math.min(nextR.length, nextX.length);
        if (vertexIndex < 0 || vertexIndex >= count) return zone;
        const closesToStart =
          count > 2 &&
          Math.abs(nextR[0] - nextR[count - 1]) < 1e-9 &&
          Math.abs(nextX[0] - nextX[count - 1]) < 1e-9;

        if (axis === "r") nextR[vertexIndex] = value;
        else nextX[vertexIndex] = value;

        if (closesToStart && vertexIndex === 0) {
          nextR[count - 1] = nextR[0];
          nextX[count - 1] = nextX[0];
        }

        return { ...zone, poly_r: nextR, poly_x: nextX };
      })
    );
  }

  function shapeChoiceFor(zone?: Zone): ZoneShapeChoice {
    if (!zone || zone.shape === "mho") return "mho";
    return zone.reach_mode === "z" ? "quad_z" : "quad_rx";
  }

  function zoneForShapeChoice(zone: Zone, choice: ZoneShapeChoice): Zone {
    if (choice === "mho") {
      return { ...zone, shape: "mho", poly_r: undefined, poly_x: undefined };
    }

    if (choice === "quad_z") {
      const zFwd = zone.z_fwd ?? reachFromReactance(zone.xf, zone.line_angle_deg);
      const zRev = zone.z_rev ?? reachFromReactance(zone.xr, zone.line_angle_deg);
      return {
        ...zone,
        shape: "quad",
        reach_mode: "z",
        z_fwd: Number.isFinite(zFwd) ? zFwd : 0,
        z_rev: Number.isFinite(zRev) ? zRev : 0,
        poly_r: undefined,
        poly_x: undefined,
      };
    }

    const xf = zone.reach_mode === "z"
      ? reactanceFromReach(zone.z_fwd ?? zone.xf, zone.line_angle_deg)
      : zone.xf;
    const xr = zone.reach_mode === "z"
      ? reactanceFromReach(zone.z_rev ?? zone.xr, zone.line_angle_deg)
      : zone.xr;
    return {
      ...zone,
      shape: "quad",
      reach_mode: "rx",
      xf: Number.isFinite(xf) ? xf : 0,
      xr: Number.isFinite(xr) ? xr : 0,
      poly_r: undefined,
      poly_x: undefined,
    };
  }

  function updateFamilyShape(family: ZoneFamily, choice: ZoneShapeChoice) {
    updateZoneSet(family, (zones) => zones.map((zone) => zoneForShapeChoice(zone, choice)));
  }

  async function handleRelayFile(file: File) {
    try {
      const text = await file.text();
      const imported = file.name.toLowerCase().endsWith(".xrio") ? parseXRIO(text) : parseRIO(text);
      if (!imported) {
        setRelayStatus("Format relay tidak bisa diparse. Gunakan file .rio atau .xrio yang valid.");
        return;
      }

      const renderableGroundZones = imported.phGnd.filter(isRenderableImportedZone);
      const renderablePhaseZones = imported.phPh.filter(isRenderableImportedZone);

      const nextGroundZones = renderableGroundZones.slice(0, 3).map((zone, idx) =>
        importedZoneToConfig(zone, { ...GROUND_ZONE_TEMPLATES[idx % GROUND_ZONE_TEMPLATES.length] })
      );
      const nextPhaseZones = renderablePhaseZones.slice(0, 3).map((zone, idx) =>
        importedZoneToConfig(zone, { ...PHASE_ZONE_TEMPLATES[idx % PHASE_ZONE_TEMPLATES.length] })
      );

      setGroundZones(nextGroundZones);
      setPhaseZones(nextPhaseZones);

      let nextCtRatio = ctRatioOverride;
      let nextVtRatio = vtRatioOverride;
      if (imported.ctRatio != null && imported.vtRatio != null) {
        const previousRatioNote = ctRatioOverride != null && vtRatioOverride != null
          ? ` Sebelumnya CT=${ctRatioOverride.toFixed(2)}:1, VT=${vtRatioOverride.toFixed(2)}:1.`
          : "";
        nextCtRatio = imported.ctRatio;
        nextVtRatio = imported.vtRatio;
        setCtRatioOverride(nextCtRatio);
        setVtRatioOverride(nextVtRatio);
        setRatioNotice(
          `CT/VT dari ${file.name} dipakai untuk locus: CT=${nextCtRatio.toFixed(2)}:1, VT=${nextVtRatio.toFixed(2)}:1.${previousRatioNote} Koreksi manual bila tidak sesuai nameplate/setting relay.`
        );
      } else {
        setRatioNotice(
          `File ${file.name} tidak memuat pasangan CT dan VT yang lengkap. Locus tetap memakai rasio CFG/parser atau nilai manual yang sedang aktif.`
        );
      }

      const gndCount = imported.phGnd.length;
      const phCount = imported.phPh.length;
      const polyCount = [...imported.phGnd, ...imported.phPh].filter((z) => z.shapeType === "poly").length;
      const polyNote = polyCount > 0 ? ` (${polyCount} zona polygon dari .rio)` : "";
      const ratioNote = nextCtRatio != null && nextVtRatio != null
        ? ` | CT=${nextCtRatio.toFixed(0)}:1, VT=${nextVtRatio.toFixed(1)}:1`
        : "";
      setRelayStatus(`${file.name} dimuat: ${gndCount} zona ground, ${phCount} zona phase.${polyNote}${ratioNote}`);
      await fetchAllLoci(nextGroundZones, nextPhaseZones, nextCtRatio, nextVtRatio);
    } catch {
      setRelayStatus("Gagal membaca file relay.");
    }
  }

  const allTimeRange = useMemo((): [number, number] => {
    const times = Object.values(pointsByLoop)
      .flatMap((points) => points ?? [])
      .map((point) => point.t * 1000)
      .filter((value) => Number.isFinite(value));
    if (!times.length) return [0, 0];
    return [Math.min(...times), Math.max(...times)];
  }, [pointsByLoop]);

  const activeTimeRange = useMemo((): [number, number] => {
    if (timeMode === "fault" && timing.inceptionMs !== null && timing.durationMs !== null) {
      // 20ms pre-inception + fault + 50% of fault duration post-trip (min 100ms)
      const postTail = Math.max(100, timing.durationMs * 0.5);
      const start = Math.max(allTimeRange[0], timing.inceptionMs - 20);
      const end = Math.min(allTimeRange[1], timing.inceptionMs + timing.durationMs + postTail);
      if (end > start) return [start, end];
    }
    return allTimeRange;
  }, [allTimeRange, timeMode, timing]);

  useEffect(() => {
    setPlayMs(activeTimeRange[1]);
    setPlaying(false);
  }, [activeTimeRange[0], activeTimeRange[1], analysisId, dataRevision]);

  useEffect(() => {
    setPlotRanges({ ground: {}, phase: {} });
  }, [analysisId, dataRevision, timeMode]);

  useEffect(() => {
    if (!playing) return undefined;
    const span = Math.max(activeTimeRange[1] - activeTimeRange[0], 1);
    const step = Math.max((span / 300) * replaySpeed, 0.25);
    const timer = window.setInterval(() => {
      setPlayMs((prev) => {
        const next = Math.min((prev ?? activeTimeRange[0]) + step, activeTimeRange[1]);
        if (next >= activeTimeRange[1]) {
          window.clearInterval(timer);
          setPlaying(false);
        }
        return next;
      });
    }, 80);
    return () => window.clearInterval(timer);
  }, [activeTimeRange, playing, replaySpeed]);

  const currentPlayMs = playMs ?? activeTimeRange[1];
  const visiblePointsByLoop = useMemo(() => {
    const next: Partial<Record<LoopName, LocusPoint[]>> = {};
    ([...GROUND_LOOPS, ...PHASE_LOOPS] as LoopName[]).forEach((loop) => {
      next[loop] = (pointsByLoop[loop] ?? []).filter((point) => {
        const tMs = point.t * 1000;
        return tMs >= activeTimeRange[0] && tMs <= currentPlayMs;
      });
    });
    return next;
  }, [activeTimeRange, currentPlayMs, pointsByLoop]);

  const activeWindowLabel =
    timeMode === "fault" && timing.inceptionMs !== null && timing.durationMs !== null
      ? `Fault window ${activeTimeRange[0].toFixed(1)}-${activeTimeRange[1].toFixed(1)} ms`
      : `Full record ${activeTimeRange[0].toFixed(1)}-${activeTimeRange[1].toFixed(1)} ms`;
  const relayTripMs =
    timing.tripMs !== null
      ? timing.tripMs
      : (timing.inceptionMs !== null && timing.durationMs !== null ? timing.inceptionMs + timing.durationMs : null);
  const relayTripSource = timing.tripMs !== null
    ? timing.tripSource
    : (relayTripMs !== null ? "estimated" : null);
  const relayTripTraceLabel = relayTripSource === "soe"
    ? "Relay trip time (SOE)"
    : relayTripSource === "status_edge"
      ? "Relay trip time (digital edge)"
      : "Estimated clearing time";
  const groundEventLoops = useMemo(
    () => eventLoopsForFamily("ground", faultClassification),
    [faultClassification],
  );
  const phaseEventLoops = useMemo(
    () => eventLoopsForFamily("phase", faultClassification),
    [faultClassification],
  );
  const markerScopeLabel = faultClassification
    ? `${faultClassification.to_ground ? "Phase-to-ground" : "Phase-to-phase"} ${faultClassification.phases_label || faultClassification.phases?.join("+") || ""}`.trim()
    : "Belum teridentifikasi";
  const hasRelayZones = groundZones.length > 0 || phaseZones.length > 0;

  const groundTraces = useMemo(() => {
    const loci = GROUND_LOOPS.filter((loop) => (visiblePointsByLoop[loop] ?? []).length > 0).map((loop) =>
      loopTrace(loop, visiblePointsByLoop[loop] ?? [], detailMode)
    );
    const directions = detailMode === "detailed"
      ? GROUND_LOOPS.map((loop) => directionTrace(loop, visiblePointsByLoop[loop] ?? [])).filter(Boolean)
      : [];
    const showPlayHead = currentPlayMs < activeTimeRange[1] - 0.01;
    const heads = showPlayHead
      ? GROUND_LOOPS.map((loop) => headTrace(loop, visiblePointsByLoop[loop] ?? [], currentPlayMs)).filter(Boolean)
      : [];
    const inception = timing.inceptionMs !== null && groundEventLoops.length
      ? eventTrace("Inception marker", groundEventLoops, pointsByLoop, timing.inceptionMs, "#facc15")
      : null;
    const trip = relayTripMs !== null
      ? eventTrace(relayTripTraceLabel, GROUND_LOOPS, pointsByLoop, relayTripMs, "#2563eb")
      : null;
    const zoneTraces = groundZones.map((zone) => (zone.shape === "mho" ? mhoCircleTrace(zone) : quadTrace(zone)));
    const measured = detailMode === "detailed" ? measuredTrace(GROUND_LOOPS, visiblePointsByLoop) : null;
    const line = detailMode === "detailed" ? lineImpedanceTrace(groundZones) : null;
    if (detailMode !== "detailed") {
      return [...loci, ...(heads as Plotly.Data[]), ...(inception ? [inception as Plotly.Data] : []), ...(trip ? [trip as Plotly.Data] : []), ...zoneTraces] as Plotly.Data[];
    }
    return [
      ...zoneTraces,
      ...(line ? [line as Plotly.Data] : []),
      ...loci,
      ...(directions as Plotly.Data[]),
      ...(heads as Plotly.Data[]),
      ...(measured ? [measured as Plotly.Data] : []),
      ...(inception ? [inception as Plotly.Data] : []),
      ...(trip ? [trip as Plotly.Data] : []),
    ] as Plotly.Data[];
  }, [activeTimeRange, currentPlayMs, detailMode, groundEventLoops, groundZones, pointsByLoop, relayTripMs, relayTripTraceLabel, timing.inceptionMs, visiblePointsByLoop]);

  const phaseTraces = useMemo(() => {
    const loci = PHASE_LOOPS.filter((loop) => (visiblePointsByLoop[loop] ?? []).length > 0).map((loop) =>
      loopTrace(loop, visiblePointsByLoop[loop] ?? [], detailMode)
    );
    const directions = detailMode === "detailed"
      ? PHASE_LOOPS.map((loop) => directionTrace(loop, visiblePointsByLoop[loop] ?? [])).filter(Boolean)
      : [];
    const showPlayHead = currentPlayMs < activeTimeRange[1] - 0.01;
    const heads = showPlayHead
      ? PHASE_LOOPS.map((loop) => headTrace(loop, visiblePointsByLoop[loop] ?? [], currentPlayMs)).filter(Boolean)
      : [];
    const inception = timing.inceptionMs !== null && phaseEventLoops.length
      ? eventTrace("Inception marker", phaseEventLoops, pointsByLoop, timing.inceptionMs, "#facc15")
      : null;
    const trip = relayTripMs !== null
      ? eventTrace(relayTripTraceLabel, PHASE_LOOPS, pointsByLoop, relayTripMs, "#2563eb")
      : null;
    const zoneTraces = phaseZones.map((zone) => (zone.shape === "mho" ? mhoCircleTrace(zone) : quadTrace(zone)));
    const measured = detailMode === "detailed" ? measuredTrace(PHASE_LOOPS, visiblePointsByLoop) : null;
    const line = detailMode === "detailed" ? lineImpedanceTrace(phaseZones) : null;
    if (detailMode !== "detailed") {
      return [...loci, ...(heads as Plotly.Data[]), ...(inception ? [inception as Plotly.Data] : []), ...(trip ? [trip as Plotly.Data] : []), ...zoneTraces] as Plotly.Data[];
    }
    return [
      ...zoneTraces,
      ...(line ? [line as Plotly.Data] : []),
      ...loci,
      ...(directions as Plotly.Data[]),
      ...(heads as Plotly.Data[]),
      ...(measured ? [measured as Plotly.Data] : []),
      ...(inception ? [inception as Plotly.Data] : []),
      ...(trip ? [trip as Plotly.Data] : []),
    ] as Plotly.Data[];
  }, [activeTimeRange, currentPlayMs, detailMode, phaseEventLoops, phaseZones, pointsByLoop, relayTripMs, relayTripTraceLabel, timing.inceptionMs, visiblePointsByLoop]);

  // When zones are loaded, merge zone bounds with locus points but cap at 4× the
  // zone span so pre-fault load impedance (100-150 Ω away) doesn't crush the view.
  const groundRanges = useMemo(() => {
    const pointBounds = boundsFromPoints(pointsByLoop, GROUND_LOOPS);
    if (groundZones.length === 0) return finalizeRange(pointBounds);
    const zoneBounds = boundsFromZones(groundZones);
    const zoneSpanR = Math.max(10, zoneBounds.xMax - zoneBounds.xMin);
    const zoneSpanX = Math.max(10, zoneBounds.yMax - zoneBounds.yMin);
    const capped = {
      xMin: Math.max(pointBounds.xMin, zoneBounds.xMin - zoneSpanR * 2),
      xMax: Math.min(pointBounds.xMax, zoneBounds.xMax + zoneSpanR * 4),
      yMin: Math.max(pointBounds.yMin, zoneBounds.yMin - zoneSpanX * 2),
      yMax: Math.min(pointBounds.yMax, zoneBounds.yMax + zoneSpanX * 2),
    };
    return finalizeRange(mergeBounds(zoneBounds, capped));
  }, [groundZones, pointsByLoop]);

  const phaseRanges = useMemo(() => {
    const pointBounds = boundsFromPoints(pointsByLoop, PHASE_LOOPS);
    if (phaseZones.length === 0) return finalizeRange(pointBounds);
    const zoneBounds = boundsFromZones(phaseZones);
    const zoneSpanR = Math.max(10, zoneBounds.xMax - zoneBounds.xMin);
    const zoneSpanX = Math.max(10, zoneBounds.yMax - zoneBounds.yMin);
    const capped = {
      xMin: Math.max(pointBounds.xMin, zoneBounds.xMin - zoneSpanR * 2),
      xMax: Math.min(pointBounds.xMax, zoneBounds.xMax + zoneSpanR * 4),
      yMin: Math.max(pointBounds.yMin, zoneBounds.yMin - zoneSpanX * 2),
      yMax: Math.min(pointBounds.yMax, zoneBounds.yMax + zoneSpanX * 2),
    };
    return finalizeRange(mergeBounds(zoneBounds, capped));
  }, [phaseZones, pointsByLoop]);

  const groundShape = shapeChoiceFor(groundZones[0]);
  const phaseShape = shapeChoiceFor(phaseZones[0]);
  const groundViewRange = {
    x: plotRanges.ground.x ?? groundRanges.x,
    y: plotRanges.ground.y ?? groundRanges.y,
  };
  const phaseViewRange = {
    x: plotRanges.phase.x ?? phaseRanges.x,
    y: plotRanges.phase.y ?? phaseRanges.y,
  };

  function rememberPlotRange(family: PlotFamily, event: Readonly<Record<string, unknown>>) {
    const range = rangeFromRelayout(event);
    if (!range) return;
    setPlotRanges((prev) => ({
      ...prev,
      [family]: {
        x: Object.prototype.hasOwnProperty.call(range, "x") ? range.x : prev[family].x,
        y: Object.prototype.hasOwnProperty.call(range, "y") ? range.y : prev[family].y,
      },
    }));
  }

  function renderZoneEditor(title: string, family: ZoneFamily, zones: Zone[], familyShape: ZoneShapeChoice) {
    return (
      <div className={styles.locusEditorSection}>
        <div className={styles.locusEditorHeader}>
          <h3 className={styles.locusEditorTitle}>{title}</h3>
          <div className={styles.controls}>
            <label className={styles.label}>Zone Shape</label>
            <select
              className={styles.selectField}
              value={familyShape}
              onChange={(e) => updateFamilyShape(family, e.target.value as ZoneShapeChoice)}
            >
              <option value="mho">Mho</option>
              <option value="quad_rx">Quadrilateral R/X</option>
              <option value="quad_z">Quadrilateral Z reach</option>
            </select>
          </div>
        </div>

        {zones.map((zone, idx) => (
          <div key={`${family}-${zone.label}`} className={styles.locusZoneCard}>
            <div className={styles.row}>
              <span className={styles.label} style={{ fontWeight: 700, width: 28 }}>{zone.label}</span>
              <input
                type="color"
                value={zone.color}
                onChange={(e) => updateZone(family, idx, "color", e.target.value)}
                style={{ width: 34, height: 28, border: "none", cursor: "pointer", background: "transparent" }}
              />
            </div>

            {familyShape === "mho" ? (
              <div className={styles.zoneEditorRow}>
                <label className={styles.zoneLabel}>
                  Center R
                  <input
                    className={styles.inputField}
                    type="number"
                    value={zone.center_r}
                    onChange={(e) => updateZone(family, idx, "center_r", Number.parseFloat(e.target.value))}
                  />
                </label>
                <label className={styles.zoneLabel}>
                  Center X
                  <input
                    className={styles.inputField}
                    type="number"
                    value={zone.center_x}
                    onChange={(e) => updateZone(family, idx, "center_x", Number.parseFloat(e.target.value))}
                  />
                </label>
                <label className={styles.zoneLabel}>
                  Radius
                  <input
                    className={styles.inputField}
                    type="number"
                    value={zone.radius}
                    onChange={(e) => updateZone(family, idx, "radius", Number.parseFloat(e.target.value))}
                  />
                </label>
              </div>
            ) : zone.poly_r ? (
              <>
                <div style={{ fontSize: "0.75rem", color: "#64748b", padding: "4px 0 8px" }}>
                  Polygon ({polygonVertexRows(zone).length} vertices) dari file RIO/XRIO. Nilai R/X bisa dikoreksi bila hasil import tidak sesuai.
                </div>
                <div className={styles.zoneEditorRow}>
                  {polygonVertexRows(zone).map((point) => (
                    <div
                      key={`${zone.label}-${point.vertexIdx}`}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "36px minmax(0, 1fr) minmax(0, 1fr)",
                        alignItems: "end",
                        gap: 6,
                      }}
                    >
                      <span style={{ fontSize: "0.74rem", fontWeight: 700, color: "#475569", paddingBottom: 8 }}>
                        V{point.vertexIdx + 1}
                      </span>
                      <label className={styles.zoneLabel}>
                        R
                        <input
                          className={styles.inputField}
                          type="number"
                          step="0.01"
                          value={point.r}
                          onChange={(e) =>
                            updatePolygonVertex(family, idx, point.vertexIdx, "r", Number.parseFloat(e.target.value))
                          }
                        />
                      </label>
                      <label className={styles.zoneLabel}>
                        X
                        <input
                          className={styles.inputField}
                          type="number"
                          step="0.01"
                          value={point.x}
                          onChange={(e) =>
                            updatePolygonVertex(family, idx, point.vertexIdx, "x", Number.parseFloat(e.target.value))
                          }
                        />
                      </label>
                    </div>
                  ))}
                </div>
              </>
            ) : familyShape === "quad_z" ? (
              <div className={styles.zoneEditorRow}>
                <label className={styles.zoneLabel}>
                  Z Forward
                  <input
                    className={styles.inputField}
                    type="number"
                    value={zone.z_fwd ?? reachFromReactance(zone.xf, zone.line_angle_deg)}
                    onChange={(e) => updateZone(family, idx, "z_fwd", Number.parseFloat(e.target.value))}
                  />
                </label>
                <label className={styles.zoneLabel}>
                  Z Reverse
                  <input
                    className={styles.inputField}
                    type="number"
                    value={zone.z_rev ?? reachFromReactance(zone.xr, zone.line_angle_deg)}
                    onChange={(e) => updateZone(family, idx, "z_rev", Number.parseFloat(e.target.value))}
                  />
                </label>
                <label className={styles.zoneLabel}>
                  Rf Forward
                  <input
                    className={styles.inputField}
                    type="number"
                    value={zone.rf_fwd}
                    onChange={(e) => updateZone(family, idx, "rf_fwd", Number.parseFloat(e.target.value))}
                  />
                </label>
                <label className={styles.zoneLabel}>
                  Rf Reverse
                  <input
                    className={styles.inputField}
                    type="number"
                    value={zone.rf_rev}
                    onChange={(e) => updateZone(family, idx, "rf_rev", Number.parseFloat(e.target.value))}
                  />
                </label>
                <label className={styles.zoneLabel}>
                  Reach Angle
                  <input
                    className={styles.inputField}
                    type="number"
                    value={zone.line_angle_deg}
                    onChange={(e) => updateZone(family, idx, "line_angle_deg", Number.parseFloat(e.target.value))}
                  />
                </label>
              </div>
            ) : (
              <div className={styles.zoneEditorRow}>
                <label className={styles.zoneLabel}>
                  Rf Forward
                  <input
                    className={styles.inputField}
                    type="number"
                    value={zone.rf_fwd}
                    onChange={(e) => updateZone(family, idx, "rf_fwd", Number.parseFloat(e.target.value))}
                  />
                </label>
                <label className={styles.zoneLabel}>
                  Rf Reverse
                  <input
                    className={styles.inputField}
                    type="number"
                    value={zone.rf_rev}
                    onChange={(e) => updateZone(family, idx, "rf_rev", Number.parseFloat(e.target.value))}
                  />
                </label>
                <label className={styles.zoneLabel}>
                  X Forward
                  <input
                    className={styles.inputField}
                    type="number"
                    value={zone.xf}
                    onChange={(e) => updateZone(family, idx, "xf", Number.parseFloat(e.target.value))}
                  />
                </label>
                <label className={styles.zoneLabel}>
                  X Reverse
                  <input
                    className={styles.inputField}
                    type="number"
                    value={zone.xr}
                    onChange={(e) => updateZone(family, idx, "xr", Number.parseFloat(e.target.value))}
                  />
                </label>
                <label className={styles.zoneLabel}>
                  Line Angle
                  <input
                    className={styles.inputField}
                    type="number"
                    value={zone.line_angle_deg}
                    onChange={(e) => updateZone(family, idx, "line_angle_deg", Number.parseFloat(e.target.value))}
                  />
                </label>
              </div>
            )}
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Impedance Locus Diagram</h2>
        <div className={styles.controls} style={{ flexWrap: "wrap", gap: 10 }}>
          <button
            className={styles.applyBtn}
            onClick={() => void fetchAllLoci(groundZones, phaseZones, ctRatioOverride, vtRatioOverride)}
            disabled={loading}
          >
            {loading ? "Computing..." : "Refresh All Loci"}
          </button>
        </div>
      </div>

      <div style={{ marginBottom: 12 }}>
        <div className={styles.label} style={{ fontWeight: 600, marginBottom: 6 }}>Import Relay Settings</div>
        <div
          onClick={() => rioInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setRioOver(true); }}
          onDragLeave={() => setRioOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setRioOver(false);
            const file = e.dataTransfer.files[0];
            if (file) void handleRelayFile(file);
          }}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            padding: "10px 14px",
            border: `1.5px dashed ${rioOver ? "#f59e0b" : "#cbd5e1"}`,
            borderRadius: 8,
            background: rioOver ? "#fffbeb" : "#f8fafc",
            cursor: "pointer",
            transition: "border-color 0.15s, background 0.15s",
            fontSize: "0.82rem",
            color: "#475569",
            userSelect: "none",
          }}
        >
          <span style={{ fontSize: "1.1rem" }}>📂</span>
          <span>Click or drag <strong>.rio</strong> / <strong>.xrio</strong> here</span>
          <input
            ref={rioInputRef}
            type="file"
            accept=".rio,.xrio,.RIO,.XRIO"
            style={{ display: "none" }}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void handleRelayFile(file);
              e.target.value = "";
            }}
          />
        </div>
        <div style={{ marginTop: 6, fontSize: "0.78rem", color: "#64748b" }}>{relayStatus}</div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
            gap: 8,
            alignItems: "end",
            marginTop: 10,
          }}
        >
          <label className={styles.zoneLabel}>
            CT ratio used by locus
            <input
              className={styles.inputField}
              type="number"
              min="0"
              step="0.01"
              placeholder="CFG/parser"
              value={ctRatioOverride ?? ""}
              onChange={(event) => updateRatioOverride("ct", event.target.value)}
            />
          </label>
          <label className={styles.zoneLabel}>
            VT ratio used by locus
            <input
              className={styles.inputField}
              type="number"
              min="0"
              step="0.01"
              placeholder="CFG/parser"
              value={vtRatioOverride ?? ""}
              onChange={(event) => updateRatioOverride("vt", event.target.value)}
            />
          </label>
          <button
            type="button"
            className={styles.applyBtn}
            onClick={() => void fetchAllLoci(groundZones, phaseZones, ctRatioOverride, vtRatioOverride)}
            disabled={loading}
          >
            Apply CT/VT
          </button>
        </div>
        <div className={styles.badge} style={{ marginTop: 8, whiteSpace: "normal", lineHeight: 1.45 }}>
          {ratioNotice}
        </div>
      </div>

      {error && (
        <div className={styles.warning} style={{ marginBottom: 12 }}>
          {error}
        </div>
      )}

      <div className={styles.locusTimebar}>
        <div className={styles.locusTimeControls}>
          <label className={styles.zoneLabel}>
            Time window
            <select
              className={styles.selectField}
              value={timeMode}
              onChange={(event) => setTimeMode(event.target.value as TimeMode)}
            >
              <option value="fault">Fault window</option>
              <option value="all">Full record</option>
            </select>
          </label>
          <label className={styles.zoneLabel}>
            Plot detail
            <select
              className={styles.selectField}
              value={detailMode}
              onChange={(event) => setDetailMode(event.target.value as DetailMode)}
            >
              <option value="standard">Standard</option>
              <option value="detailed">Detailed</option>
            </select>
          </label>
          <button
            type="button"
            className={styles.applyBtn}
            onClick={() => {
              if (currentPlayMs >= activeTimeRange[1]) setPlayMs(activeTimeRange[0]);
              setPlaying((value) => !value);
            }}
            disabled={activeTimeRange[1] <= activeTimeRange[0]}
          >
            {playing ? "Pause Locus" : "Play Locus"}
          </button>
          <button
            type="button"
            className={styles.waveGhostBtn}
            onClick={() => {
              setPlaying(false);
              setPlayMs(activeTimeRange[1]);
            }}
          >
            Show End
          </button>
          <label className={styles.zoneLabel}>
            Replay speed
            <select
              className={styles.selectField}
              value={replaySpeed}
              onChange={(event) => setReplaySpeed(Number(event.target.value))}
            >
              <option value={0.25}>0.25x</option>
              <option value={0.5}>0.5x</option>
              <option value={1}>1x</option>
              <option value={2}>2x</option>
            </select>
          </label>
        </div>
        <input
          type="range"
          min={activeTimeRange[0]}
          max={activeTimeRange[1]}
          step={0.5}
          value={Math.min(Math.max(currentPlayMs, activeTimeRange[0]), activeTimeRange[1])}
          onChange={(event) => {
            setPlaying(false);
            setPlayMs(Number(event.target.value));
          }}
          className={styles.locusTimeSlider}
        />
        <div className={styles.locusTimeReadout}>
          <strong>{currentPlayMs.toFixed(2)} ms</strong>
          <span>{activeWindowLabel}</span>
          <span>Inception marker scope: {markerScopeLabel}; trip marker follows SOE/digital trip time on both plots.</span>
          {timing.inceptionMs !== null && timing.durationMs !== null && (
            <span>
              Inception {timing.inceptionMs.toFixed(1)} ms | {relayTripTraceLabel} {relayTripMs !== null ? relayTripMs.toFixed(1) : "-"} ms | FCT {timing.durationMs.toFixed(1)} ms
            </span>
          )}
        </div>
      </div>

      <div className={styles.locusPlotStack}>
        <div className={styles.locusPlotCard}>
          <div className={styles.locusPlotTitle}>Phase-to-Ground | ZA, ZB, ZC</div>
          <Plot
            data={groundTraces}
            layout={familyLayout("Phase-to-Ground", groundViewRange.x, groundViewRange.y, currentPlayMs, detailMode, groundZones)}
            config={{ displayModeBar: true, responsive: true, displaylogo: false, toImageButtonOptions: { scale: detailMode === "detailed" ? 3 : 2 } }}
            style={{ width: "100%" }}
            onRelayout={(event) => rememberPlotRange("ground", event as Readonly<Record<string, unknown>>)}
          />
        </div>

        <div className={styles.locusPlotCard}>
          <div className={styles.locusPlotTitle}>Phase-to-Phase | ZAB, ZBC, ZCA</div>
          <Plot
            data={phaseTraces}
            layout={familyLayout("Phase-to-Phase", phaseViewRange.x, phaseViewRange.y, currentPlayMs, detailMode, phaseZones)}
            config={{ displayModeBar: true, responsive: true, displaylogo: false, toImageButtonOptions: { scale: detailMode === "detailed" ? 3 : 2 } }}
            style={{ width: "100%" }}
            onRelayout={(event) => rememberPlotRange("phase", event as Readonly<Record<string, unknown>>)}
          />
        </div>
      </div>

      {hasRelayZones ? (
        <div className={styles.locusEditorGrid}>
          {groundZones.length > 0 && renderZoneEditor("Ground Zones", "ground", groundZones, groundShape)}
          {phaseZones.length > 0 && renderZoneEditor("Phase-Phase Zones", "phase", phaseZones, phaseShape)}
        </div>
      ) : (
        <div className={styles.warning} style={{ marginTop: 12 }}>
          Belum ada karakteristik zona relay. Upload file RIO/XRIO untuk menampilkan mho atau polygon protection zone.
        </div>
      )}
    </div>
  );
}
