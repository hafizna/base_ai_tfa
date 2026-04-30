import { useEffect, useMemo, useRef, useState } from "react";

import { computeLocus, fetchElectricalParams21 } from "../../../api/client";
import Plot from "../../plot/PlotlyChart";
import styles from "../../panels/Panel.module.css";

interface Zone {
  label: string;
  shape: "mho" | "quad";
  center_r: number;
  center_x: number;
  radius: number;
  rf_fwd: number;
  rf_rev: number;
  xf: number;
  xr: number;
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

function quadVertices(zone: Zone) {
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

  return { kind: "rio", phGnd: phGnd.slice(0, 3), phPh: phPh.slice(0, 3), ...(earthComp ? { earthComp } : {}) };
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

function loopTrace(loop: LoopName, points: LocusPoint[]): Partial<Plotly.ScatterData> {
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
    line: { color: LOOP_COLORS[loop], width: 1.5, shape: "spline", smoothing: 0.8 },
    marker: { color: LOOP_COLORS[loop], size: 5, symbol: "square" },
    connectgaps: false,
    hovertemplate: `${loop}<br>t=%{customdata:.2f} ms<br>R=%{x:.2f} Ω<br>X=%{y:.2f} Ω<extra></extra>`,
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

function familyLayout(title: string, xRange: [number, number], yRange: [number, number], currentMs?: number): Partial<Plotly.Layout> {
  return {
    uirevision: title,
    height: 460,
    margin: { t: 36, b: 56, l: 64, r: 20 },
    autosize: true,
    xaxis: { title: { text: "R (secondary ohm)" }, range: xRange, zeroline: false, tickfont: { size: 10 } },
    yaxis: {
      title: { text: "X (secondary ohm)" },
      range: yRange,
      zeroline: false,
      tickfont: { size: 10 },
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
      { type: "line", x0: xRange[0], x1: xRange[1], y0: 0, y1: 0, line: { color: "#cbd5e1", width: 1 } },
      { type: "line", x0: 0, x1: 0, y0: yRange[0], y1: yRange[1], line: { color: "#cbd5e1", width: 1 } },
    ] as Plotly.Shape[],
    annotations: currentMs !== undefined ? [{
      xref: "paper",
      yref: "paper",
      x: 1,
      y: 1.08,
      xanchor: "right",
      text: `t <= ${currentMs.toFixed(2)} ms`,
      showarrow: false,
      font: { size: 10, color: "#475569" },
      bgcolor: "rgba(255,255,255,0.9)",
      bordercolor: "#cbd5e1",
      borderpad: 3,
    }] : [],
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
  const [timing, setTiming] = useState<{ inceptionMs: number | null; durationMs: number | null; tripMs: number | null }>({
    inceptionMs: null,
    durationMs: null,
    tripMs: null,
  });
  const [playMs, setPlayMs] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [replaySpeed, setReplaySpeed] = useState(0.5);
  const [plotRanges, setPlotRanges] = useState<Record<PlotFamily, PlotRange>>({ ground: {}, phase: {} });
  const [ctRatioOverride, setCtRatioOverride] = useState<number | null>(null);
  const [vtRatioOverride, setVtRatioOverride] = useState<number | null>(null);
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
  }, [dataRevision]);

  useEffect(() => {
    let alive = true;
    fetchElectricalParams21(analysisId)
      .then((params) => {
        if (!alive) return;
        setTiming({
          inceptionMs: typeof params.inception_time_ms === "number" ? params.inception_time_ms : null,
          durationMs: typeof params.fault_duration_ms === "number" ? params.fault_duration_ms : null,
          tripMs: typeof params.trip_time_ms === "number" ? params.trip_time_ms : null,
        });
      })
      .catch(() => {
        if (alive) setTiming({ inceptionMs: null, durationMs: null, tripMs: null });
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

  function updateFamilyShape(family: ZoneFamily, shape: Zone["shape"]) {
    updateZoneSet(family, (zones) => zones.map((zone) => ({ ...zone, shape })));
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
        nextCtRatio = imported.ctRatio;
        nextVtRatio = imported.vtRatio;
        setCtRatioOverride(nextCtRatio);
        setVtRatioOverride(nextVtRatio);
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
  const hasRelayZones = groundZones.length > 0 || phaseZones.length > 0;

  const groundTraces = useMemo(() => {
    const loci = GROUND_LOOPS.filter((loop) => (visiblePointsByLoop[loop] ?? []).length > 0).map((loop) =>
      loopTrace(loop, visiblePointsByLoop[loop] ?? [])
    );
    const showPlayHead = currentPlayMs < activeTimeRange[1] - 0.01;
    const heads = showPlayHead
      ? GROUND_LOOPS.map((loop) => headTrace(loop, visiblePointsByLoop[loop] ?? [], currentPlayMs)).filter(Boolean)
      : [];
    const inception = timing.inceptionMs !== null
      ? eventTrace("Inception marker", GROUND_LOOPS, pointsByLoop, timing.inceptionMs, "#facc15")
      : null;
    const trip = relayTripMs !== null
      ? eventTrace("Relay trip marker", GROUND_LOOPS, pointsByLoop, relayTripMs, "#2563eb")
      : null;
    const zoneTraces = groundZones.map((zone) => (zone.shape === "mho" ? mhoCircleTrace(zone) : quadTrace(zone)));
    return [...loci, ...(heads as Plotly.Data[]), ...(inception ? [inception as Plotly.Data] : []), ...(trip ? [trip as Plotly.Data] : []), ...zoneTraces] as Plotly.Data[];
  }, [activeTimeRange, currentPlayMs, groundZones, pointsByLoop, relayTripMs, timing.inceptionMs, visiblePointsByLoop]);

  const phaseTraces = useMemo(() => {
    const loci = PHASE_LOOPS.filter((loop) => (visiblePointsByLoop[loop] ?? []).length > 0).map((loop) =>
      loopTrace(loop, visiblePointsByLoop[loop] ?? [])
    );
    const showPlayHead = currentPlayMs < activeTimeRange[1] - 0.01;
    const heads = showPlayHead
      ? PHASE_LOOPS.map((loop) => headTrace(loop, visiblePointsByLoop[loop] ?? [], currentPlayMs)).filter(Boolean)
      : [];
    const inception = timing.inceptionMs !== null
      ? eventTrace("Inception marker", PHASE_LOOPS, pointsByLoop, timing.inceptionMs, "#facc15")
      : null;
    const trip = relayTripMs !== null
      ? eventTrace("Relay trip marker", PHASE_LOOPS, pointsByLoop, relayTripMs, "#2563eb")
      : null;
    const zoneTraces = phaseZones.map((zone) => (zone.shape === "mho" ? mhoCircleTrace(zone) : quadTrace(zone)));
    return [...loci, ...(heads as Plotly.Data[]), ...(inception ? [inception as Plotly.Data] : []), ...(trip ? [trip as Plotly.Data] : []), ...zoneTraces] as Plotly.Data[];
  }, [activeTimeRange, currentPlayMs, phaseZones, pointsByLoop, relayTripMs, timing.inceptionMs, visiblePointsByLoop]);

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

  const groundShape = groundZones[0]?.shape ?? "mho";
  const phaseShape = phaseZones[0]?.shape ?? "mho";
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

  function renderZoneEditor(title: string, family: ZoneFamily, zones: Zone[], familyShape: Zone["shape"]) {
    return (
      <div className={styles.locusEditorSection}>
        <div className={styles.locusEditorHeader}>
          <h3 className={styles.locusEditorTitle}>{title}</h3>
          <div className={styles.controls}>
            <label className={styles.label}>Zone Shape</label>
            <select
              className={styles.selectField}
              value={familyShape}
              onChange={(e) => updateFamilyShape(family, e.target.value as Zone["shape"])}
            >
              <option value="mho">Mho</option>
              <option value="quad">Quadrilateral</option>
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
              <div style={{ fontSize: "0.75rem", color: "#64748b", padding: "4px 0" }}>
                Polygon ({zone.poly_r.length - 1} vertices) — dari file .rio
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
          {timing.inceptionMs !== null && timing.durationMs !== null && (
            <span>
              Inception {timing.inceptionMs.toFixed(1)} ms | Relay trip {relayTripMs !== null ? relayTripMs.toFixed(1) : "-"} ms | FCT {timing.durationMs.toFixed(1)} ms
            </span>
          )}
        </div>
      </div>

      <div className={styles.locusPlotStack}>
        <div className={styles.locusPlotCard}>
          <div className={styles.locusPlotTitle}>Phase-to-Ground | ZA, ZB, ZC</div>
          <Plot
            data={groundTraces}
            layout={familyLayout("Phase-to-Ground", groundViewRange.x, groundViewRange.y, currentPlayMs)}
            config={{ displayModeBar: true, responsive: true }}
            style={{ width: "100%" }}
            onRelayout={(event) => rememberPlotRange("ground", event as Readonly<Record<string, unknown>>)}
          />
        </div>

        <div className={styles.locusPlotCard}>
          <div className={styles.locusPlotTitle}>Phase-to-Phase | ZAB, ZBC, ZCA</div>
          <Plot
            data={phaseTraces}
            layout={familyLayout("Phase-to-Phase", phaseViewRange.x, phaseViewRange.y, currentPlayMs)}
            config={{ displayModeBar: true, responsive: true }}
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
