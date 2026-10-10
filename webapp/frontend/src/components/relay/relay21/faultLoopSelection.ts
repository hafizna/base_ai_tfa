export type LoopName = "ZA" | "ZB" | "ZC" | "ZAB" | "ZBC" | "ZCA";

interface FaultPhases {
  phases: string[];
  to_ground: boolean;
  no_fault?: boolean;
}

/** Scope event markers to identified fault phases; locus lines stay visible. */
export function eventLoopsForFamily(family: "ground" | "phase", classification: FaultPhases | null): LoopName[] {
  if (!classification || classification.no_fault) return [];
  const phases = [...new Set(classification.phases.map((phase) => phase.trim().toUpperCase()))]
    .filter((phase) => ["A", "B", "C"].includes(phase)).sort();
  if (classification.to_ground) {
    return family === "ground" ? phases.map((phase) => `Z${phase}` as LoopName) : [];
  }
  if (family !== "phase") return [];
  if (phases.length === 3) return ["ZAB", "ZBC", "ZCA"];
  const loopByPair: Record<string, LoopName> = { AB: "ZAB", BC: "ZBC", AC: "ZCA" };
  const loop = loopByPair[phases.join("")];
  return loop ? [loop] : [];
}
