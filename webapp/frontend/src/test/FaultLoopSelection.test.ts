import { describe, expect, it } from "vitest";
import { eventLoopsForFamily } from "../components/relay/relay21/faultLoopSelection";

describe("fault event marker scope", () => {
  it("marks only ZBC for a B+C line-to-line fault", () => {
    const fault = { phases: ["B", "C"], to_ground: false };
    expect(eventLoopsForFamily("phase", fault)).toEqual(["ZBC"]);
    expect(eventLoopsForFamily("ground", fault)).toEqual([]);
  });

  it("marks ground phases without inventing phase-to-phase markers", () => {
    const fault = { phases: ["C", "A"], to_ground: true };
    expect(eventLoopsForFamily("ground", fault)).toEqual(["ZA", "ZC"]);
    expect(eventLoopsForFamily("phase", fault)).toEqual([]);
  });

  it("handles three-phase faults and suppresses unknown or absent faults", () => {
    expect(eventLoopsForFamily("phase", { phases: ["A", "B", "C"], to_ground: false }))
      .toEqual(["ZAB", "ZBC", "ZCA"]);
    for (const fault of [null, { phases: [], to_ground: false }, { phases: ["A"], to_ground: true, no_fault: true }]) {
      expect(eventLoopsForFamily("phase", fault)).toEqual([]);
      expect(eventLoopsForFamily("ground", fault)).toEqual([]);
    }
  });
});
