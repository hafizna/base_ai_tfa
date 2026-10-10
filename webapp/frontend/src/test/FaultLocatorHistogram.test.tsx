import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import FaultLocatorHistogram from "../components/relay/relay21de/FaultLocatorHistogram";

const { plot } = vi.hoisted(() => ({ plot: vi.fn() }));
vi.mock("../components/plot/PlotlyChart", () => ({
  default: (props: unknown) => { plot(props); return null; },
}));

function single(terminal: "A" | "B", distance: number) {
  return { terminal, distance_km: distance, distance_pct: distance * 5,
    fault_current_a: 500, r_measured_ohm: 1, x_measured_ohm: 2, warnings: [] };
}

describe("DE-FL histogram reference distances", () => {
  it("converts B to the A axis and keeps true out-of-line readings visible", () => {
    render(<FaultLocatorHistogram histogram={[17, 17.5]} lineLenKm={20}
      twoEnded={{ distanceKm: 17.4, faultCurrentA: 3000, loop: "ZA" }}
      singleEndedA={single("A", 24)} singleEndedB={single("B", 2.6)} />);
    const layout = plot.mock.lastCall![0].layout;
    expect(layout.shapes.map((shape: { x0: number }) => shape.x0)).toEqual([17.4, 24, 17.4]);
    expect(layout.xaxis.title.text).toBe("Distance from terminal A (km)");
    expect(layout.xaxis.range[1]).toBeGreaterThan(24);
    expect(layout.annotations[2].text).toContain("2.60 km from B");
    expect(layout.annotations[2].text).toContain("17.40 km from A");

    fireEvent.click(screen.getByLabelText("Single-ended (A)"));
    expect(plot.mock.lastCall![0].layout.shapes).toHaveLength(2);
  });
});
