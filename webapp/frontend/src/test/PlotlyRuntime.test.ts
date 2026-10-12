import { afterEach, describe, expect, it } from "vitest";
import Plotly from "../components/plot/plotlyRuntime";

const plots: HTMLElement[] = [];
afterEach(() => {
  for (const plot of plots.splice(0)) {
    Plotly.purge(plot);
    plot.remove();
  }
});

describe("Custom Plotly runtime", () => {
  it("retains the three application trace types and the report export API", () => {
    const schema = (Plotly as typeof Plotly & {
      PlotSchema: { get(): { traces: Record<string, unknown> } };
    }).PlotSchema.get();
    expect(Object.keys(schema.traces).sort()).toEqual(["histogram", "scatter", "scattergl"]);
    expect((window as unknown as { Plotly: typeof Plotly }).Plotly.toImage).toBe(Plotly.toImage);
    expect(Plotly.toImage).toBeTypeOf("function");
  });

  it("renders waveform lines, digital fills, histogram bins and locus overlays", async () => {
    const plot = document.createElement("div");
    plots.push(plot);
    document.body.append(plot);
    await Plotly.newPlot(plot, [
      { type: "scatter", x: [0, 1, 2], y: [0, 2, 0], mode: "lines", name: "waveform" },
      { type: "scatter", x: [0, 0, 1, 1, 0], y: [0, 1, 1, 0, 0], fill: "toself", name: "digital" },
      { type: "histogram", x: [0.1, 0.2, 1.1, 1.2], xbins: { start: 0, end: 2, size: 1 } },
    ], {
      width: 600, height: 400,
      shapes: [{ type: "circle", x0: 0, y0: 0, x1: 1, y1: 1 }],
      annotations: [{ x: 1, y: 2, text: "Trip", showarrow: false }],
    }, { staticPlot: true });
    expect(plot.querySelector(".scatterlayer .js-line")).not.toBeNull();
    expect(plot.querySelector(".scatterlayer .js-fill")).not.toBeNull();
    expect(plot.querySelectorAll(".barlayer .point")).toHaveLength(2);
    expect(plot.querySelector(".shapelayer path")).not.toBeNull();
    expect(plot.querySelector(".annotation-text")?.textContent).toBe("Trip");
    await Plotly.relayout(plot, { "xaxis.range": [0, 1] });
    expect((plot as unknown as { layout: Plotly.Layout }).layout.xaxis.range).toEqual([0, 1]);
    const svg = await Plotly.toImage(plot, { format: "svg", width: 600, height: 400 });
    expect(svg).toMatch(/^data:image\/svg\+xml/);
    expect(decodeURIComponent(svg)).toContain("Trip");
  });
});
