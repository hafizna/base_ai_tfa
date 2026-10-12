import type { ComponentType } from "react";
import type { PlotParams } from "react-plotly.js";
import createPlotlyModule from "react-plotly.js/factory";
import Plotly from "./plotlyRuntime";

type PlotFactory = (plotly: object) => ComponentType<PlotParams>;

const createPlotlyComponent =
  (createPlotlyModule as PlotFactory & { default?: PlotFactory }).default ?? createPlotlyModule;
const PlotlyComponent = createPlotlyComponent(Plotly);

export default function PlotlyChart(props: PlotParams) {
  return <PlotlyComponent {...props} />;
}
