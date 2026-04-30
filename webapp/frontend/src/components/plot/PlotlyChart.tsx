import type { ComponentType } from "react";
import PlotlyModule, { type PlotParams } from "react-plotly.js";

type PlotModuleShape = ComponentType<PlotParams> | { default?: ComponentType<PlotParams> };

const PlotlyComponent =
  (PlotlyModule as PlotModuleShape & { default?: ComponentType<PlotParams> }).default ??
  (PlotlyModule as ComponentType<PlotParams>);

export default function PlotlyChart(props: PlotParams) {
  return <PlotlyComponent {...props} />;
}
