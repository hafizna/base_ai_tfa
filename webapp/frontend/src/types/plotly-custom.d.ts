declare module "plotly.js/lib/core" {
  const Plotly: typeof import("plotly.js");
  export default Plotly;
}

declare module "plotly.js/lib/histogram" {
  const trace: import("plotly.js").TraceModule;
  export default trace;
}

declare module "plotly.js/lib/scattergl" {
  const trace: import("plotly.js").TraceModule;
  export default trace;
}
