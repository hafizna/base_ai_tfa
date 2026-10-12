import Plotly from "plotly.js/lib/core";
import histogram from "plotly.js/lib/histogram";
import scattergl from "plotly.js/lib/scattergl";

// Core already includes scatter. Keep histogram for DE-FL and WebGL for TWS.
Plotly.register([histogram, scattergl]);

// Workspace, DE-FL and TWS use this API to capture charts for PDF reports.
Object.assign(window, { Plotly });

export default Plotly;
