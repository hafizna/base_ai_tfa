import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { useAnalysis } from "../context/AnalysisContext";

// Keep route selection and the real provider while avoiding API/Plotly work.
vi.mock("../pages/Upload", () => ({
  default: function UploadProbe() {
    const { relayType } = useAnalysis();
    return <div>Upload selected: {relayType}</div>;
  },
}));
vi.mock("../pages/IncidentWorkspace", () => ({
  default: () => <div>Incident workspace loaded</div>,
}));

describe("Application route loading", () => {
  beforeEach(() => window.history.replaceState({}, "", "/"));

  it("preserves the selected relay when navigating to a lazy upload page", async () => {
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /21 \/ 87L - Line Protection/ }));
    expect(await screen.findByText("Upload selected: LINE")).toBeInTheDocument();
    expect(window.location.pathname).toBe("/upload");
  });

  it("loads a lazy workspace on a direct link", async () => {
    window.history.replaceState({}, "", "/incidents/example");
    render(<App />);
    expect(await screen.findByText("Incident workspace loaded")).toBeInTheDocument();
  });

  it("redirects an unknown URL to the landing page", async () => {
    window.history.replaceState({}, "", "/unknown-route");
    render(<App />);
    expect(await screen.findByText("DFR Analyser")).toBeInTheDocument();
    expect(window.location.pathname).toBe("/");
  });
});
