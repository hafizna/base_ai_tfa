import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi, beforeEach } from "vitest";
import IncidentWorkspace from "../pages/IncidentWorkspace";
import * as client from "../api/client";
import { loadReconstructionFixture } from "./fixtures/loadReconstructionFixture";

function renderWorkspace(incidentId = "inc-1") {
  return render(
    <MemoryRouter initialEntries={[`/incidents/${incidentId}`]}>
      <Routes>
        <Route path="/incidents/:incidentId" element={<IncidentWorkspace />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("IncidentWorkspace", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows 'has not been reconstructed yet' when the incident has no reconstruction", async () => {
    const { incident } = loadReconstructionFixture();
    vi.spyOn(client, "fetchIncident").mockResolvedValue(incident);
    vi.spyOn(client, "fetchHealth").mockResolvedValue({
      status: "ok",
      version: "2.0.0",
      analysis_storage: "filesystem",
      analysis_ttl_hours: 24,
      warmup: {},
      feature_flags: { multi_comtrade_enabled: true },
    });
    vi.spyOn(client, "fetchReconstruction").mockRejectedValue({ response: { status: 404 } });
    vi.spyOn(client, "listReconstructions").mockResolvedValue([]);
    vi.spyOn(client, "listIncidentEvidence").mockResolvedValue([]);

    renderWorkspace(incident.incident_id);

    await waitFor(() => expect(screen.getByText(/has not been reconstructed yet/)).toBeInTheDocument());
  });

  it("renders the reconstruction summary once a reconstruction is available", async () => {
    const { incident, reconstruction } = loadReconstructionFixture();
    vi.spyOn(client, "fetchIncident").mockResolvedValue(incident);
    vi.spyOn(client, "fetchHealth").mockResolvedValue({
      status: "ok",
      version: "2.0.0",
      analysis_storage: "filesystem",
      analysis_ttl_hours: 24,
      warmup: {},
      feature_flags: { multi_comtrade_enabled: true },
    });
    vi.spyOn(client, "fetchReconstruction").mockResolvedValue(reconstruction);
    vi.spyOn(client, "listReconstructions").mockResolvedValue([reconstruction]);
    vi.spyOn(client, "listIncidentEvidence").mockResolvedValue([]);

    renderWorkspace(incident.incident_id);

    await waitFor(() => expect(screen.getByText("ORDER ONLY")).toBeInTheDocument());
  });

  it("hides Stage 2 sections entirely when the feature flag is disabled", async () => {
    const { incident } = loadReconstructionFixture();
    vi.spyOn(client, "fetchIncident").mockResolvedValue(incident);
    vi.spyOn(client, "fetchHealth").mockResolvedValue({
      status: "ok",
      version: "2.0.0",
      analysis_storage: "filesystem",
      analysis_ttl_hours: 24,
      warmup: {},
      feature_flags: { multi_comtrade_enabled: false },
    });
    vi.spyOn(client, "fetchReconstruction").mockRejectedValue({ response: { status: 403 } });
    vi.spyOn(client, "listIncidentEvidence").mockResolvedValue([]);

    renderWorkspace(incident.incident_id);

    // The feature flag arrives asynchronously (from GET /api/health), so the
    // UI settles to its final state — Stage 2 sections must end up hidden
    // regardless of the brief optimistic-default window before the flag
    // check resolves.
    await waitFor(() => expect(screen.getByText("Attached records")).toBeInTheDocument());
    await waitFor(() => expect(screen.queryByText("Reconstruction")).not.toBeInTheDocument());
    expect(screen.queryByText("Batch upload")).not.toBeInTheDocument();
  });
});
