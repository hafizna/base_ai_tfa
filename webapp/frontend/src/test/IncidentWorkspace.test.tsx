import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import IncidentWorkspace from "../pages/IncidentWorkspace";
import * as client from "../api/client";
import bringinRaw from "./fixtures/incident_bringin.json";

type Fixture = { incident: client.IncidentOut; reconstruction: client.ReconstructionOut };
const bringin = bringinRaw as unknown as Fixture;

function health(enabled: boolean): client.HealthResponse {
  return {
    status: "ok",
    version: "2.0.0",
    analysis_storage: "filesystem",
    analysis_ttl_hours: 24,
    warmup: {},
    feature_flags: { multi_comtrade_enabled: enabled },
  };
}

function renderWorkspace(incidentId: string) {
  return render(
    <MemoryRouter initialEntries={[`/incidents/${incidentId}`]}>
      <Routes>
        <Route path="/incidents/:incidentId" element={<IncidentWorkspace />} />
      </Routes>
    </MemoryRouter>,
  );
}

function mockApi({
  incident = bringin.incident,
  reconstruction = bringin.reconstruction as client.ReconstructionOut | null,
  enabled = true,
} = {}) {
  vi.spyOn(client, "fetchHealth").mockResolvedValue(health(enabled));
  vi.spyOn(client, "fetchIncident").mockResolvedValue(incident);
  vi.spyOn(client, "listIncidentEvidence").mockResolvedValue([]);
  vi.spyOn(client, "listReconstructions").mockResolvedValue(reconstruction ? [reconstruction] : []);
  if (reconstruction) vi.spyOn(client, "fetchReconstruction").mockResolvedValue(reconstruction);
  else vi.spyOn(client, "fetchReconstruction").mockRejectedValue({ response: { status: 404 } });
  return vi.spyOn(client, "reconstructIncident").mockResolvedValue(bringin.reconstruction);
}

describe("IncidentWorkspace", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });

  it("tells the story of the latest reconstruction without rebuilding it", async () => {
    const reconstruct = mockApi();

    renderWorkspace(bringin.incident.incident_id);

    await waitFor(() => expect(screen.getByRole("heading", { name: "Gangguan berulang setelah reclose" })).toBeInTheDocument());
    expect(screen.getByRole("heading", { name: "Urutan kejadian" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Indikasi kontak fisik (pohon / benda asing)" })).toBeInTheDocument();
    expect(screen.getByText("Trip 3-pole +45 ms · Z1")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Rekaman (3)" })).toBeInTheDocument();
    expect(reconstruct).not.toHaveBeenCalled();
  });

  it("reconstructs automatically when the incident has records but no reconstruction yet", async () => {
    const reconstruct = mockApi({ reconstruction: null });

    renderWorkspace(bringin.incident.incident_id);

    await waitFor(() => expect(reconstruct).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByRole("heading", { name: "Gangguan berulang setelah reclose" })).toBeInTheDocument());
  });

  it("reconstructs again when records were attached after the last reconstruction", async () => {
    const extra = { ...bringin.incident.records[0], incident_record_id: "ir-new", analysis_id: "an-new" };
    const reconstruct = mockApi({ incident: { ...bringin.incident, records: [...bringin.incident.records, extra] } });

    renderWorkspace(bringin.incident.incident_id);

    await waitFor(() => expect(reconstruct).toHaveBeenCalledTimes(1));
  });

  it("asks for files when the incident has no records", async () => {
    const reconstruct = mockApi({ incident: { ...bringin.incident, records: [] }, reconstruction: null });

    renderWorkspace(bringin.incident.incident_id);

    await waitFor(() => expect(screen.getByRole("heading", { name: "Belum ada rekaman" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Pilih file" }));
    expect(screen.getByRole("dialog", { name: "Tambah rekaman" })).toBeInTheDocument();
    expect(reconstruct).not.toHaveBeenCalled();
  });

  it("keeps the technical details in four tabs, collapsed until asked for", async () => {
    mockApi();

    renderWorkspace(bringin.incident.incident_id);

    const toggle = await screen.findByRole("button", { name: /^Detail teknis/ });
    expect(screen.queryByRole("button", { name: "Penalaran" })).not.toBeInTheDocument();
    fireEvent.click(toggle);
    for (const tab of ["Penalaran", "Urutan sinyal", "Rekaman", "Data insiden"]) {
      expect(screen.getByRole("button", { name: tab })).toBeInTheDocument();
    }
    // Records attached before the reasoning chain existed offer to be analysed again.
    expect(await screen.findByText("Penalaran belum tersedia untuk gangguan ini")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Muat ulang analisa rekaman" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Sembunyikan" }));
    expect(screen.queryByRole("button", { name: "Penalaran" })).not.toBeInTheDocument();
  });

  it("analyses the records again and rebuilds when asked to", async () => {
    const reconstruct = mockApi();
    const refresh = vi.spyOn(client, "refreshIncidentSnapshots").mockResolvedValue(bringin.incident.records);

    renderWorkspace(bringin.incident.incident_id);

    fireEvent.click(await screen.findByRole("button", { name: /^Detail teknis/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Muat ulang analisa rekaman" }));
    await waitFor(() => expect(refresh).toHaveBeenCalledWith(bringin.incident.incident_id));
    await waitFor(() => expect(reconstruct).toHaveBeenCalledTimes(1));
  });

  it("hides the story and uploads when the multi-COMTRADE feature is disabled", async () => {
    vi.spyOn(client, "fetchHealth").mockResolvedValue(health(false));
    vi.spyOn(client, "fetchIncident").mockResolvedValue(bringin.incident);
    vi.spyOn(client, "listIncidentEvidence").mockResolvedValue([]);
    vi.spyOn(client, "fetchReconstruction").mockRejectedValue({ response: { status: 403 } });
    vi.spyOn(client, "listReconstructions").mockResolvedValue([]);

    renderWorkspace(bringin.incident.incident_id);

    // The flag arrives asynchronously (GET /api/health), so wait for the final state.
    await waitFor(() => expect(screen.getByText(/Rekonstruksi multi-COMTRADE nonaktif/)).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Tambah rekaman" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^Detail teknis/ }));
    fireEvent.click(screen.getByRole("button", { name: "Rekaman" }));
    expect(screen.getByText("ZQ6D")).toBeInTheDocument();
    expect(screen.getByText(/Belum ada\. Hubungan/)).toBeInTheDocument();
  });
});
