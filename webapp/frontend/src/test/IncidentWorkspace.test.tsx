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

const includedIds = bringin.incident.records.filter((r) => r.inclusion_status === "INCLUDED").map((r) => r.analysis_id);

function faultSection(analysisIds: string[] = includedIds): client.FaultSectionOut {
  const interval = (id: string, label: string): client.FaultSectionInterval => ({
    id,
    start_s: 0,
    end_s: 0.05,
    start_clock: "11:34:12.000",
    end_clock: "11:34:12.050",
    open: [],
    label,
    after_clearing: false,
  });
  return {
    incident_id: bringin.incident.incident_id,
    computed_at: "2026-10-13T08:00:00+00:00",
    record_count: analysisIds.length,
    analysis_ids: analysisIds,
    title: "Uji",
    template: "diameter",
    epoch: "2025-05-26T11:34:12",
    recorders: [{ id: "r1", label: "CCP 7B2", offset_s: 0, alignment: "acuan" }],
    mapping: [
      { subject: "CT CB 7B2", rule: "KCL", evidence: "jumlah arus CCP ≈ 0", certain: true },
      { subject: "CT CB 7A2", rule: "sisi CT", evidence: "sisi CT tidak terbaca", certain: false },
    ],
    mapping_choice: "",
    limits_a: {},
    notes: [],
    warnings: [],
    report_lines: [],
    episodes: [
      {
        start_s: 0,
        end_s: 0.08,
        start_clock: "11:34:12.000",
        end_clock: "11:34:12.080",
        inferred_assumptions: [],
        intervals: [interval("S0", "awal: semua CB tertutup"), interval("S1", "+ CB 7B2 open")],
        summary: {
          observations: 12,
          strong_observations: 9,
          hypotheses: 40,
          all_strong_consistent: true,
          consistent: [
            { section: "stub_cb_b", label: "Stub CB 7B2 sisi Bus B", agreements: 9, weak_mismatches: 0, variants: 2, weak_texts: [] },
          ],
          notes: [],
          inferred: ["CT CB 7B2 di sisi Bus B"],
          stuck_only: [],
          rejected: [
            { section: "bus_b", label: "Bus B", reason: "S0: CT CB 7B2 diprediksi tanpa arus, teramati membawa arus", weak_only: false },
          ],
        },
      },
    ],
  };
}

function mockApi({
  incident = bringin.incident,
  reconstruction = bringin.reconstruction as client.ReconstructionOut | null,
  enabled = true,
  storedFaultSection = null as client.FaultSectionOut | null,
} = {}) {
  vi.spyOn(client, "fetchHealth").mockResolvedValue(health(enabled));
  vi.spyOn(client, "fetchFaultSection").mockResolvedValue(storedFaultSection);
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

  it("computes where in the bay the fault was when asked", async () => {
    mockApi();
    const run = vi.spyOn(client, "runFaultSection").mockResolvedValue(faultSection());

    renderWorkspace(bringin.incident.incident_id);

    expect(await screen.findByRole("heading", { name: "Lokasi gangguan" })).toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: "Hitung lokasi gangguan" }));
    await waitFor(() => expect(run).toHaveBeenCalledWith(bringin.incident.incident_id));
    expect(await screen.findByText("Stub CB 7B2 sisi Bus B")).toBeInTheDocument();
    expect(screen.getByText("Lokasi yang konsisten dengan semua bukti kuat:")).toBeInTheDocument();
    expect(screen.getByText("CT CB 7B2 di sisi Bus B")).toBeInTheDocument();
    expect(screen.getByText("Lokasi lain yang ditolak (1)")).toBeInTheDocument();
    expect(screen.getByText("belum pasti, diuji sebagai alternatif")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hitung ulang" })).toBeInTheDocument();
  });

  it("shows the stored fault location and asks for a new run once the records changed", async () => {
    mockApi({ storedFaultSection: faultSection(["an-old"]) });

    renderWorkspace(bringin.incident.incident_id);

    expect(await screen.findByText("Stub CB 7B2 sisi Bus B")).toBeInTheDocument();
    expect(screen.getByText(/Rekaman insiden berubah sejak lokasi ini dihitung/)).toBeInTheDocument();
  });

  it("does not flag the stored fault location when it covers the current records", async () => {
    mockApi({ storedFaultSection: faultSection() });

    renderWorkspace(bringin.incident.incident_id);

    expect(await screen.findByText("Stub CB 7B2 sisi Bus B")).toBeInTheDocument();
    expect(screen.queryByText(/Rekaman insiden berubah/)).not.toBeInTheDocument();
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
