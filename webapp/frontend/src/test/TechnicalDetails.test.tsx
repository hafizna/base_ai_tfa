import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import type { IncidentOut, ReconstructionOut } from "../api/client";
import { buildIncidentStory } from "../components/incidents/incidentStory";
import TechnicalDetails from "../components/incidents/TechnicalDetails";
import twoEndedRaw from "./fixtures/incident_two_ended.json";

// Bringin ZQ6D/E/F with the Mojosongo end's Qualitrol files, reconstructed by
// the current engine: every fault carries its reasoning ledger.
type Fixture = { incident: IncidentOut; reconstruction: ReconstructionOut };
const twoEnded = twoEndedRaw as unknown as Fixture;

function renderDetails(overrides: Partial<Parameters<typeof TechnicalDetails>[0]> = {}) {
  const { incident, reconstruction } = twoEnded;
  const props = {
    incident,
    reconstruction,
    episodes: reconstruction.episodes ?? [],
    relationships: reconstruction.relationships ?? [],
    story: buildIncidentStory(incident, reconstruction),
    refreshing: false,
    onHide: vi.fn(),
    onAddRecords: vi.fn(),
    onDetach: vi.fn(),
    onSave: vi.fn().mockResolvedValue(undefined),
    onArchive: vi.fn(),
    onRefreshAnalyses: vi.fn(),
    onRelationshipChanged: vi.fn(),
    ...overrides,
  };
  render(
    <MemoryRouter>
      <TechnicalDetails {...props} />
    </MemoryRouter>,
  );
  return props;
}

describe("TechnicalDetails — Penalaran", () => {
  it("shows each fault's ledger with its rules and confidence", () => {
    renderDetails();

    expect(screen.getByRole("button", { name: "Gangguan #1 · 15:15:03,738" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("S-T, antar fasa, tidak ke tanah")).toBeInTheDocument();
    expect(screen.getByText("77 ms setelah gangguan, di bawah batas 120 ms")).toBeInTheDocument();
    expect(screen.getByText("Trip 3-pole, reclose berhasil setelah 5,0 s (rekaman ZQ6E)")).toBeInTheDocument();
    expect(screen.getByText("GI MOJOSONGO: fasa S-T, padam 59 ms")).toBeInTheDocument();
    expect(screen.getByText(/pola echo weak infeed pada skema POTT/)).toBeInTheDocument();
    // The far end received while this end's recorded Send channel never asserted (P4).
    expect(screen.getByText("GI lawan menerima sinyal, kanal Send di GI ini tidak aktif")).toBeInTheDocument();
    expect(screen.getByText("1 ditandai")).toBeInTheDocument();
    expect(screen.getAllByText("F3.5").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Tinggi").length).toBeGreaterThan(0);
    expect(screen.getByText("Bacaan AI")).toBeInTheDocument();
    // The far end is attached, so there is nothing to ask for.
    expect(screen.queryByText("Skema dan lokasi baru terkonfirmasi dari sisi lawan")).not.toBeInTheDocument();
  });

  it("switches to the re-fault", () => {
    renderDetails();

    fireEvent.click(screen.getByRole("button", { name: "Gangguan #2 · 15:15:14,564" }));
    expect(screen.getByText("Tidak ada penyebab yang dominan (bacaan AI)")).toBeInTheDocument();
    expect(screen.getByText("Trip 3-pole; tidak ada reclose yang terekam di insiden ini")).toBeInTheDocument();
  });
});

describe("TechnicalDetails — Urutan sinyal", () => {
  it("lists the fault record's status changes and the channels that never changed", () => {
    renderDetails();
    fireEvent.click(screen.getByRole("button", { name: "Urutan sinyal" }));

    expect(screen.getByRole("button", { name: "ZQ6D" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText(/Waktu dihitung dari awal gangguan \(15:15:03,738 jam DFR GI BRINGIN\)/)).toBeInTheDocument();
    expect(screen.getAllByText("TRIP Z1 MJSNG2").length).toBeGreaterThan(0);
    expect(screen.getByText("Direkam, tidak pernah aktif")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^Semua kanal/ }));
    expect(screen.getAllByText("Tidak pernah aktif").length).toBeGreaterThan(0);
  });

  it("reads the far end's own channels", () => {
    renderDetails();
    fireEvent.click(screen.getByRole("button", { name: "Urutan sinyal" }));
    fireEvent.click(screen.getByRole("button", { name: "230821,081503670" }));

    expect(screen.getAllByText("DIST RECEIVE BRINGIN 2").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Aktif: sinyal dikirim").length).toBeGreaterThan(0);
  });
});

describe("TechnicalDetails — Rekaman and Data insiden", () => {
  it("lists the records and how they relate, each relation correctable", () => {
    const props = renderDetails();
    fireEvent.click(screen.getByRole("button", { name: "Rekaman" }));

    expect(screen.getAllByText("Other line end").length).toBe(3);
    expect(screen.getAllByText("Reclose sequence").length).toBe(2);
    const zq6d = screen.getAllByText("ZQ6D")[0].closest("div")?.parentElement as HTMLElement;
    fireEvent.click(within(zq6d).getByRole("button", { name: "Lepas" }));
    expect(props.onDetach).toHaveBeenCalledWith(twoEnded.incident.records.find((r) => r.source_filename === "ZQ6D.cfg")?.incident_record_id);
  });

  it("saves the incident's own data", async () => {
    const props = renderDetails();
    fireEvent.click(screen.getByRole("button", { name: "Data insiden" }));

    fireEvent.change(screen.getByLabelText("Bay"), { target: { value: "MJSNG2" } });
    fireEvent.change(screen.getByLabelText("Tegangan (kV)"), { target: { value: "150" } });
    fireEvent.click(screen.getByRole("button", { name: "Simpan" }));
    expect(props.onSave).toHaveBeenCalledWith(expect.objectContaining({ bay_name: "MJSNG2", voltage_level_kv: 150 }));
  });
});
