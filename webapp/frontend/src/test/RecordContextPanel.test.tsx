import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import RecordContextPanel from "../components/panels/RecordContextPanel";
import { fetchCanonicalAnalysis, type CanonicalRecordAnalysis } from "../api/client";

vi.mock("../api/client", () => ({ fetchCanonicalAnalysis: vi.fn() }));

describe("single-record context", () => {
  it("distinguishes closure from restoration and shows both fault episodes", async () => {
    vi.mocked(fetchCanonicalAnalysis).mockResolvedValue({
      event_window: { sequence: { mechanical_close_confirmed: true, restoration_outcome: "failed",
        refault_after_reclose: true, sotf_after_reclose: true } },
      fault_episodes: [
        { inception_time_ms: 495, clearing_time_ms: 563, fault_duration_ms: 68 },
        { inception_time_ms: 1585, clearing_time_ms: 1662, fault_duration_ms: 77, after_reclose: true },
      ],
      reasoning: { conclusions: [{ key: "trip_reclose", label: "Trip dan reclose", title: "SOTF/TOR trip",
        evidence: ["SOTF aktif sesudah PMT menutup kembali."], conflicts: [] }] },
    } as unknown as CanonicalRecordAnalysis);
    const { rerender } = render(<RecordContextPanel analysisId="rawalo" />);
    expect(await screen.findByText("2 episode gangguan dalam satu rekaman.")).toBeInTheDocument();
    expect(screen.getByText(/Pemulihan gagal/)).toBeInTheDocument();
    expect(screen.getByText("SOTF/TOR trip setelah penutupan kembali.")).toBeInTheDocument();
    expect(screen.getByText("2 — setelah reclose")).toBeInTheDocument();
    rerender(<RecordContextPanel analysisId="rawalo" dataRevision={1} />);
    await waitFor(() => expect(fetchCanonicalAnalysis).toHaveBeenCalledWith("rawalo", 1));
  });
});
