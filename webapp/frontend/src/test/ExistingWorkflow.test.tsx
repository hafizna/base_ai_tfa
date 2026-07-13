import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { useEffect } from "react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { AnalysisProvider, useAnalysis } from "../context/AnalysisContext";
import Upload from "../pages/Upload";
import Workspace from "../pages/Workspace";
import * as client from "../api/client";

/** Seeds AnalysisContext.relayType so Upload doesn't immediately redirect to "/". */
function SeedRelayType({ children }: { children: React.ReactNode }) {
  const { setRelayType } = useAnalysis();
  useEffect(() => {
    setRelayType("21");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return <>{children}</>;
}

describe("Existing single-record workflow is unchanged by Stage 2", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("Upload page still renders its file-selection form for a chosen relay type", async () => {
    render(
      <MemoryRouter initialEntries={["/upload"]}>
        <AnalysisProvider>
          <SeedRelayType>
            <Upload />
          </SeedRelayType>
        </AnalysisProvider>
      </MemoryRouter>,
    );

    await waitFor(() => expect(screen.getByText(/21 - Distance/)).toBeInTheDocument());
  });

  it("Workspace page still renders for a single-record analysis_id without any Stage 2 calls blocking it", async () => {
    vi.spyOn(client, "fetchAnalysis").mockResolvedValue({
      station_name: "GI TEST",
      rec_dev_id: "RELAY1",
      rev_year: "2013",
      sampling_rates: [[1200, 100]],
      trigger_time: 0,
      total_samples: 100,
      frequency: 50,
      time: Array.from({ length: 100 }, (_, i) => i / 1200),
      analog_channels: [],
      status_channels: [],
      warnings: [],
    });

    render(
      <MemoryRouter initialEntries={["/workspace/21/abc123"]}>
        <AnalysisProvider>
          <Routes>
            <Route path="/workspace/:relayType/:analysisId" element={<Workspace />} />
          </Routes>
        </AnalysisProvider>
      </MemoryRouter>,
    );

    await waitFor(() => expect(screen.getAllByText((_, el) => el?.textContent === "GI TEST").length).toBeGreaterThan(0));
  });
});
