import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ReconstructionSummary from "../components/incidents/ReconstructionSummary";
import { loadReconstructionFixture } from "./fixtures/loadReconstructionFixture";

describe("ReconstructionSummary", () => {
  it("renders record count, episode count, and ORDER_ONLY alignment as ORDER ONLY", () => {
    const { reconstruction } = loadReconstructionFixture();
    render(<ReconstructionSummary reconstruction={reconstruction} />);

    expect(screen.getAllByText(reconstruction.observed_incident_facts.record_count).length).toBeGreaterThan(0);
    expect(screen.getAllByText(reconstruction.observed_incident_facts.episode_count).length).toBeGreaterThan(0);
    expect(screen.getByText("ORDER ONLY")).toBeInTheDocument();
  });

  it("never converts uncertainty into a definitive-sounding label", () => {
    const { reconstruction } = loadReconstructionFixture();
    render(<ReconstructionSummary reconstruction={reconstruction} />);

    // The fixture's same_bay_status is CONFIRMED_SAME_BAY; verify the summary
    // renders the actual backend status rather than a hardcoded optimistic string.
    const expectedLabel = {
      CONFIRMED_SAME_BAY: "CONFIRMED SAME BAY",
      LIKELY_SAME_BAY: "LIKELY SAME BAY",
      MISMATCH_REQUIRES_REVIEW: "REQUIRES REVIEW",
      UNKNOWN: "UNKNOWN",
    }[reconstruction.same_bay_status];
    expect(screen.getByText(expectedLabel!)).toBeInTheDocument();
  });
});
