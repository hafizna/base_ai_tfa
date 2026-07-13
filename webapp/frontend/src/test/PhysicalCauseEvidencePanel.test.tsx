import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import PhysicalCauseEvidencePanel from "../components/incidents/PhysicalCauseEvidencePanel";
import { loadReconstructionFixture } from "./fixtures/loadReconstructionFixture";

describe("PhysicalCauseEvidencePanel", () => {
  it("renders one row per record with its own confidence, never an averaged value", () => {
    const { reconstruction, episodes } = loadReconstructionFixture();
    const evidence = reconstruction.physical_cause_evidence;

    render(<PhysicalCauseEvidencePanel physicalCauseEvidence={evidence} episodes={episodes} recordLabel={(id) => id} />);

    // Every per-record confidence value from the fixture must appear as its
    // own distinct number in the table — nothing should be combined into a
    // single incident-wide probability.
    evidence.records.forEach((r) => {
      if (r.confidence != null) {
        expect(screen.getAllByText(`${Math.round(r.confidence * 100)}%`).length).toBeGreaterThan(0);
      }
    });

    expect(screen.getByText(evidence.consistency)).toBeInTheDocument();
    expect(screen.getByText(/Incident root cause: UNCONFIRMED/)).toBeInTheDocument();
  });

  it("does not render any incident-level averaged-probability field", () => {
    const { reconstruction, episodes } = loadReconstructionFixture();
    render(
      <PhysicalCauseEvidencePanel
        physicalCauseEvidence={reconstruction.physical_cause_evidence}
        episodes={episodes}
        recordLabel={(id) => id}
      />,
    );
    expect(screen.queryByText(/incident_probability/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/average confidence/i)).not.toBeInTheDocument();
  });

  it("shows the RECORD_LOCAL_SIGNATURES scope tag", () => {
    const { reconstruction, episodes } = loadReconstructionFixture();
    render(
      <PhysicalCauseEvidencePanel
        physicalCauseEvidence={reconstruction.physical_cause_evidence}
        episodes={episodes}
        recordLabel={(id) => id}
      />,
    );
    expect(screen.getByText("RECORD_LOCAL_SIGNATURES")).toBeInTheDocument();
  });
});
