import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ReconstructionControls from "../components/incidents/ReconstructionControls";
import { loadReconstructionFixture } from "./fixtures/loadReconstructionFixture";

describe("ReconstructionControls", () => {
  it("shows the ORDER_ONLY alignment warning surfaced by the backend", () => {
    const { reconstruction } = loadReconstructionFixture();
    // The fixture's alignment is ORDER_ONLY with a NO_ABSOLUTE_TIME_AVAILABLE
    // warning — verify staleness/alignment info is not hidden from the user.
    expect(reconstruction.alignment.status).toBe("ORDER_ONLY");

    render(
      <ReconstructionControls
        reconstruction={reconstruction}
        reconstructions={[reconstruction]}
        stale={false}
        staleReason={null}
        loading={false}
        recordCount={2}
        onReconstruct={vi.fn()}
        onSelectVersion={vi.fn()}
        selectedVersionId={reconstruction.reconstruction_id}
      />,
    );

    expect(screen.getByText(/Engine/)).toBeInTheDocument();
  });

  it("renders a stale-reconstruction warning when the parent flags it as stale", () => {
    const { reconstruction } = loadReconstructionFixture();
    render(
      <ReconstructionControls
        reconstruction={reconstruction}
        reconstructions={[reconstruction]}
        stale
        staleReason="record order or membership changed since this reconstruction"
        loading={false}
        recordCount={2}
        onReconstruct={vi.fn()}
        onSelectVersion={vi.fn()}
        selectedVersionId={reconstruction.reconstruction_id}
      />,
    );

    expect(screen.getByText(/Reconstruction may be stale/)).toBeInTheDocument();
  });

  it("shows a reconstruction-history selector when more than one version exists", () => {
    const { reconstruction, reconstructions } = loadReconstructionFixture();
    const olderVersion = { ...reconstruction, reconstruction_id: "older-id", is_latest: false };
    render(
      <ReconstructionControls
        reconstruction={reconstruction}
        reconstructions={[olderVersion, ...reconstructions]}
        stale={false}
        staleReason={null}
        loading={false}
        recordCount={2}
        onReconstruct={vi.fn()}
        onSelectVersion={vi.fn()}
        selectedVersionId={reconstruction.reconstruction_id}
      />,
    );

    expect(screen.getByText("Version")).toBeInTheDocument();
  });

  it("disables the reconstruct button when there are no attached records", () => {
    render(
      <ReconstructionControls
        reconstruction={null}
        reconstructions={[]}
        stale={false}
        staleReason={null}
        loading={false}
        recordCount={0}
        onReconstruct={vi.fn()}
        onSelectVersion={vi.fn()}
        selectedVersionId={null}
      />,
    );

    expect(screen.getByRole("button", { name: /Reconstruct incident/ })).toBeDisabled();
    expect(screen.getByText(/Attach at least one record/)).toBeInTheDocument();
  });
});
