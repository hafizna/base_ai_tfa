import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import RelationshipInspector from "../components/incidents/RelationshipInspector";
import * as client from "../api/client";
import { loadReconstructionFixture } from "./fixtures/loadReconstructionFixture";

describe("RelationshipInspector", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders evidence for/against and metrics for each relationship", () => {
    const { relationships } = loadReconstructionFixture();
    render(
      <RelationshipInspector
        incidentId="inc-1"
        relationships={relationships}
        recordLabel={(id) => id}
        onOverridden={vi.fn()}
      />,
    );

    expect(screen.getByText("Evidence for")).toBeInTheDocument();
    expect(screen.getByText("Evidence against")).toBeInTheDocument();
  });

  it("displays an UNCERTAIN relationship conservatively, not as a resolved conclusion", () => {
    const { relationships } = loadReconstructionFixture();
    const uncertain = relationships.find((r) => r.relationship_type === "UNCERTAIN");
    expect(uncertain).toBeDefined();

    render(
      <RelationshipInspector
        incidentId="inc-1"
        relationships={relationships}
        recordLabel={(id) => id}
        onOverridden={vi.fn()}
      />,
    );

    expect(screen.getByText(/Insufficient evidence to classify this pair with confidence/)).toBeInTheDocument();
  });

  it("filters relationships by the uncertain/unrelated group", () => {
    const { relationships } = loadReconstructionFixture();
    render(
      <RelationshipInspector
        incidentId="inc-1"
        relationships={relationships}
        recordLabel={(id) => id}
        onOverridden={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByText("Uncertain / unrelated"));
    expect(screen.getByText("UNCERTAIN")).toBeInTheDocument();
  });

  it("requires both operator name and reason before submitting an override", async () => {
    const { relationships } = loadReconstructionFixture();
    const overrideSpy = vi.spyOn(client, "overrideRelationship").mockResolvedValue({
      ...relationships[0],
      relationship_type: "CONTINUATION",
      overridden: true,
    });

    render(
      <RelationshipInspector
        incidentId="inc-1"
        relationships={relationships}
        recordLabel={(id) => id}
        onOverridden={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Override relationship" }));
    fireEvent.click(screen.getByRole("button", { name: "Save override" }));

    expect(await screen.findByText(/Operator name and reason are required/)).toBeInTheDocument();
    expect(overrideSpy).not.toHaveBeenCalled();

    fireEvent.change(screen.getByPlaceholderText("e.g. engineer1"), { target: { value: "engineer1" } });
    fireEvent.change(screen.getByPlaceholderText("Why is the algorithm result wrong?"), {
      target: { value: "Field inspection confirmed continuation." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save override" }));

    await waitFor(() => expect(overrideSpy).toHaveBeenCalledWith(
      "inc-1",
      relationships[0].relationship_id,
      expect.objectContaining({ operator: "engineer1", reason: "Field inspection confirmed continuation." }),
    ));
  });

  it("shows an empty state instead of crashing when there are no relationships", () => {
    render(
      <RelationshipInspector incidentId="inc-1" relationships={[]} recordLabel={(id) => id} onOverridden={vi.fn()} />,
    );
    expect(screen.getByText(/No relationships computed yet/)).toBeInTheDocument();
  });
});
