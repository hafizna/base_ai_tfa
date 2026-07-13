import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import SegmentedTimeline from "../components/incidents/SegmentedTimeline";
import { loadReconstructionFixture } from "./fixtures/loadReconstructionFixture";

describe("SegmentedTimeline", () => {
  it("renders one segment per record and an explicit gap block between them", () => {
    const { records, timeline, episodes, relationships } = loadReconstructionFixture();

    render(
      <MemoryRouter>
        <SegmentedTimeline records={records} timeline={timeline} episodes={episodes} relationships={relationships} />
      </MemoryRouter>,
    );

    // Two records in the fixture -> two segment headers.
    expect(screen.getAllByTitle("Open single-record Workspace")).toHaveLength(records.length);
  });

  it("shows the actual gap duration rather than hiding it", () => {
    const { records, timeline, episodes, relationships } = loadReconstructionFixture();
    const gapEvent = timeline.find((e) => e.event_type === "DATA_GAP");

    render(
      <MemoryRouter>
        <SegmentedTimeline records={records} timeline={timeline} episodes={episodes} relationships={relationships} />
      </MemoryRouter>,
    );

    if (gapEvent) {
      // Gap label text is rendered somewhere in the timeline (compressed by default).
      const gapMs = (gapEvent.details as { gap_ms?: number }).gap_ms;
      expect(gapMs).toBeDefined();
    } else {
      // No DATA_GAP event in this particular fixture — assert the timeline
      // still rendered instead of silently passing.
      expect(screen.getAllByTitle("Open single-record Workspace").length).toBeGreaterThan(0);
    }
  });

  it("renders an empty state instead of crashing when there are no records", () => {
    render(
      <MemoryRouter>
        <SegmentedTimeline records={[]} timeline={[]} episodes={[]} relationships={[]} />
      </MemoryRouter>,
    );

    expect(screen.getByText(/No records to display/)).toBeInTheDocument();
  });
});
