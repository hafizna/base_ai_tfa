import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import EpisodeCards from "../components/incidents/EpisodeCards";
import type { FaultEpisodeOut, IncidentRecordOut } from "../api/client";

function makeRecord(id: string, filename: string): IncidentRecordOut {
  return {
    incident_record_id: id,
    incident_id: "inc-1",
    analysis_id: `analysis-${id}`,
    source_filename: filename,
    station_name: "GI TEST",
    bay_name: "BAY 1",
    relay_id: id === "r1" ? "RELAY_A" : "RELAY_B",
    relay_model: null,
    protection_type: "21",
    record_start_iso: "2026-01-01T00:00:00+00:00",
    trigger_time_iso: "2026-01-01T00:00:00.300+00:00",
    trigger_offset_s: 0.3,
    sequence_index: id === "r1" ? 0 : 1,
    manual_order: null,
    order_source: "ABSOLUTE_TIME",
    attachment_role: "UNKNOWN",
    inclusion_status: "INCLUDED",
    exclusion_reason: null,
    canonical_snapshot: {},
    attachment_warnings: [],
    created_at: "2026-01-01T00:00:00+00:00",
  };
}

describe("Duplicate captures are grouped into a single episode", () => {
  it("renders both duplicate-capture records as members of one episode card, not two", () => {
    const records = [makeRecord("r1", "relay_capture.cfg"), makeRecord("r2", "dfr_capture.cfg")];
    const episodes: FaultEpisodeOut[] = [
      {
        episode_id: "ep-1",
        incident_id: "inc-1",
        member_record_ids: ["r1", "r2"],
        episode_index: 0,
        start_iso: "2026-01-01T00:00:00.300+00:00",
        end_iso: "2026-01-01T00:00:00.300+00:00",
        duration_ms: 100,
        faulted_phases: ["A"],
        fault_type: "SLG",
        zone_operations: [],
        trip_types: [],
        reclose_outcome: "successful",
        electrical_summary: {},
        local_cause_hypotheses: [],
        relationship_to_previous: null,
        confidence: 0.9,
        observed_facts: {},
        interpretation: {},
        missing_evidence: [
          { type: "MULTIPLE_RECORDS_ONE_EPISODE", description: "This episode is backed by 2 records; treat as one electrical event." },
        ],
        provenance: {},
      },
    ];

    render(<EpisodeCards episodes={episodes} records={records} />);

    // Exactly one episode card, and it lists both duplicate-capture records
    // as members rather than splitting them into separate episodes.
    expect(screen.getAllByText(/Episode 1/).length).toBeGreaterThan(0);
    expect(screen.queryByText(/Episode 2/)).not.toBeInTheDocument();
    expect(screen.getByText("relay_capture.cfg (RELAY_A)")).toBeInTheDocument();
    expect(screen.getByText("dfr_capture.cfg (RELAY_B)")).toBeInTheDocument();
    expect(screen.getByText(/This episode is backed by 2 records/)).toBeInTheDocument();
  });
});
