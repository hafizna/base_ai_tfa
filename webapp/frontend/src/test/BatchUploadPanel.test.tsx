import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import BatchUploadPanel from "../components/incidents/BatchUploadPanel";
import * as client from "../api/client";

function makeFile(name: string, content = "data"): File {
  return new File([content], name, { type: "text/plain" });
}

describe("BatchUploadPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("previews detected CFG/DAT pairs by filename stem", () => {
    render(<BatchUploadPanel incidentId="inc-1" onUploaded={vi.fn()} />);
    const input = screen.getByLabelText("Select COMTRADE files") as HTMLInputElement;

    fireEvent.change(input, {
      target: { files: [makeFile("record1.cfg"), makeFile("record1.dat")] },
    });

    expect(screen.getByText("CFG+DAT")).toBeInTheDocument();
    expect(screen.getByText("record1.cfg + record1.dat")).toBeInTheDocument();
  });

  it("shows an orphan CFG warning when no matching DAT is selected", () => {
    render(<BatchUploadPanel incidentId="inc-1" onUploaded={vi.fn()} />);
    const input = screen.getByLabelText("Select COMTRADE files") as HTMLInputElement;

    fireEvent.change(input, { target: { files: [makeFile("orphan.cfg")] } });

    expect(screen.getByText(/Orphan \.cfg file with no matching \.dat file/)).toBeInTheDocument();
  });

  it("shows a duplicate-stem error when two .cfg files share a stem", () => {
    render(<BatchUploadPanel incidentId="inc-1" onUploaded={vi.fn()} />);
    const input = screen.getByLabelText("Select COMTRADE files") as HTMLInputElement;

    fireEvent.change(input, {
      target: { files: [makeFile("record1.cfg"), makeFile("record1.cfg"), makeFile("record1.dat")] },
    });

    expect(screen.getByText(/Duplicate files for stem/)).toBeInTheDocument();
  });

  it("uploads files and refreshes records via onUploaded callback on success", async () => {
    const onUploaded = vi.fn();
    const mockResponse: client.BatchUploadResponse = {
      incident_id: "inc-1",
      records_created: [
        { analysis_id: "a1", incident_record_id: "ir1", source_files: ["record1.cfg", "record1.dat"], status: "created" },
      ],
      errors: [],
      reconstruction_status: "completed",
    };
    vi.spyOn(client, "uploadIncidentRecords").mockResolvedValue(mockResponse);

    render(<BatchUploadPanel incidentId="inc-1" onUploaded={onUploaded} />);
    const input = screen.getByLabelText("Select COMTRADE files") as HTMLInputElement;
    fireEvent.change(input, { target: { files: [makeFile("record1.cfg"), makeFile("record1.dat")] } });

    fireEvent.click(screen.getByRole("button", { name: /Upload/ }));

    await waitFor(() => expect(onUploaded).toHaveBeenCalledWith(mockResponse, false));
    expect(screen.getByText("COMPLETED")).toBeInTheDocument();
    expect(screen.getByText(/analysis_id: a1/)).toBeInTheDocument();
  });

  it("displays atomic-mode abort errors without silently discarding them", async () => {
    const onUploaded = vi.fn();
    const mockResponse: client.BatchUploadResponse = {
      incident_id: "inc-1",
      records_created: [],
      errors: [{ files: ["orphan.cfg"], reason: "Orphan .cfg file with no matching .dat file." }],
      reconstruction_status: "aborted_atomic",
    };
    vi.spyOn(client, "uploadIncidentRecords").mockResolvedValue(mockResponse);

    render(<BatchUploadPanel incidentId="inc-1" onUploaded={onUploaded} />);
    const input = screen.getByLabelText("Select COMTRADE files") as HTMLInputElement;
    fireEvent.change(input, {
      target: { files: [makeFile("record1.cfg"), makeFile("record1.dat"), makeFile("orphan.cfg")] },
    });

    // Enable partial success so the upload button is not blocked by client-side preview errors.
    fireEvent.click(screen.getByLabelText(/Partial-success mode/));
    fireEvent.click(screen.getByRole("button", { name: /Upload/ }));

    await waitFor(() => expect(screen.getByText("ABORTED ATOMIC")).toBeInTheDocument());
    expect(screen.getAllByText(/Orphan \.cfg file with no matching \.dat file/).length).toBeGreaterThan(0);
  });
});
