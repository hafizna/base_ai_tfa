import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import BatchUploadPanel from "../components/incidents/BatchUploadPanel";
import * as client from "../api/client";

function makeFile(name: string, content = "data"): File {
  return new File([content], name, { type: "text/plain" });
}

/** A file as a folder picker hands it over: its name plus the path inside the picked folder. */
function folderFile(path: string): File {
  const file = makeFile(path.split("/").pop() as string);
  Object.defineProperty(file, "webkitRelativePath", { value: path });
  return file;
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

  it("pairs records per relay folder when a whole incident folder is picked", () => {
    render(<BatchUploadPanel incidentId="inc-1" onUploaded={vi.fn()} />);
    const input = screen.getByLabelText("Pilih folder rekaman") as HTMLInputElement;

    fireEvent.change(input, {
      target: {
        files: [
          folderFile("TRIP 7B2/CCP DIA 2/DR/rec1.cfg"),
          folderFile("TRIP 7B2/CCP DIA 2/DR/rec1.dat"),
          // Another relay naming its record alike: paired within its own folder, not flagged as a duplicate.
          folderFile("TRIP 7B2/CBF B/DR/rec1.cfg"),
          folderFile("TRIP 7B2/CBF B/DR/rec1.dat"),
          folderFile("TRIP 7B2/CBF B/Events/events.evt"),
          folderFile("TRIP 7B2/foto relay.jpg"),
        ],
      },
    });

    expect(screen.getByText("TRIP 7B2/CCP DIA 2/DR/rec1.cfg + TRIP 7B2/CCP DIA 2/DR/rec1.dat")).toBeInTheDocument();
    expect(screen.getByText("TRIP 7B2/CBF B/DR/rec1.cfg + TRIP 7B2/CBF B/DR/rec1.dat")).toBeInTheDocument();
    expect(screen.getByText("EVT")).toBeInTheDocument();
    expect(screen.getByText("5 file(s) selected")).toBeInTheDocument();
    expect(screen.getByText(/1 file lain dilewati/)).toBeInTheDocument();
    expect(screen.queryByText(/Duplicate files for stem/)).not.toBeInTheDocument();
  });

  it("lists the relay event files stored with the incident", async () => {
    const upload = vi.spyOn(client, "uploadIncidentRecords").mockResolvedValue({
      incident_id: "inc-1",
      records_created: [],
      errors: [],
      reconstruction_status: "completed",
      event_files: ["CBF B/Events/events.evt"],
    });
    render(<BatchUploadPanel incidentId="inc-1" onUploaded={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Pilih folder rekaman"), {
      target: { files: [folderFile("CBF B/DR/rec1.cfg"), folderFile("CBF B/DR/rec1.dat"), folderFile("CBF B/Events/events.evt")] },
    });

    fireEvent.click(screen.getByRole("button", { name: /Upload/ }));

    await waitFor(() => expect(upload).toHaveBeenCalledTimes(1));
    expect(await screen.findByText("✓ CBF B/Events/events.evt")).toBeInTheDocument();
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
