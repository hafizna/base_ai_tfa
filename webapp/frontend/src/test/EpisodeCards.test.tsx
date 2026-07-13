import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import EpisodeCards from "../components/incidents/EpisodeCards";
import { loadReconstructionFixture } from "./fixtures/loadReconstructionFixture";

describe("EpisodeCards", () => {
  it("renders one card per episode with facts, hypotheses, and missing evidence", () => {
    const { episodes, records } = loadReconstructionFixture();
    render(<EpisodeCards episodes={episodes} records={records} />);

    episodes.forEach((ep) => {
      expect(screen.getByText(`Episode ${ep.episode_index + 1}`)).toBeInTheDocument();
    });
    expect(screen.getAllByText(/Local cause hypotheses/).length).toBe(episodes.length);
  });

  it("labels local cause hypotheses as record-local, not confirmed root cause", () => {
    const { episodes, records } = loadReconstructionFixture();
    render(<EpisodeCards episodes={episodes} records={records} />);

    expect(screen.getAllByText(/record-local, not confirmed root cause/).length).toBeGreaterThan(0);
  });

  it("shows a helpful empty state instead of crashing when there are no episodes", () => {
    render(<EpisodeCards episodes={[]} records={[]} />);
    expect(screen.getByText(/No episodes reconstructed yet/)).toBeInTheDocument();
  });
});
