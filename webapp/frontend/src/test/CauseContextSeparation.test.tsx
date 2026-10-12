import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import AIFaultResultView, { type AIFaultResult } from "../components/relay/shared/AIFaultResultView";

const result: AIFaultResult = {
  cause_ranking: [{ cause: "POHON", label: "Pohon", confidence: .63 }],
  fault_type: "permanent", overall_confidence: .63, evidence: [],
  event_context: { restoration_outcome: "failed", summary: "Pemulihan gagal pada reclose terekam." },
  cause_assessment: { status: "hypothesis", confirmed: false, confidence_kind: "model_score",
    additional_episodes_evaluated_by_model: false },
  tier1: { fired: true, scope: "context", confidence: .9, label: "Konteks reclose", rule_name: "explicit_failed_reclose" },
};

describe("cause and recovery evidence", () => {
  it("keeps the context rule score separate from the cause model score", () => {
    render(<AIFaultResultView result={result} />);
    expect(screen.getByLabelText("Overall confidence 63%")).toBeInTheDocument();
    expect(screen.queryByLabelText("Overall confidence 90%")).not.toBeInTheDocument();
    expect(screen.getByText("Cause model score")).toBeInTheDocument();
    expect(screen.getByText("Context rule:")).toBeInTheDocument();
    expect(screen.getByText("Pemulihan gagal pada reclose terekam.")).toBeInTheDocument();
    expect(screen.queryByText("Permanent fault behavior")).not.toBeInTheDocument();
    expect(screen.getByText(/Model penyebab memakai fitur episode pertama/)).toBeInTheDocument();
  });

  it("does not present a fabricated score when cause inference is unavailable", () => {
    render(<AIFaultResultView result={{ ...result, cause_ranking: [], overall_confidence: 0,
      cause_assessment: { status: "unavailable" } }} />);
    expect(screen.getByText("Penyebab belum dapat dinilai")).toBeInTheDocument();
    expect(screen.queryByText("Cause model score")).not.toBeInTheDocument();
  });
});
