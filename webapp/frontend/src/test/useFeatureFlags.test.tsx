import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useMultiComtradeEnabled } from "../hooks/useFeatureFlags";
import * as client from "../api/client";

describe("useMultiComtradeEnabled", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("reflects multi_comtrade_enabled=false from the backend health check", async () => {
    vi.spyOn(client, "fetchHealth").mockResolvedValue({
      status: "ok",
      version: "2.0.0",
      analysis_storage: "filesystem",
      analysis_ttl_hours: 24,
      warmup: {},
      feature_flags: { multi_comtrade_enabled: false },
    });

    const { result } = renderHook(() => useMultiComtradeEnabled());

    await waitFor(() => expect(result.current).toBe(false));
  });

  it("defaults to enabled while the health check is pending or unreachable", async () => {
    vi.spyOn(client, "fetchHealth").mockRejectedValue(new Error("network error"));

    const { result } = renderHook(() => useMultiComtradeEnabled());

    // Should remain the optimistic default even after the rejected promise settles.
    await waitFor(() => expect(client.fetchHealth).toHaveBeenCalled());
    expect(result.current).toBe(true);
  });
});
