import { useEffect, useState } from "react";
import { fetchHealth } from "../api/client";

/**
 * Reads Stage 2 (multi-COMTRADE incident reconstruction) feature-flag state
 * from GET /api/health. Defaults to enabled while the health check is in
 * flight or if it fails, matching the backend's own default-on behavior
 * (MULTI_COMTRADE_ENABLED defaults to "1") — so a slow/unreachable health
 * check does not spuriously hide functionality that the backend would
 * otherwise serve.
 */
export function useMultiComtradeEnabled(): boolean {
  const [enabled, setEnabled] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetchHealth()
      .then((health) => {
        if (!cancelled && health.feature_flags) {
          setEnabled(health.feature_flags.multi_comtrade_enabled !== false);
        }
      })
      .catch(() => {
        // Backend unreachable — keep the optimistic default so a health-check
        // hiccup doesn't hide the whole Stage 2 UI; the actual API calls will
        // surface their own errors if the backend is truly down.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return enabled;
}
