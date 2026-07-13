import { createContext, useContext, useState } from "react";
import type { ReactNode } from "react";

export type RelayType = "LINE" | "21" | "87L" | "87T" | "OCR" | "REF" | "SBEF" | "CCP" | "TWS_FL";

export interface AnalogChannel {
  id: string;
  name: string;
  canonical_name: string;
  unit: string;
  phase: string | null;
  measurement: string;
  ct_primary: number;
  ct_secondary: number;
  pors: string;
  samples: number[];
}

export interface StatusChannel {
  id: string;
  name: string;
  samples: number[];
}

export interface ComtradeData {
  station_name: string;
  rec_dev_id: string;
  rev_year: string;
  sampling_rates: [number, number][];
  trigger_time: number;
  // Absolute COMTRADE timing metadata (Stage 0). null when the source file did
  // not carry a usable wall-clock timestamp - never inferred/guessed client-side.
  start_time_iso?: string | null;
  trigger_time_iso?: string | null;
  trigger_offset_s?: number;
  time_code?: string | null;
  local_code?: string | null;
  clock_quality?: string | null;
  total_samples: number;
  frequency: number;
  time: number[];
  analog_channels: AnalogChannel[];
  status_channels: StatusChannel[];
  warnings: string[];
}

interface AnalysisState {
  relayType: RelayType | null;
  comtrade: ComtradeData | null;
  setRelayType: (t: RelayType) => void;
  setComtrade: (d: ComtradeData) => void;
  reset: () => void;
}

const Ctx = createContext<AnalysisState | null>(null);

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [relayType, setRelayType] = useState<RelayType | null>(null);
  const [comtrade, setComtrade] = useState<ComtradeData | null>(null);

  function reset() {
    setRelayType(null);
    setComtrade(null);
  }

  return (
    <Ctx.Provider value={{ relayType, comtrade, setRelayType, setComtrade, reset }}>
      {children}
    </Ctx.Provider>
  );
}

export function useAnalysis() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useAnalysis must be inside AnalysisProvider");
  return ctx;
}
