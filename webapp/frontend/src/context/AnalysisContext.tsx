import { createContext, useContext, useState } from "react";
import type { ReactNode } from "react";

export type RelayType = "21" | "87L" | "87T" | "OCR" | "REF" | "SBEF";

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
