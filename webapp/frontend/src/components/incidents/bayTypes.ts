import type { AssetType } from "../../api/client";

/**
 * The bay an incident happened in, which picks the layout "Lokasi gangguan"
 * tests: a 1½-breaker diameter (line bays) or a transformer bay. Left empty,
 * it is read from the relays in the records.
 */
export const BAY_TYPE_OPTIONS: Array<{ value: "" | AssetType; label: string }> = [
  { value: "", label: "Otomatis (dari rekaman)" },
  { value: "TRANSMISSION_LINE", label: "Penghantar / diameter 1½ CB" },
  { value: "TRANSFORMER", label: "Trafo" },
];
