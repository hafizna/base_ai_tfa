import { useEffect, useMemo, useState } from "react";

import type { ComtradeData } from "../../context/AnalysisContext";
import { recalculateRatio } from "../../api/client";
import styles from "./Panel.module.css";

interface Props {
  analysisId: string;
  comtrade: ComtradeData;
  onUpdate: (updated: ComtradeData) => void;
}

interface RatioGroup {
  type: "CT" | "VT";
  channelIds: string[];
  cfgPrimary: number;
  cfgSecondary: number;
  newPrimary: number;
  newSecondary: number;
}

// ---------------------------------------------------------------------------
// VT system voltage knowledge base (Indonesian transmission systems)
// Ratio = primary / secondary as stored in COMTRADE CFG.
// When secondary=1 (ratio convention), primary IS the ratio number.
// ---------------------------------------------------------------------------
const VT_SYSTEMS: { kv: number; label: string }[] = [
  { kv: 30, label: "30 kV" },
  { kv: 70, label: "70 kV" },
  { kv: 150, label: "150 kV" },
  { kv: 275, label: "275 kV" },
  { kv: 500, label: "500 kV" },
];

// Known ratios per system voltage and secondary convention (V → ratio)
// secondary=1  means the CFG stores ratio directly (e.g., 1500:1 for 150kV/100V)
// secondary=100 / 110 / 115 means the CFG stores actual voltages
const VT_RATIO_TABLE: Array<{ ratio: number; kv: number; secV: number }> = [
  { ratio: 300,  kv: 30,  secV: 100 }, { ratio: 273,  kv: 30,  secV: 110 },
  { ratio: 700,  kv: 70,  secV: 100 }, { ratio: 636,  kv: 70,  secV: 110 },
  { ratio: 1500, kv: 150, secV: 100 }, { ratio: 1364, kv: 150, secV: 110 },
  { ratio: 1304, kv: 150, secV: 115 },
  { ratio: 2750, kv: 275, secV: 100 }, { ratio: 2500, kv: 275, secV: 110 },
  { ratio: 2391, kv: 275, secV: 115 },
  { ratio: 5000, kv: 500, secV: 100 }, { ratio: 4545, kv: 500, secV: 110 },
  { ratio: 4348, kv: 500, secV: 115 },
];

/** Detect system voltage in kV from primary/secondary ratio. Returns null if unrecognised. */
function detectVtKv(primary: number, secondary: number): { kv: number; secV: number } | null {
  const ratio = secondary > 0 ? primary / secondary : primary;
  const match = VT_RATIO_TABLE.find((m) => Math.abs(m.ratio - ratio) / (m.ratio || 1) < 0.03);
  return match ? { kv: match.kv, secV: match.secV } : null;
}

/** Given a target system voltage and current secondary convention, return the standard primary. */
function vtPrimaryForSystem(kv: number, secondary: number): number {
  if (secondary <= 1) {
    // ratio-as-primary convention (secondary=1)
    // 275kV stored as 2500:1 (110V physical secondary), others as kv×10 (100V secondary)
    if (kv === 275) return 2500;
    return kv * 10;
  }
  // actual-voltage convention (secondary=100/110/115 etc.)
  return Math.round((kv * 1000) / secondary);
}

const CT_SECONDARY_OPTIONS = [1, 5];
const VT_SECONDARY_OPTIONS = [1, 100, 110, 115, 125];

/**
 * Expand a compressed VT ratio to actual-voltage convention.
 * Some manufacturers store 150kV/100V as 1500/1 (simplified).
 * When secondary ≤ 1 and the ratio matches a known system, expand to e.g. 150000/100
 * so the input fields show recognisable values.
 * The ratio (primary/secondary) is preserved, so no recalculation error is introduced.
 */
function expandVtRatio(primary: number, secondary: number): { primary: number; secondary: number } {
  if (secondary > 1) return { primary, secondary };
  const detected = detectVtKv(primary, secondary);
  if (!detected) return { primary, secondary };
  return { primary: Math.round(primary * detected.secV), secondary: detected.secV };
}

function buildGroups(comtrade: ComtradeData): RatioGroup[] {
  const currChs = comtrade.analog_channels.filter((ch) => ch.measurement === "current");
  const voltChs = comtrade.analog_channels.filter((ch) => ch.measurement === "voltage");
  const groups: RatioGroup[] = [];

  if (currChs.length > 0) {
    const cfgP = currChs[0].ct_primary;
    const cfgS = currChs[0].ct_secondary;
    groups.push({
      type: "CT",
      channelIds: currChs.map((ch) => ch.id),
      cfgPrimary: cfgP,
      cfgSecondary: cfgS,
      newPrimary: cfgP,
      newSecondary: cfgS,
    });
  }

  if (voltChs.length > 0) {
    const cfgP = voltChs[0].ct_primary;
    const cfgS = voltChs[0].ct_secondary;
    const expanded = expandVtRatio(cfgP, cfgS);
    groups.push({
      type: "VT",
      channelIds: voltChs.map((ch) => ch.id),
      cfgPrimary: cfgP,
      cfgSecondary: cfgS,
      newPrimary: expanded.primary,
      newSecondary: expanded.secondary,
    });
  }

  return groups;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function CTVTRatioCorrection({ analysisId, comtrade, onUpdate }: Props) {
  const baseGroups = useMemo(() => buildGroups(comtrade), [comtrade]);
  const [groups, setGroups] = useState<RatioGroup[]>(baseGroups);
  const [loading, setLoading] = useState(false);
  const [applied, setApplied] = useState(false);

  useEffect(() => {
    setGroups(baseGroups);
    setApplied(false);
  }, [baseGroups]);

  function updateField(type: "CT" | "VT", field: "newPrimary" | "newSecondary", raw: string | number) {
    const value = typeof raw === "number" ? raw : parseFloat(raw);
    if (!Number.isFinite(value) || value <= 0) return;
    setGroups((prev) => prev.map((g) => (g.type === type ? { ...g, [field]: value } : g)));
    setApplied(false);
  }

  function applySystemVoltage(kv: number) {
    setGroups((prev) =>
      prev.map((g) => {
        if (g.type !== "VT") return g;
        return { ...g, newPrimary: vtPrimaryForSystem(kv, g.newSecondary) };
      })
    );
    setApplied(false);
  }

  function reset() {
    setGroups(baseGroups);
    setApplied(false);
  }

  async function apply() {
    setLoading(true);
    try {
      const ratios = groups.flatMap((g) =>
        g.channelIds.map((id) => ({ channel_id: id, primary: g.newPrimary, secondary: g.newSecondary }))
      );
      const updated = await recalculateRatio(analysisId, ratios);
      onUpdate(updated);
      setApplied(true);
    } finally {
      setLoading(false);
    }
  }

  const hasChanges = groups.some((g) => {
    const oldR = g.cfgSecondary !== 0 ? g.cfgPrimary / g.cfgSecondary : 1;
    const newR = g.newSecondary !== 0 ? g.newPrimary / g.newSecondary : 1;
    return Math.abs(newR / (oldR || 1) - 1) > 0.001;
  });

  const vtGroup = groups.find((g) => g.type === "VT");
  const detectedSystem = vtGroup ? detectVtKv(vtGroup.cfgPrimary, vtGroup.cfgSecondary) : null;
  const currentDetected = vtGroup ? detectVtKv(vtGroup.newPrimary, vtGroup.newSecondary) : null;

  return (
    <div className={styles.panel} style={{ padding: "14px 20px" }}>
      {/* Main row */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "#64748b", whiteSpace: "nowrap" }}>
          Rasio CT / VT
        </span>
        <span className={styles.badge} style={{ fontSize: "0.7rem", background: "#f1f5f9", color: "#94a3b8" }}>
          dari .cfg
        </span>

        {/* Ratio inputs */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          {groups.map((g) => {
            const factor = (() => {
              const oldR = g.cfgSecondary > 0 ? g.cfgPrimary / g.cfgSecondary : 1;
              const newR = g.newSecondary > 0 ? g.newPrimary / g.newSecondary : 1;
              return oldR > 0 ? newR / oldR : 1;
            })();
            const changed = Math.abs(factor - 1) > 0.001;
            return (
              <div key={g.type} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <span
                  className={styles.badge}
                  style={
                    g.type === "CT"
                      ? { background: "#eff6ff", color: "#3b82f6", fontSize: "0.72rem" }
                      : { background: "#faf5ff", color: "#7c3aed", fontSize: "0.72rem" }
                  }
                >
                  {g.type}
                </span>
                <input
                  className={styles.ratioInput}
                  type="number"
                  min={0.001}
                  step="any"
                  value={g.newPrimary}
                  style={{ width: 72 }}
                  onChange={(e) => updateField(g.type, "newPrimary", e.target.value)}
                />
                <span style={{ color: "#94a3b8", fontSize: "0.85rem" }}>/</span>
                <select
                  className={styles.ratioInput}
                  value={g.newSecondary}
                  style={{ width: 68 }}
                  onChange={(e) => updateField(g.type, "newSecondary", parseFloat(e.target.value))}
                >
                  {(g.type === "CT" ? CT_SECONDARY_OPTIONS : VT_SECONDARY_OPTIONS).map((v) => (
                    <option key={v} value={v}>
                      {g.type === "CT" ? `${v} A` : v === 1 ? "1 (rasio)" : `${v} V`}
                    </option>
                  ))}
                </select>
                {changed && (
                  <span style={{ fontSize: "0.72rem", color: "#dc2626", fontWeight: 700, whiteSpace: "nowrap" }}>
                    ×{factor.toFixed(3)}
                  </span>
                )}
              </div>
            );
          })}
        </div>

        {/* Buttons */}
        <div style={{ display: "flex", gap: 8, marginLeft: "auto" }}>
          <button
            className={styles.applyBtn}
            onClick={apply}
            disabled={loading || !hasChanges}
            style={{ padding: "5px 14px", fontSize: "0.8rem" }}
          >
            {loading ? "Menghitung..." : applied ? "Terapkan Ulang" : "Terapkan & Hitung Ulang"}
          </button>
          <button
            onClick={reset}
            disabled={loading || !hasChanges}
            style={{
              padding: "5px 12px",
              fontSize: "0.8rem",
              border: "1.5px solid #cbd5e1",
              borderRadius: 7,
              background: "#fff",
              color: "#64748b",
              cursor: hasChanges ? "pointer" : "default",
            }}
          >
            Reset
          </button>
        </div>
        {applied && (
          <span className={styles.badge} style={{ background: "#f0fdf4", color: "#16a34a" }}>
            Diterapkan
          </span>
        )}
      </div>

      {/* VT system voltage quick-selector */}
      {vtGroup && (
        <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <span style={{ fontSize: "0.72rem", color: "#64748b", whiteSpace: "nowrap" }}>
            Tegangan sistem:
          </span>
          {VT_SYSTEMS.map(({ kv, label }) => {
            const isActive = currentDetected?.kv === kv;
            return (
              <button
                key={kv}
                type="button"
                onClick={() => applySystemVoltage(kv)}
                style={{
                  padding: "2px 10px",
                  fontSize: "0.72rem",
                  fontWeight: 600,
                  borderRadius: 99,
                  border: "1.5px solid",
                  borderColor: isActive ? "#7c3aed" : "#cbd5e1",
                  background: isActive ? "#faf5ff" : "#fff",
                  color: isActive ? "#7c3aed" : "#64748b",
                  cursor: "pointer",
                }}
              >
                {label}
              </button>
            );
          })}
        </div>
      )}

      {/* Hint line */}
      <div style={{ fontSize: "0.7rem", color: "#94a3b8", marginTop: 6, lineHeight: 1.5 }}>
        {groups.every((g) => g.cfgPrimary === 1 && g.cfgSecondary === 1) ? (
          <span style={{ color: "#b45309" }}>
            <strong>Rasio tidak tersimpan di .cfg.</strong> File ini menyimpan faktor skala langsung di
            multiplier tiap kanal (bukan field ct_primary/ct_secondary). Nilai sampel sudah dalam satuan
            primer (A / kV). Tidak perlu koreksi kecuali multiplier di .cfg tidak sesuai nameplate CT/VT.
          </span>
        ) : (
          <>
            {vtGroup && detectedSystem && (
              <span>
                Dari .cfg: VT rasio {vtGroup.cfgPrimary}/{vtGroup.cfgSecondary} →{" "}
                <strong style={{ color: "#7c3aed" }}>
                  sistem {detectedSystem.kv} kV ({detectedSystem.secV} V sekunder)
                </strong>
                .{" "}
              </span>
            )}
            {vtGroup && !detectedSystem && (
              <span>
                Dari .cfg: VT rasio {vtGroup.cfgPrimary}/{vtGroup.cfgSecondary} (tegangan sistem tidak dikenali — pilih di atas).{" "}
              </span>
            )}
            Nilai Z, R, X dihitung dari sampel yang sudah diskalakan. Koreksi hanya perlu jika rasio di .cfg{" "}
            <em>tidak sesuai</em> setting CT/VT aktual di lapangan.
          </>
        )}
      </div>
    </div>
  );
}
