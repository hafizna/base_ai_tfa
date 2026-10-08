import { describe, expect, it } from "vitest";
import type { IncidentOut, ReconstructionOut } from "../api/client";
import {
  buildIncidentStory,
  formatClock,
  isReconstructionStale,
  isoToMs,
  phaseLabel,
  type SequenceEntry,
} from "../components/incidents/incidentStory";
import bringinRaw from "./fixtures/incident_bringin.json";
import cibatuRaw from "./fixtures/incident_cibatu.json";

// Real API responses (incident + reconstruction) built from the field records:
// - Bringin ZQ6D/E/F, 21 Aug 2023: S-T fault, reclose after 5.0 s, same fault 5.7 s later.
// - Cibatu–Mekarsari 2, 26 Apr 2024: R-N fault tripped by Z2 + carrier receive, 1-pole reclose.
type Fixture = { incident: IncidentOut; reconstruction: ReconstructionOut };
const bringin = bringinRaw as unknown as Fixture;
const cibatu = cibatuRaw as unknown as Fixture;

function cards(sequence: SequenceEntry[]) {
  return sequence.flatMap((e) => (e.type === "card" ? [e.card] : []));
}
function connectors(sequence: SequenceEntry[]) {
  return sequence.flatMap((e) => (e.type === "connector" ? [e.connector] : []));
}

describe("formatting helpers", () => {
  it("uses PLN phase names", () => {
    expect(phaseLabel(["B", "C"])).toBe("S-T");
    expect(phaseLabel(["A"])).toBe("R-N");
    expect(phaseLabel(["C", "A", "B"])).toBe("R-S-T");
  });

  it("reads naive ISO times as written and prints milliseconds with a comma", () => {
    const ms = isoToMs("2023-08-21T15:15:03.738333");
    expect(ms).not.toBeNull();
    expect(formatClock(ms as number)).toBe("15:15:03,738");
    expect(isoToMs(null)).toBeNull();
  });
});

describe("incident story — Bringin reclose then re-fault", () => {
  const story = buildIncidentStory(bringin.incident, bringin.reconstruction);

  it("summarises the sequence", () => {
    expect(story.headline).toBe("Gangguan berulang setelah reclose");
    expect(story.chips.map((c) => c.label)).toEqual(["Reclose did not hold", "Final state: trip — no further reclose"]);
    expect(story.narrative).toBe(
      "Line MJSNG2 trip karena gangguan fasa S-T, berhasil reclose setelah 5,0 detik, lalu terganggu lagi " +
        "5,7 detik kemudian di fasa yang sama dan trip kembali. Tidak ada reclose berikutnya yang terekam.",
    );
    expect(story.tiles.map((t) => [t.label, t.value])).toEqual([
      ["First fault", "15:15:03,738"],
      ["Fault current", "±6,4 kA"],
      ["Dead time", "5,0 s"],
      ["Re-fault after reclose", "+5,7 s"],
      ["Analysed line", "MJSNG2"],
    ]);
    expect(story.sequenceMeta).toBe("Dari 3 rekaman · waktu menurut jam DFR GI BRINGIN");
  });

  it("lays out fault, reclose, re-fault and the final state in order", () => {
    const [fault1, reclose, fault2, after] = cards(story.sequence);
    expect(fault1).toMatchObject({ title: "Gangguan #1", time: "15:15:03,738", headline: "Fasa S-T · ±6,4 kA", recordName: "ZQ6D" });
    expect(fault1.bullets).toEqual(["Trip 3-pole +45 ms · Z1", "Gangguan padam setelah 65 ms"]);
    expect(reclose).toMatchObject({ title: "Reclose successful", time: "15:15:08,834", headline: "Line bertegangan kembali", recordName: "ZQ6E" });
    expect(reclose.bullets).toEqual(["Voltage 83 kV, load ±470 A", "No fault current"]);
    expect(fault2).toMatchObject({ title: "Gangguan #2", time: "15:15:14,564", headline: "Fasa S-T · ±6,0 kA", emphasis: true });
    expect(fault2.bullets).toContain("Fasa sama dengan gangguan #1");
    expect(after).toMatchObject({ kind: "after", headline: "Tidak ada reclose berikutnya terekam" });
    expect(connectors(story.sequence)).toEqual([
      { kind: "dead_time", label: "Dead time 5,0 s", detail: "CB open" },
      { kind: "refault", label: "5,7 s kemudian", detail: "reclose did not hold" },
    ]);
  });

  it("separates the sequence pattern from the per-record AI readings", () => {
    const { cause } = story;
    expect(cause.status.label).toBe("Unconfirmed");
    expect(cause.headline).toBe("Indikasi kontak fisik (pohon / benda asing)");
    expect(cause.pattern?.strength).toBe("Pattern evidence · medium");
    expect(cause.pattern?.text).toContain("5,7 detik setelah reclose berhasil, di fasa yang sama (S-T)");
    expect(cause.ai).toEqual([
      {
        title: "Gangguan #1",
        recordName: "ZQ6D",
        kind: "reading",
        cause: "Petir",
        percent: 82,
        note: "Turun dari 92% karena gangguan berulang setelah reclose tidak cocok dengan penyebab transien",
      },
      { title: "Reclose", recordName: "ZQ6E", kind: "skipped", note: "Tidak dianalisa — rekaman reclose, tanpa gangguan" },
      {
        title: "Gangguan #2",
        recordName: "ZQ6F",
        kind: "no_dominant",
        candidates: [
          { cause: "Petir", percent: 35 },
          { cause: "Benda asing", percent: 32 },
          { cause: "Layang-layang", percent: 31 },
        ],
      },
    ]);
    expect(cause.footnote).toContain("belum sepakat");
  });

  it("lists what to check and the records with their roles", () => {
    expect(story.checklist.map((c) => c.id)).toEqual(["row", "lds", "two-ended", "bay"]);
    expect(story.checklist.find((c) => c.id === "lds")?.detail).toBe("21 Agu 2023 sekitar 15:15 (jam DFR) di koridor line");
    expect(story.records.map((r) => [r.name, `${r.roleLabel}${r.roleSuffix}`, r.start, r.line])).toEqual([
      ["ZQ6D", "Gangguan #1 + trip", "15:15:03,620", "MJSNG2"],
      ["ZQ6E", "Reclose (mulai saat dead time)", "15:15:08,700", "MJSNG2"],
      ["ZQ6F", "Gangguan #2 + trip", "15:15:14,440", "MJSNG2"],
    ]);
    expect(story.records[0].note).toContain("MJSNG1 de-energized");
    expect(story.records[2].note).toContain("5,7 s setelah reclose");
  });
});

describe("incident story — Cibatu teleprotection-aided trip", () => {
  const story = buildIncidentStory(cibatu.incident, cibatu.reconstruction);

  it("reads the trip as Z2 accelerated by carrier receive, single pole R", () => {
    const [fault, reclose, after] = cards(story.sequence);
    expect(fault.headline).toBe("Fasa R-N · ±12,0 kA");
    expect(fault.bullets[0]).toBe("Trip pole R +37 ms · Z2 + carrier receive (teleprotection-aided)");
    expect(reclose).toMatchObject({ title: "Reclose successful", headline: "CB menutup kembali" });
    expect(reclose.bullets).toContain("Single-pole reclose");
    expect(after.headline).toBe("Line kembali beroperasi");
    expect(connectors(story.sequence)).toEqual([{ kind: "dead_time", label: "Dead time 1,0 s", detail: "CB open" }]);
  });

  it("summarises a transient fault and asks to confirm Zone 1 at the remote end", () => {
    expect(story.headline).toBe("Gangguan sementara, reclose berhasil");
    expect(story.chips.map((c) => c.label)).toEqual(["Reclose successful", "Final state: line in service"]);
    expect(story.cause.headline).toBe("Gangguan sementara (transien)");
    expect(story.cause.pattern?.strength).toBe("Pattern evidence · weak");
    expect(story.cause.ai[0]).toMatchObject({ kind: "reading", cause: "Petir", percent: 92 });
    expect(story.cause.ai[0].note).toBeUndefined();
    expect(story.cause.footnote).toContain("searah");
    expect(story.checklist.map((c) => c.id)).toEqual(["lds", "remote-z1", "two-ended", "bay"]);
  });
});

describe("isReconstructionStale", () => {
  it("is fresh when the reconstruction covers exactly the attached records", () => {
    expect(isReconstructionStale(bringin.incident, bringin.reconstruction)).toBe(false);
  });

  it("is stale when a record was added or none exists yet", () => {
    const extra = { ...bringin.incident.records[0], analysis_id: "new-record", incident_record_id: "new" };
    expect(isReconstructionStale({ ...bringin.incident, records: [...bringin.incident.records, extra] }, bringin.reconstruction)).toBe(true);
    expect(isReconstructionStale(bringin.incident, null)).toBe(true);
    expect(isReconstructionStale({ ...bringin.incident, records: [] }, null)).toBe(false);
  });
});
