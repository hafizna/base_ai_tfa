/**
 * The incident page's story: what happened in order, what the sequence
 * pattern and the per-record AI say about the cause, and what to check next —
 * built from one incident and its reconstruction.
 *
 * Pure functions only, so every sentence the page shows is unit-testable.
 * Wording follows the agreed UI convention: titles, narrative and guidance in
 * Indonesian; status chips, technical terms and data labels in English.
 * Phases use PLN naming (A/B/C shown as R/S/T).
 */
import type {
  CanonicalRecordAnalysis,
  ElectricalMeasurements,
  EpisodeOtherRecorder,
  FaultEpisodeOut,
  IncidentHypothesis,
  IncidentOut,
  IncidentRecordOut,
  PhaseValues,
  PhysicalCauseRecordEntry,
  ProtectionOperation,
  ReconstructionOut,
  TimeAxisPlacement,
} from "../../api/client";

export type Tone = "fault" | "reclose" | "neutral" | "warning";

export interface Chip {
  label: string;
  tone: Tone;
}

export interface Tile {
  label: string;
  value: string;
  detail?: string;
  mono?: boolean;
}

export interface SequenceCard {
  kind: "fault" | "reclose" | "after";
  title: string;
  time: string | null;
  headline: string;
  bullets: string[];
  recordId: string | null;
  recordName: string | null;
  emphasis: boolean;
}

export interface SequenceConnector {
  kind: "dead_time" | "refault" | "gap";
  label: string;
  detail: string;
}

export type SequenceEntry =
  | { type: "card"; card: SequenceCard }
  | { type: "connector"; connector: SequenceConnector };

export interface AiReading {
  title: string;
  recordName: string;
  kind: "reading" | "no_dominant" | "skipped" | "unavailable";
  cause?: string;
  percent?: number;
  note?: string;
  candidates?: Array<{ cause: string; percent: number }>;
}

export interface CauseStory {
  status: Chip;
  headline: string;
  /** tone: "fault" for physical-contact patterns, "reclose" for transient ones, "warning" when undecided. */
  pattern: { title: string; strength: string; text: string; tone: Tone } | null;
  ai: AiReading[];
  footnote: string;
}

export interface ChecklistItem {
  id: string;
  title: string;
  detail: string;
  link?: { label: string; to: string };
}

export interface RecordRow {
  recordId: string;
  name: string;
  roleLabel: string;
  roleTone: Tone;
  roleSuffix: string;
  start: string;
  line: string;
  note: string;
}

export interface IncidentStory {
  chips: Chip[];
  headline: string;
  narrative: string;
  tiles: Tile[];
  sequenceMeta: string;
  sequence: SequenceEntry[];
  cause: CauseStory;
  checklist: ChecklistItem[];
  records: RecordRow[];
}

// --- formatting ---------------------------------------------------------------

const PLN_PHASE: Record<string, string> = { A: "R", B: "S", C: "T" };
const PHASE_ORDER = ["A", "B", "C"] as const;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"];
const CAUSE_NAME: Record<string, string> = {
  PETIR: "Petir",
  POHON: "Pohon",
  LAYANG: "Layang-layang",
  BENDA_ASING: "Benda asing",
  KONDUKTOR: "Konduktor",
  PERALATAN: "Peralatan",
  HEWAN: "Hewan",
};
const TRANSIENT_CAUSES = new Set(["PETIR", "LAYANG", "HEWAN"]);

/** PLN phase naming: ["B","C"] -> "S-T", a single phase -> "R-N". */
export function phaseLabel(phases: string[]): string {
  const names = PHASE_ORDER.filter((p) => phases.includes(p)).map((p) => PLN_PHASE[p]);
  if (names.length === 0) return "?";
  return names.length === 1 ? `${names[0]}-N` : names.join("-");
}

function poleLabel(phases: Iterable<string>): string {
  return PHASE_ORDER.filter((p) => [...phases].includes(p)).map((p) => PLN_PHASE[p]).join("-");
}

/** Indonesian number formatting: 5.73 -> "5,7". */
export function formatNumber(value: number, digits = 1): string {
  return value.toLocaleString("id-ID", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** Epoch ms of a naive ISO timestamp, read as written (no time-zone shift). */
export function isoToMs(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?/.exec(iso);
  if (!m) return null;
  const micros = Number((m[7] ?? "0").padEnd(6, "0").slice(0, 6));
  return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]) + micros / 1000;
}

/** "15:15:03,738" */
export function formatClock(ms: number): string {
  const total = Math.round(ms);
  const d = new Date(total);
  const pad = (n: number, width = 2) => String(n).padStart(width, "0");
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())},${pad(((total % 1000) + 1000) % 1000, 3)}`;
}

/** "21 Agu 2023" */
export function formatDate(ms: number): string {
  const d = new Date(Math.round(ms));
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

function formatCurrent(amps: number): string {
  return amps >= 1000 ? `±${formatNumber(amps / 1000)} kA` : `±${Math.round(amps / 10) * 10} A`;
}

function causeName(code: string | null | undefined, label?: string): string {
  if (code && CAUSE_NAME[code]) return CAUSE_NAME[code];
  return (label ?? code ?? "?").split(" / ")[0];
}

// --- per-record facts -----------------------------------------------------------

interface LineSelectionSnapshot {
  selected_line?: string;
  lines?: Array<{ line: string; state: string; breaker_open_throughout?: boolean }>;
}

export interface RecordFacts {
  record: IncidentRecordOut;
  name: string;
  /** First sample on the incident clock (epoch ms, read as written). */
  startAbs: number | null;
  /** How the record was placed on the incident time axis; null on older reconstructions. */
  placement: TimeAxisPlacement | null;
  eventClass: string | null;
  inceptionAxisMs: number | null;
  inceptionAbs: number | null;
  durationMs: number | null;
  /** Fault start to the last current zero, read from the waveforms (analog trace). */
  fctMs: number | null;
  clearingAxisMs: number | null;
  phases: string[];
  reclose: { axisMs: number; abs: number | null; success: boolean | null } | null;
  electrical: ElectricalMeasurements;
  ops: ProtectionOperation[];
  selectedLine: string | null;
  otherLines: Array<{ line: string; state: string; breakerOpen: boolean }>;
  station: string | null;
}

/**
 * "ZQ6D" for "ZQ6D.cfg+ZQ6D.dat" or "ZQ6D.cfg": the record's stem, as engineers
 * name it. A file named the IEEE C37.232 way ("230821,081503670,+7h0,GI
 * MOJOSONGO,BRINGIN 1-2,Qualitrol LLC.cfg") goes by its leading date and time.
 */
export function recordName(record: IncidentRecordOut): string {
  const stem = (record.source_filename ?? "").split(/\.(?:cfg|dat|cff)\b/i)[0].trim();
  const comname = /^(\d{6},\d{6,9}),[+-]?\d/.exec(stem);
  return (comname ? comname[1] : stem) || record.analysis_id.slice(0, 8);
}

/**
 * Facts of one record, with its times on the incident clock when the
 * reconstruction placed it there (another recorder's clock, e.g. a far-end DFR
 * stamping UTC, is lined up on the fault both ends recorded) — else on the
 * record's own clock.
 */
export function readFacts(record: IncidentRecordOut, placement: TimeAxisPlacement | null = null): RecordFacts {
  const snap = (record.canonical_snapshot ?? {}) as Partial<CanonicalRecordAnalysis>;
  const ew = snap.event_window ?? null;
  const startMs = isoToMs(placement?.start_iso ?? record.record_start_iso);
  const axis0 = ew?.record_start_ms ?? 0;
  const toAbs = (axisMs: number | null | undefined) =>
    startMs !== null && axisMs !== null && axisMs !== undefined ? startMs + (axisMs - axis0) : null;
  const deadTimeRecording = ew?.method === "dead_time_recording";
  const verified = (ew?.reclose_events ?? []).filter((e) => e.cb_open_verified !== false);
  const last = verified[verified.length - 1];
  const lineSelection = ((snap.provenance ?? {}) as { line_selection?: LineSelectionSnapshot | null }).line_selection ?? null;
  const selected = lineSelection?.selected_line ?? null;
  return {
    record,
    name: recordName(record),
    startAbs: startMs,
    placement,
    eventClass: ((snap.protection_interpretation ?? {}) as { event_class?: string }).event_class ?? null,
    inceptionAxisMs: deadTimeRecording ? null : (ew?.inception_time_ms ?? null),
    inceptionAbs: deadTimeRecording ? null : toAbs(ew?.inception_time_ms),
    durationMs: ew?.fault_duration_ms ?? null,
    fctMs: snap.analog_trace?.summary?.fct_ms ?? null,
    clearingAxisMs: ew?.clearing_time_ms ?? null,
    phases: ew?.faulted_phases ?? [],
    reclose:
      last && typeof last.time === "number"
        ? { axisMs: last.time * 1000, abs: toAbs(last.time * 1000), success: last.success ?? null }
        : null,
    electrical: snap.electrical_measurements ?? {},
    ops: Array.isArray(snap.protection_operations) ? snap.protection_operations : [],
    selectedLine: selected,
    otherLines: (lineSelection?.lines ?? [])
      .filter((l) => l.line !== selected)
      .map((l) => ({ line: l.line, state: l.state, breakerOpen: Boolean(l.breaker_open_throughout) })),
    station: record.station_name,
  };
}

function firstOn(op: ProtectionOperation): number | null {
  return op.initially_on ? null : (op.on_ms[0] ?? null);
}

function isReceive(op: ProtectionOperation): boolean {
  return op.role === "teleprotection" && /\b(CR|RCV|RECV|RECEIVE|RECEIVED|RX)\b/i.test(op.name.replace(/[^A-Za-z0-9]+/g, " "));
}

export interface TripReading {
  text: string;
  atMs: number;
  zones: number[];
  aided: boolean;
}

/**
 * The trip as the status channels recorded it, relative to inception:
 * "Trip 3-pole +45 ms · Z1", or "Trip pole R +37 ms · Z2 + carrier receive
 * (teleprotection-aided)" when a zone beyond Z1 tripped with a carrier
 * receive asserted (permissive scheme: the remote end saw the fault in its Z1).
 */
export function readTrip(facts: RecordFacts): TripReading | null {
  const t0 = facts.inceptionAxisMs;
  if (t0 === null) return null;
  const onAfterInception = (op: ProtectionOperation) => {
    const on = firstOn(op);
    return on !== null && on >= t0 - 20 ? on : null;
  };
  const trips = facts.ops.filter((op) => op.role === "trip" && onAfterInception(op) !== null);
  if (trips.length === 0) return null;
  const tripAt = Math.min(...trips.map((op) => onAfterInception(op) as number));
  const atTrip = (op: ProtectionOperation) => {
    const on = onAfterInception(op);
    return on !== null && on <= tripAt + 10;
  };
  const poles = new Set<string>();
  let threePole = false;
  for (const op of trips.filter(atTrip)) {
    if (op.phase === "3P") threePole = true;
    else if (op.phase) poles.add(op.phase);
  }
  if (poles.size === 3) threePole = true;
  const poleText = threePole ? " 3-pole" : poles.size > 1 ? ` poles ${poleLabel(poles)}` : poles.size === 1 ? ` pole ${poleLabel(poles)}` : "";
  const zones = [
    ...new Set(
      facts.ops
        .filter((op) => (op.role === "zone" || op.role === "trip") && op.zone !== null && atTrip(op))
        .map((op) => op.zone as number),
    ),
  ].sort((a, b) => a - b);
  const aided = facts.ops.some((op) => isReceive(op) && atTrip(op));
  const elements = zones.map((z) => `Z${z}`);
  if (aided) elements.push("carrier receive");
  let text = `Trip${poleText} +${Math.round(tripAt - t0)} ms${elements.length ? ` · ${elements.join(" + ")}` : ""}`;
  if (aided && zones.length > 0 && !zones.includes(1)) text += " (teleprotection-aided)";
  return { text, atMs: tripAt - t0, zones, aided };
}

function faultCurrentAmps(facts: RecordFacts): number | null {
  const rms: PhaseValues | undefined = facts.electrical.fault?.current_rms_max;
  if (!rms) return null;
  const phases = facts.phases.length > 0 ? facts.phases : [...PHASE_ORDER];
  const values = phases.map((p) => rms[p as "A" | "B" | "C"]).filter((v): v is number => typeof v === "number");
  if (values.length === 0) return null;
  const max = Math.max(...values);
  return (facts.electrical.current_unit ?? "A").toLowerCase() === "ka" ? max * 1000 : max;
}

function mean(values: PhaseValues | undefined): number | null {
  const list = Object.values(values ?? {}).filter((v): v is number => typeof v === "number");
  return list.length ? list.reduce((a, b) => a + b, 0) / list.length : null;
}

function afterRecloseBullet(facts: RecordFacts): string | null {
  const after = facts.electrical.after_reclose;
  if (!after) return null;
  const amps = mean(after.current_rms);
  let volts = mean(after.voltage_rms);
  if (volts !== null && (facts.electrical.voltage_unit ?? "kV").toLowerCase() === "v") volts /= 1000;
  const parts: string[] = [];
  if (volts !== null) parts.push(`Voltage ${Math.round(volts)} kV`);
  if (amps !== null) parts.push(`load ${formatCurrent(amps)}`);
  return parts.length ? parts.join(", ") : null;
}

/** "Single-pole reclose" / "3-pole reclose" when the auto-reclose channel says which. */
function recloseMode(facts: RecordFacts): string | null {
  const names = facts.ops.filter((op) => op.role === "reclose").map((op) => op.name.toUpperCase());
  if (names.some((n) => /\b1P\b|1-?POLE|SINGLE/.test(n))) return "Single-pole reclose";
  if (names.some((n) => /\b3P\b|3-?POLE|THREE/.test(n))) return "3-pole reclose";
  return null;
}

/** Dead time of a reclose captured inside the fault record itself (seconds). */
function singleRecordDeadTimeS(facts: RecordFacts): number | null {
  if (!facts.reclose || facts.inceptionAxisMs === null) return null;
  const t0 = facts.inceptionAxisMs;
  const breakerOpened = facts.ops
    .filter((op) => op.role === "breaker")
    .map(firstOn)
    .filter((on): on is number => on !== null && on >= t0 && on < (facts.reclose as { axisMs: number }).axisMs);
  const openedAt = breakerOpened.length ? Math.min(...breakerOpened) : facts.clearingAxisMs;
  return openedAt !== null ? (facts.reclose.axisMs - openedAt) / 1000 : null;
}

// --- story --------------------------------------------------------------------

const HEADLINES: Record<string, string> = {
  RECLOSE_THEN_REFAULT: "Gangguan berulang setelah reclose",
  SINGLE_TRANSIENT_FAULT: "Gangguan sementara, reclose berhasil",
  SINGLE_FAULT_FAILED_RECLOSE: "Gangguan dengan reclose gagal",
  SINGLE_FAULT_EPISODE: "Gangguan tunggal",
  REPEATED_FAULT_WITH_FINAL_FAILED_RECLOSE: "Gangguan berulang, reclose terakhir gagal",
  POSSIBLE_EVOLVING_FAULT_SEQUENCE: "Gangguan yang kemungkinan berkembang",
  REPEATED_INDEPENDENT_FAULTS: "Beberapa gangguan terpisah",
  TRANSIENT_FAULT_WITH_RECLOSE_SEQUENCE: "Gangguan sementara dengan urutan reclose",
  MULTIPLE_EPISODES_MIXED_RELATIONSHIP: "Beberapa gangguan berurutan",
  NO_EPISODES: "Belum ada gangguan yang terbaca",
};

const PATTERNS: Record<string, { headline: string; contact: boolean | null }> = {
  REFAULT_AFTER_SUCCESSFUL_RECLOSE: { headline: "Indikasi kontak fisik (pohon / benda asing)", contact: true },
  FAILED_RECLOSE_INDICATES_PERMANENT_FAULT: { headline: "Indikasi gangguan permanen", contact: true },
  ESCALATING_PHASE_INVOLVEMENT: { headline: "Indikasi kontak fisik yang meluas", contact: true },
  RECURRING_SAME_SIGNATURE: { headline: "Gangguan berulang dengan pola sama", contact: null },
  SINGLE_TRANSIENT_NO_RECURRENCE: { headline: "Gangguan sementara (transien)", contact: false },
  REPEATED_ESCALATING_SIGNATURE_AMBIGUOUS: { headline: "Petir atau kontak fisik — belum bisa dibedakan", contact: null },
  POSSIBLE_EVOLVING_FAULT: { headline: "Kemungkinan gangguan yang berkembang", contact: null },
};

const RELATION_DETAIL: Record<string, string> = {
  NEW_FAULT_EPISODE: "new fault",
  REPEATED_FAULT: "repeated fault",
  POSSIBLE_EVOLVING_FAULT: "possibly evolving fault",
  UNRELATED: "unrelated",
  UNCERTAIN: "relationship uncertain",
};

const LINE_STATE: Record<string, string> = {
  DE_ENERGIZED: "de-energized",
  QUIET: "not disturbed",
  IMPACTED: "impacted",
  BREAKER_OR_RECLOSE_ONLY: "CB/AR operation only",
  ALSO_OPERATED: "also operated",
};

interface EpisodeView {
  episode: FaultEpisodeOut;
  number: number;
  fault: RecordFacts | null;
  reclose: RecordFacts | null;
  members: RecordFacts[];
  /** Other recorders' view of the episode: the far line end, a second device in the bay. */
  others: EpisodeOtherRecorder[];
  trip: TripReading | null;
  deadTimeS: number | null;
  refaultAfterS: number | null;
}

function strengthOf(confidence: number | null): string {
  if (confidence === null) return "unscored";
  if (confidence >= 0.75) return "strong";
  if (confidence >= 0.5) return "medium";
  return "weak";
}

function patternText(h: IncidentHypothesis, views: EpisodeView[]): string {
  const [first, second] = views;
  const samePhases = Boolean(
    first && second && first.episode.faulted_phases.join() === second.episode.faulted_phases.join(),
  );
  switch (h.hypothesis) {
    case "REFAULT_AFTER_SUCCESSFUL_RECLOSE": {
      const s = second?.refaultAfterS ?? first?.refaultAfterS;
      const when = s !== null && s !== undefined ? `${formatNumber(s)} detik` : "beberapa detik";
      const where = samePhases
        ? `di fasa yang sama (${phaseLabel(first.episode.faulted_phases)})`
        : `di fasa ${phaseLabel(second?.episode.faulted_phases ?? [])}`;
      return (
        `Gangguan muncul lagi ${when} setelah reclose berhasil, ${where}. Artinya penyebab gangguan pertama ` +
        "masih ada saat line diberi tegangan lagi — khas kontak fisik, bukan sambaran petir tunggal."
      );
    }
    case "FAILED_RECLOSE_INDICATES_PERMANENT_FAULT":
      return (
        "Reclose gagal: gangguan masih ada saat CB menutup kembali. Penyebab yang menetap — pohon, benda asing " +
        "yang tersangkut, atau konduktor/peralatan rusak — lebih mungkin daripada sambaran petir."
      );
    case "ESCALATING_PHASE_INVOLVEMENT":
      return "Fasa yang terganggu bertambah dari kejadian ke kejadian — khas kontak fisik yang bergerak atau meluas.";
    case "RECURRING_SAME_SIGNATURE":
      return (
        "Gangguan berulang dengan fasa dan bentuk yang sama. Bisa kontak fisik yang hilang-timbul atau sambaran " +
        "berulang; pola ini sendiri belum bisa membedakan keduanya."
      );
    case "SINGLE_TRANSIENT_NO_RECURRENCE":
      return (
        "Satu gangguan singkat yang padam dan reclose berhasil tanpa gangguan ulang — pola khas sambaran petir " +
        "atau switching. Pola ini sendiri tidak membedakan petir dari layang-layang atau hewan."
      );
    case "REPEATED_ESCALATING_SIGNATURE_AMBIGUOUS":
      return "Pola urutan dan bacaan AI menunjuk ke arah berbeda; perlu bukti lapangan untuk memilih.";
    default:
      return h.description ?? "";
  }
}

function capNote(entry: PhysicalCauseRecordEntry, view: EpisodeView | undefined): string | undefined {
  const caps = (entry.applied_caps ?? []).filter((c) => c.name !== "ceiling_92" && c.after < c.before);
  if (caps.length === 0) return undefined;
  const cap = caps[caps.length - 1];
  const before = Math.round(cap.before * 100);
  if (cap.name === "reclose_outcome_conflict") {
    return view?.episode.reclose_outcome === "failed"
      ? `Turun dari ${before}% karena reclose gagal — tidak cocok dengan penyebab transien`
      : `Turun dari ${before}% karena gangguan berulang setelah reclose tidak cocok dengan penyebab transien`;
  }
  return `Turun dari ${before}% (${cap.name})`;
}

function aiReading(entry: PhysicalCauseRecordEntry, title: string, name: string, view: EpisodeView | undefined): AiReading {
  if (entry.skip_reason === "reclose_capture") {
    return { title, recordName: name, kind: "skipped", note: "Tidak dianalisa — rekaman reclose, tanpa gangguan" };
  }
  if (entry.skip_reason) {
    return { title, recordName: name, kind: "skipped", note: `Tidak dianalisa — ${entry.skip_reason.replace(/_/g, " ")}` };
  }
  const ranking = entry.cause_ranking ?? [];
  if (!entry.top_hypothesis || ranking.length === 0) {
    return { title, recordName: name, kind: "unavailable", note: "Bacaan AI tidak tersedia untuk rekaman ini" };
  }
  const [top, second] = ranking;
  if (top.confidence < 0.5 && second && top.confidence - second.confidence < 0.1) {
    return {
      title,
      recordName: name,
      kind: "no_dominant",
      candidates: ranking.slice(0, 3).map((c) => ({ cause: causeName(c.cause, c.label), percent: Math.round(c.confidence * 100) })),
    };
  }
  const note =
    entry.evidence_role === "aftermath"
      ? "Rekaman lanjutan — bukan bukti penyebab terpisah"
      : entry.evidence_role === "remote_end"
        ? "Rekaman ujung lain — bukan bukti penyebab terpisah"
        : capNote(entry, view);
  return {
    title,
    recordName: name,
    kind: "reading",
    cause: causeName(top.cause, top.label),
    percent: Math.round((entry.confidence ?? top.confidence) * 100),
    note,
  };
}

/** The episode's other recorders, from `observed_facts.other_recorders` (absent on older reconstructions). */
function episodeOtherRecorders(episode: FaultEpisodeOut): EpisodeOtherRecorder[] {
  const value = (episode.observed_facts as { other_recorders?: unknown }).other_recorders;
  return Array.isArray(value) ? (value as EpisodeOtherRecorder[]) : [];
}

function buildEpisodeViews(reconstruction: ReconstructionOut, factsById: Map<string, RecordFacts>): EpisodeView[] {
  const episodes = [...(reconstruction.episodes ?? [])].sort((a, b) => a.episode_index - b.episode_index);
  const relationships = reconstruction.relationships ?? [];
  return episodes.map((episode, i) => {
    const members = episode.member_record_ids.map((id) => factsById.get(id)).filter((f): f is RecordFacts => !!f);
    // The episode is read from its own end; other recorders' records (the far
    // line end) carry their own breaker's trip and reclose.
    const others = episodeOtherRecorders(episode);
    const otherIds = new Set(others.flatMap((o) => o.member_record_ids));
    const own = members.filter((f) => !otherIds.has(f.record.incident_record_id));
    const fault = own.find((f) => f.inceptionAxisMs !== null && f.eventClass !== "RECLOSE_CAPTURE") ?? own[0] ?? null;
    const capture = own.find((f) => f.eventClass === "RECLOSE_CAPTURE") ?? null;
    const reclose = capture ?? (fault?.reclose ? fault : null);
    const facts = episode.observed_facts as Record<string, number | undefined>;
    const relDeadTime = relationships.find(
      (r) =>
        r.relationship_type === "RECLOSE_SEQUENCE" &&
        own.some((f) => f.record.incident_record_id === r.right_record_id) &&
        !otherIds.has(r.left_record_id),
    )?.metrics?.dead_time_s as number | undefined;
    const deadTimeS =
      facts.reclose_dead_time_s ?? relDeadTime ?? (reclose && reclose === fault ? singleRecordDeadTimeS(fault) : null);
    return {
      episode,
      number: i + 1,
      fault,
      reclose: episode.reclose_outcome || capture ? reclose : null,
      members,
      others,
      trip: fault ? readTrip(fault) : null,
      deadTimeS: deadTimeS ?? null,
      refaultAfterS: facts.seconds_after_previous_reclose ?? null,
    };
  });
}

function lineName(incident: IncidentOut, views: EpisodeView[], records: RecordFacts[]): string {
  return (
    incident.bay_name ||
    views.find((v) => v.fault?.selectedLine)?.fault?.selectedLine ||
    records.find((r) => r.selectedLine)?.selectedLine ||
    incident.station_name ||
    records[0]?.station ||
    "—"
  );
}

function buildNarrative(line: string, views: EpisodeView[]): string {
  if (views.length === 0) return "Belum ada gangguan yang terbaca dari rekaman yang dilampirkan.";
  const sentences: string[] = [];
  views.forEach((view, i) => {
    const phases = phaseLabel(view.episode.faulted_phases);
    let clause: string;
    if (i === 0) {
      clause = `Line ${line} trip karena gangguan fasa ${phases}`;
    } else if (view.episode.relationship_to_previous === "REFAULT_AFTER_RECLOSE") {
      const prev = views[i - 1].episode.faulted_phases.join();
      const where = prev === view.episode.faulted_phases.join() ? "di fasa yang sama" : `di fasa ${phases}`;
      const when = view.refaultAfterS !== null ? `${formatNumber(view.refaultAfterS)} detik kemudian ` : "";
      clause = `lalu terganggu lagi ${when}${where} dan trip kembali`;
    } else {
      clause = `lalu terjadi gangguan baru di fasa ${phases}`;
    }
    if (view.episode.reclose_outcome === "successful") {
      const after = view.deadTimeS !== null ? ` setelah ${formatNumber(view.deadTimeS)} detik` : "";
      clause += `, berhasil reclose${after}`;
    } else if (view.episode.reclose_outcome === "failed") {
      clause += ", reclose gagal";
    }
    sentences.push(clause);
  });
  let text = `${sentences.join(", ")}.`;
  const last = views[views.length - 1].episode.reclose_outcome;
  if (last === null) text += " Tidak ada reclose berikutnya yang terekam.";
  else if (last === "successful") text += " Reclose bertahan sampai akhir rekaman.";
  return text;
}

function buildSequence(views: EpisodeView[]): SequenceEntry[] {
  const entries: SequenceEntry[] = [];
  views.forEach((view, i) => {
    const { episode, fault } = view;
    if (i > 0) {
      if (episode.relationship_to_previous === "REFAULT_AFTER_RECLOSE") {
        entries.push({
          type: "connector",
          connector: {
            kind: "refault",
            label: view.refaultAfterS !== null ? `${formatNumber(view.refaultAfterS)} s kemudian` : "Kemudian",
            detail: "reclose did not hold",
          },
        });
      } else {
        const prev = views[i - 1].fault?.inceptionAbs ?? null;
        const here = fault?.inceptionAbs ?? null;
        entries.push({
          type: "connector",
          connector: {
            kind: "gap",
            label: prev !== null && here !== null ? `${formatNumber((here - prev) / 1000)} s kemudian` : "Kemudian",
            detail: RELATION_DETAIL[episode.relationship_to_previous ?? "UNCERTAIN"] ?? "next fault",
          },
        });
      }
    }

    const bullets: string[] = [];
    if (view.trip) bullets.push(view.trip.text);
    // The waveform's last current zero; the episode duration is the trip
    // contact's pulse width, which is not the fault clearing time.
    const duration = fault?.fctMs ?? episode.duration_ms ?? fault?.durationMs ?? null;
    if (duration !== null) bullets.push(`Gangguan padam setelah ${Math.round(duration)} ms`);
    if (i > 0 && episode.relationship_to_previous === "REFAULT_AFTER_RECLOSE") {
      const prev = views[i - 1];
      bullets.push(
        prev.episode.faulted_phases.join() === episode.faulted_phases.join()
          ? `Fasa sama dengan gangguan #${prev.number}`
          : `Fasa berubah dari ${phaseLabel(prev.episode.faulted_phases)}`,
      );
    }
    for (const other of view.others) {
      if (other.same_station || !other.fault_record_id) continue;
      const fct = other.fct_ms !== null ? `, padam setelah ${Math.round(other.fct_ms)} ms` : "";
      bullets.push(`Ujung ${other.station}: fasa ${phaseLabel(other.faulted_phases)}${fct}`);
    }
    const amps = fault ? faultCurrentAmps(fault) : null;
    const time = fault?.inceptionAbs ?? isoToMs(episode.start_iso);
    entries.push({
      type: "card",
      card: {
        kind: "fault",
        title: `Gangguan #${view.number}`,
        time: time !== null ? formatClock(time) : null,
        headline: `Fasa ${phaseLabel(episode.faulted_phases)}${amps !== null ? ` · ${formatCurrent(amps)}` : ""}`,
        bullets,
        recordId: fault?.record.incident_record_id ?? null,
        recordName: fault?.name ?? null,
        emphasis: episode.relationship_to_previous === "REFAULT_AFTER_RECLOSE",
      },
    });

    if (view.reclose) {
      const success = episode.reclose_outcome !== "failed";
      if (view.deadTimeS !== null) {
        entries.push({
          type: "connector",
          connector: { kind: "dead_time", label: `Dead time ${formatNumber(view.deadTimeS)} s`, detail: "CB open" },
        });
      }
      const recloseBullets: string[] = [];
      const sameRecord = view.reclose === fault;
      const mode = recloseMode(view.reclose);
      if (mode) recloseBullets.push(mode);
      const power = afterRecloseBullet(view.reclose);
      if (power) recloseBullets.push(power);
      recloseBullets.push(success ? "No fault current" : "Gangguan masih ada saat CB menutup");
      for (const other of view.others) {
        if (other.same_station || !other.reclose_outcome) continue;
        const after = other.reclose_dead_time_s !== null ? ` setelah dead time ${formatNumber(other.reclose_dead_time_s)} s` : "";
        recloseBullets.push(`Ujung ${other.station}: reclose ${other.reclose_outcome === "successful" ? "berhasil" : "gagal"}${after}`);
      }
      const at = view.reclose.reclose?.abs ?? null;
      entries.push({
        type: "card",
        card: {
          kind: "reclose",
          title: success ? "Reclose successful" : "Reclose failed",
          time: at !== null ? formatClock(at) : null,
          // A reclose inside the fault record closes the tripped pole(s); a
          // separate capture shows the whole line coming back.
          headline: !success ? "Reclose ke gangguan" : sameRecord ? "CB menutup kembali" : "Line bertegangan kembali",
          bullets: recloseBullets,
          recordId: view.reclose.record.incident_record_id,
          recordName: view.reclose.name,
          emphasis: false,
        },
      });
    }
  });

  if (views.length > 0) {
    const last = views[views.length - 1].episode.reclose_outcome;
    const [headline, note] =
      last === "successful"
        ? ["Line kembali beroperasi", "Reclose bertahan sampai akhir rekaman."]
        : last === "failed"
          ? ["Reclose gagal — line trip lagi", "Kemungkinan gangguan permanen. Pastikan dengan log gardu."]
          : ["Tidak ada reclose berikutnya terekam", "Kemungkinan lockout. Pastikan dengan log gardu."];
    entries.push({
      type: "card",
      card: { kind: "after", title: "Sesudahnya", time: null, headline, bullets: [note], recordId: null, recordName: null, emphasis: false },
    });
  }
  return entries;
}

function buildCause(reconstruction: ReconstructionOut, views: EpisodeView[], factsById: Map<string, RecordFacts>): CauseStory {
  const evidence = reconstruction.physical_cause_evidence;
  const viewOf = (recordId: string) => views.find((v) => v.episode.member_record_ids.includes(recordId));
  const ai = (evidence?.records ?? []).map((entry) => {
    const view = viewOf(entry.incident_record_id);
    const facts = factsById.get(entry.incident_record_id);
    const title =
      entry.evidence_role === "remote_end"
        ? `Ujung ${facts?.station ?? "lain"}`
        : facts?.eventClass === "RECLOSE_CAPTURE"
          ? "Reclose"
          : view && view.fault?.record.incident_record_id === entry.incident_record_id
            ? `Gangguan #${view.number}`
            : "Rekaman lanjutan";
    return aiReading(entry, title, facts?.name ?? entry.analysis_id.slice(0, 8), view);
  });

  const signals = (reconstruction.incident_hypotheses ?? []).filter((h) => PATTERNS[h.hypothesis]);
  signals.sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  const signal = signals[0] ?? null;
  const contact = signal ? PATTERNS[signal.hypothesis].contact : null;
  const pattern = signal
    ? {
        title: "Dari urutan kejadian",
        strength: `Pattern evidence · ${strengthOf(signal.confidence)}`,
        text: patternText(signal, views),
        tone: (contact === true ? "fault" : contact === false ? "reclose" : "warning") as Tone,
      }
    : null;

  const topAi = (evidence?.records ?? []).find((e) => e.evidence_role === "inception" && e.top_hypothesis);
  let headline: string;
  if (signal) headline = PATTERNS[signal.hypothesis].headline;
  else if (topAi) headline = `Bacaan AI: ${causeName(topAi.top_hypothesis, topAi.cause_ranking?.[0]?.label)}`;
  else headline = "Penyebab belum terbaca";

  let footnote = "AI membaca tiap rekaman sendiri-sendiri. Penyebab perlu dikonfirmasi di lapangan.";
  if (signal && topAi?.top_hypothesis) {
    const aiTransient = TRANSIENT_CAUSES.has(topAi.top_hypothesis);
    if (contact !== null && contact === aiTransient) {
      footnote =
        "AI membaca tiap rekaman sendiri-sendiri, jadi tidak melihat pola urutan kejadian. Pola urutan dan bacaan " +
        "AI belum sepakat — penyebab perlu dikonfirmasi di lapangan.";
    } else if (contact !== null) {
      footnote = "Pola urutan dan bacaan AI searah, tetapi penyebab tetap perlu dikonfirmasi di lapangan.";
    }
  }

  const confirmed = evidence?.incident_root_cause && evidence.incident_root_cause !== "UNCONFIRMED";
  return {
    status: confirmed
      ? { label: `Confirmed: ${causeName(evidence.incident_root_cause)}`, tone: "reclose" }
      : { label: "Unconfirmed", tone: "neutral" },
    headline,
    pattern,
    ai,
    footnote,
  };
}

function buildChecklist(
  incident: IncidentOut,
  reconstruction: ReconstructionOut,
  views: EpisodeView[],
  pattern: { contact: boolean | null } | null,
): ChecklistItem[] {
  const items: ChecklistItem[] = [];
  const first = views[0];
  const missing = new Set(incident.missing_evidence.map((m) => m.type));
  if (first && pattern?.contact) {
    items.push({
      id: "row",
      title: "Inspeksi ROW di sekitar titik gangguan",
      detail: `Pohon atau vegetasi dekat konduktor fasa ${phaseLabel(first.episode.faulted_phases)}`,
    });
  }
  const firstTime = first?.fault?.inceptionAbs ?? null;
  if (firstTime !== null) {
    const hhmm = formatClock(firstTime).slice(0, 5);
    items.push({
      id: "lds",
      title: "Data sambaran petir (LDS)",
      detail: `${formatDate(firstTime)} sekitar ${hhmm} (jam DFR) di koridor line`,
    });
  }
  const aided = views.find((v) => v.trip?.aided && v.trip.zones.length > 0 && !v.trip.zones.includes(1));
  if (aided) {
    items.push({
      id: "remote-z1",
      title: "Konfirmasi Zone 1 di GI lawan",
      detail:
        `Trip gangguan #${aided.number} dipercepat teleproteksi (Z${aided.trip!.zones.join("/Z")} + carrier receive): ` +
        "gangguan kemungkinan dekat GI lawan. Rekaman GI lawan menunjukkan Z1 dan sinyal send-nya.",
    });
  }
  const hasRemote =
    incident.records.some((r) => r.attachment_role === "REMOTE_END") ||
    views.some((v) => v.others.some((o) => !o.same_station));
  items.push({
    id: "two-ended",
    title: "Lokasi gangguan dari dua ujung",
    detail: hasRemote
      ? "Rekaman GI lawan sudah dilampirkan — hitung lokasi gangguan."
      : "Tambahkan rekaman GI lawan untuk menghitung lokasi gangguan dari dua ujung.",
    link: { label: "Hitung lokasi (Double Ended FL)", to: "/de-fl" },
  });
  if (reconstruction.same_bay_status === "MISMATCH_REQUIRES_REVIEW") {
    items.push({
      id: "same-bay",
      title: "Pastikan semua rekaman dari bay yang sama",
      detail: "Nama GI/bay di rekaman tidak sama — urutan kejadian bisa tercampur.",
    });
  }
  if (missing.has("RECORDS_MISSING_ABSOLUTE_TIME")) {
    items.push({
      id: "order",
      title: "Periksa urutan rekaman",
      detail: "Ada rekaman tanpa waktu absolut, jadi urutannya mengikuti urutan upload.",
    });
  }
  if (views.some((v) => v.episode.missing_evidence.some((m) => m.type === "NO_PROTECTION_OPERATION"))) {
    items.push({
      id: "no-protection",
      title: "Cek operasi proteksi",
      detail: "Gangguan terlihat di waveform, tetapi tidak ada kanal trip yang aktif di rekaman.",
    });
  }
  if (views.some((v) => v.episode.missing_evidence.some((m) => m.type === "ENDS_DISAGREE_ON_FAULTED_PHASES"))) {
    items.push({
      id: "ends-phases",
      title: "Periksa fasa gangguan di kedua ujung",
      detail: "Kedua ujung line membaca fasa gangguan yang berbeda — salah satu bacaan perlu dicek (sering di ujung weak infeed).",
    });
  }
  if (missing.has("BAY_NAME_UNDETERMINED")) {
    items.push({ id: "bay", title: "Lengkapi nama bay", detail: "Belum diisi pada insiden ini" });
  }
  return items;
}

function buildRecordRows(records: RecordFacts[], views: EpisodeView[]): RecordRow[] {
  return records.map((facts) => {
    const id = facts.record.incident_record_id;
    const view = views.find((v) => v.episode.member_record_ids.includes(id));
    let roleLabel = "Tidak terkait";
    let roleTone: Tone = "neutral";
    let roleSuffix = "";
    let note = "";
    if (facts.eventClass === "RECLOSE_CAPTURE") {
      const farEnd = view?.others.some((o) => !o.same_station && o.member_record_ids.includes(id));
      roleLabel = "Reclose";
      roleTone = "reclose";
      roleSuffix = farEnd ? " (ujung lain, mulai saat dead time)" : " (mulai saat dead time)";
      note = "Tidak ada gangguan di rekaman ini";
    } else if (view && view.fault?.record.incident_record_id === id) {
      roleLabel = `Gangguan #${view.number}`;
      roleTone = "fault";
      roleSuffix = view.trip ? " + trip" : "";
      if (view.episode.relationship_to_previous === "REFAULT_AFTER_RECLOSE" && view.refaultAfterS !== null) {
        note = `${formatNumber(view.refaultAfterS)} s setelah reclose`;
      }
    } else if (view) {
      const other = view.others.find((o) => o.member_record_ids.includes(id));
      roleLabel = `Gangguan #${view.number}`;
      roleSuffix = other ? (other.same_station ? " (perekam lain)" : " (ujung lain)") : " (rekaman lanjutan)";
    }
    const others = facts.otherLines
      .map((l) => `${l.line} ${LINE_STATE[l.state] ?? l.state.toLowerCase()}${l.breakerOpen ? " (CB open)" : ""} — diabaikan`)
      .join("; ");
    if (others) note = note ? `${note}; ${others}` : others;
    const clock = clockNote(facts.placement);
    if (clock) note = note ? `${note}; ${clock}` : clock;
    const start = facts.startAbs;
    return {
      recordId: id,
      name: facts.name,
      roleLabel,
      roleTone,
      roleSuffix,
      start: start !== null ? formatClock(start) : "—",
      line: facts.selectedLine ?? "—",
      note,
    };
  });
}

/** How a record from another recorder was put on the incident clock, when that needed a correction. */
function clockNote(placement: TimeAxisPlacement | null): string {
  if (!placement) return "";
  if (placement.method === "own_clock_unverified") return "Jam perekam belum terverifikasi terhadap jam acuan";
  if (placement.method !== "fault_aligned") return "";
  const parts: string[] = [];
  if (placement.zone_offset_h) {
    const hours = placement.zone_offset_h;
    parts.push(`${hours > 0 ? "+" : "−"}${formatNumber(Math.abs(hours), Number.isInteger(hours) ? 0 : 2)} jam`);
  }
  if (placement.clock_offset_ms !== null && Math.abs(placement.clock_offset_ms) >= 1) {
    const ms = placement.clock_offset_ms;
    parts.push(`${ms > 0 ? "+" : "−"}${formatNumber(Math.abs(ms), 0)} ms`);
  }
  return parts.length
    ? `Jam perekam ${parts.join(" ")} dari jam acuan — diselaraskan pada awal gangguan`
    : "Diselaraskan pada awal gangguan";
}

function orderedRecords(incident: IncidentOut, reconstruction: ReconstructionOut): IncidentRecordOut[] {
  const order = reconstruction.alignment?.record_order ?? [];
  const byId = new Map(incident.records.map((r) => [r.incident_record_id, r]));
  const ordered = order.map((id) => byId.get(id)).filter((r): r is IncidentRecordOut => !!r);
  for (const r of incident.records) if (!ordered.includes(r)) ordered.push(r);
  return ordered;
}

/** True when the reconstruction was built from a different set of records than the incident has now. */
export function isReconstructionStale(incident: IncidentOut, reconstruction: ReconstructionOut | null): boolean {
  if (!reconstruction) return incident.records.length > 0;
  const built = new Set(
    (reconstruction.record_snapshot_versions ?? [])
      .map((v) => (v as { analysis_id?: string }).analysis_id)
      .filter((id): id is string => !!id),
  );
  const current = incident.records.filter((r) => r.inclusion_status !== "EXCLUDED").map((r) => r.analysis_id);
  if (built.size === 0) return reconstruction.observed_incident_facts.record_count !== current.length;
  return built.size !== current.length || current.some((id) => !built.has(id));
}

export function buildIncidentStory(incident: IncidentOut, reconstruction: ReconstructionOut): IncidentStory {
  const placements = new Map(
    (reconstruction.alignment?.time_axis?.records ?? []).map((p) => [p.incident_record_id, p]),
  );
  const records = orderedRecords(incident, reconstruction).map((r) => readFacts(r, placements.get(r.incident_record_id) ?? null));
  const factsById = new Map(records.map((f) => [f.record.incident_record_id, f]));
  const views = buildEpisodeViews(reconstruction, factsById);
  const line = lineName(incident, views, records);

  const eventClass = reconstruction.protection_sequence_interpretation?.event_class ?? "";
  const last = views[views.length - 1];
  const chips: Chip[] = [];
  if (views.some((v) => v.episode.relationship_to_previous === "REFAULT_AFTER_RECLOSE")) {
    chips.push({ label: "Reclose did not hold", tone: "fault" });
  } else if (views.some((v) => v.episode.reclose_outcome === "failed")) {
    chips.push({ label: "Reclose failed", tone: "fault" });
  } else if (views.some((v) => v.episode.reclose_outcome === "successful")) {
    chips.push({ label: "Reclose successful", tone: "reclose" });
  }
  if (last) {
    chips.push({
      label:
        last.episode.reclose_outcome === "successful"
          ? "Final state: line in service"
          : last.episode.reclose_outcome === "failed"
            ? "Final state: trip after failed reclose"
            : "Final state: trip — no further reclose",
      tone: "neutral",
    });
  }

  const tiles: Tile[] = [];
  const firstTime = views[0]?.fault?.inceptionAbs ?? null;
  if (firstTime !== null) tiles.push({ label: "First fault", value: formatClock(firstTime), mono: true });
  const peak = Math.max(...views.map((v) => (v.fault ? (faultCurrentAmps(v.fault) ?? 0) : 0)), 0);
  if (peak > 0) tiles.push({ label: "Fault current", value: formatCurrent(peak), mono: true });
  const deadTime = views.find((v) => v.deadTimeS !== null)?.deadTimeS ?? null;
  if (deadTime !== null) tiles.push({ label: "Dead time", value: `${formatNumber(deadTime)} s`, mono: true });
  const refault = views.find((v) => v.refaultAfterS !== null)?.refaultAfterS ?? null;
  if (refault !== null) tiles.push({ label: "Re-fault after reclose", value: `+${formatNumber(refault)} s`, mono: true });
  const selected = views.find((v) => v.fault?.selectedLine)?.fault?.selectedLine ?? null;
  tiles.push({ label: "Analysed line", value: incident.bay_name || selected || line, detail: incident.bay_name && selected ? selected : undefined });

  const cause = buildCause(reconstruction, views, factsById);
  const signal = (reconstruction.incident_hypotheses ?? [])
    .filter((h) => PATTERNS[h.hypothesis])
    .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))[0];
  const station = records.find((r) => r.station)?.station ?? incident.station_name;

  return {
    chips,
    headline: HEADLINES[eventClass] ?? "Urutan gangguan",
    narrative: buildNarrative(line, views),
    tiles,
    sequenceMeta: `Dari ${records.length} rekaman${station ? ` · waktu menurut jam DFR ${station}` : ""}`,
    sequence: buildSequence(views),
    cause,
    checklist: buildChecklist(incident, reconstruction, views, signal ? PATTERNS[signal.hypothesis] : null),
    records: buildRecordRows(records, views),
  };
}
