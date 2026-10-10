import { useState } from "react";
import { Link } from "react-router-dom";
import {
  overrideRelationship,
  type CanonicalRecordAnalysis,
  type EpisodeReasoning,
  type FaultEpisodeOut,
  type IncidentOut,
  type IncidentRecordOut,
  type ReasoningRow,
  type ReconstructionOut,
  type RecordReasoning,
  type RecordRelationshipOut,
  type RelationshipType,
} from "../../api/client";
import { recordName, type IncidentStory } from "./incidentStory";
import styles from "./TechnicalDetails.module.css";

type Tab = "reasoning" | "soe" | "records" | "meta";

const TABS: Array<[Tab, string]> = [
  ["reasoning", "Penalaran"],
  ["soe", "Urutan sinyal"],
  ["records", "Rekaman"],
  ["meta", "Data insiden"],
];

const CONFIDENCE: Record<ReasoningRow["confidence"], { label: string; className: string }> = {
  high: { label: "Tinggi", className: styles.confHigh },
  medium: { label: "Sedang", className: styles.confMedium },
  low: { label: "Rendah", className: styles.confLow },
  ai: { label: "Bacaan AI", className: styles.confAi },
  flag: { label: "Perlu dicek", className: styles.confFlag },
};

const ROLE_CLASS: Record<string, string> = {
  Gelombang: styles.roleWave,
  Zona: styles.roleZone,
  Trip: styles.roleTrip,
  Teleproteksi: styles.roleTele,
  Reclose: styles.roleReclose,
  PMT: styles.roleBreaker,
};

const RELATIONSHIP_LABEL: Record<RelationshipType, string> = {
  DUPLICATE_TRIGGER: "Duplicate capture",
  OVERLAPPING_CAPTURE: "Overlapping capture",
  REMOTE_END_CAPTURE: "Other line end",
  CONTINUATION: "Continuation",
  RECLOSE_SEQUENCE: "Reclose sequence",
  REFAULT_AFTER_RECLOSE: "Re-fault after reclose",
  NEW_FAULT_EPISODE: "New fault",
  REPEATED_FAULT: "Repeated fault",
  POSSIBLE_EVOLVING_FAULT: "Possibly evolving fault",
  UNRELATED: "Unrelated",
  UNCERTAIN: "Uncertain",
};

interface Props {
  incident: IncidentOut;
  reconstruction: ReconstructionOut | null;
  episodes: FaultEpisodeOut[];
  relationships: RecordRelationshipOut[];
  story: IncidentStory | null;
  refreshing: boolean;
  onHide: () => void;
  onAddRecords: () => void;
  onDetach: (incidentRecordId: string) => void;
  onSave: (fields: Record<string, unknown>) => Promise<void>;
  onArchive: () => void;
  onRefreshAnalyses: () => void;
  onRelationshipChanged: () => void;
}

function snapshot(record: IncidentRecordOut): Partial<CanonicalRecordAnalysis> {
  return (record.canonical_snapshot ?? {}) as Partial<CanonicalRecordAnalysis>;
}

function formatMs(value: number): string {
  const text = Math.abs(value).toLocaleString("id-ID", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  return value < 0 ? `−${text}` : value === 0 ? text : `+${text}`;
}

export default function TechnicalDetails(props: Props) {
  const [tab, setTab] = useState<Tab>("reasoning");

  function download() {
    const blob = new Blob([JSON.stringify({ incident: props.incident, reconstruction: props.reconstruction }, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `insiden-${props.incident.incident_id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.head}>
        <div className={styles.headText}>
          <h2 className={styles.title}>Detail teknis</h2>
          <p className={styles.muted}>Bukti dan aturan di balik setiap kesimpulan di atas.</p>
        </div>
        <div className={styles.headActions}>
          <button type="button" className={styles.button} onClick={download}>
            Unduh data (JSON)
          </button>
          <button type="button" className={styles.button} onClick={props.onHide}>
            Sembunyikan
          </button>
        </div>
      </div>

      <div role="group" aria-label="Bagian detail teknis" className={styles.tabs}>
        {TABS.map(([id, label]) => (
          <button
            key={id}
            type="button"
            aria-pressed={tab === id}
            className={tab === id ? styles.tabOn : styles.tab}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "reasoning" && <ReasoningTab {...props} />}
      {tab === "soe" && <SignalsTab {...props} />}
      {tab === "records" && <RecordsTab {...props} />}
      {tab === "meta" && <MetaTab {...props} />}
    </div>
  );
}

// --- Penalaran --------------------------------------------------------------------------------

function ReasoningTab({ episodes, story, refreshing, onAddRecords, onRefreshAnalyses }: Props) {
  const faults = story?.faults ?? episodes.map((e, i) => ({ episodeId: e.episode_id, number: i + 1, time: null }));
  const [picked, setPicked] = useState(0);
  const fault = faults[Math.min(picked, Math.max(0, faults.length - 1))];
  const ledger: EpisodeReasoning | undefined = episodes.find((e) => e.episode_id === fault?.episodeId)?.interpretation
    .reasoning;

  if (faults.length === 0) {
    return <p className={styles.muted}>Belum ada gangguan yang tersusun dari rekaman insiden ini.</p>;
  }
  const rows = ledger?.rows ?? [];
  const conclusions = rows.filter((r) => r.confidence !== "flag").length;
  const hasOtherEnd = rows.some((r) => r.key === "other_end");

  return (
    <div className={styles.stack}>
      <div className={styles.toolbar}>
        {faults.map((f, i) => (
          <button
            key={f.episodeId}
            type="button"
            aria-pressed={f === fault}
            className={f === fault ? styles.pillOn : styles.pill}
            onClick={() => setPicked(i)}
          >
            Gangguan #{f.number}
            {f.time ? ` · ${f.time}` : ""}
          </button>
        ))}
        <span className={styles.muted}>
          {story?.clockLabel ?? "jam DFR"}
          {faults.length === 1 ? " · satu-satunya gangguan di insiden ini" : ""}
        </span>
        <span className={styles.spacer} />
        {ledger && (
          <>
            <span className={styles.count}>{conclusions} kesimpulan</span>
            <span className={ledger.flag_count ? styles.countWarn : styles.count}>{ledger.flag_count} ditandai</span>
            <span className={ledger.conflict_count ? styles.countWarn : styles.count}>{ledger.conflict_count} konflik</span>
            {/* A ledger keeps the rules of the day it was built; this reads the
                same records again with the current ones. */}
            <button
              type="button"
              className={styles.button}
              disabled={refreshing}
              title="Analisa ulang rekaman dengan aturan terbaru, lalu susun ulang urutan kejadian"
              onClick={onRefreshAnalyses}
            >
              {refreshing ? "Memuat ulang…" : "Muat ulang analisa"}
            </button>
          </>
        )}
      </div>

      {!ledger ? (
        <div className={styles.callout}>
          <div className={styles.calloutText}>
            <span className={styles.strong}>Penalaran belum tersedia untuk gangguan ini</span>
            <span className={styles.muted}>
              Rekamannya dianalisa sebelum rantai aturan ada. Muat ulang analisanya untuk menyusun penalaran dari
              rekaman yang sama.
            </span>
          </div>
          <button type="button" className={styles.primary} disabled={refreshing} onClick={onRefreshAnalyses}>
            {refreshing ? "Memuat ulang…" : "Muat ulang analisa rekaman"}
          </button>
        </div>
      ) : (
        <div className={styles.table}>
          <div className={`${styles.ledgerRow} ${styles.tableHead}`}>
            <span>Langkah</span>
            <span>Kesimpulan dan bukti</span>
            <span>Aturan</span>
            <span>Keyakinan</span>
          </div>
          {rows.map((row, i) => (
            <div key={`${row.key}-${i}`} className={`${styles.ledgerRow} ${row.confidence === "flag" ? styles.flagRow : ""}`}>
              <span className={styles.step}>{row.label}</span>
              <div className={styles.cell}>
                <span className={styles.rowTitle}>{row.title}</span>
                {row.evidence.map((line, j) => (
                  <span key={j} className={styles.evidence}>
                    {line}
                  </span>
                ))}
                {row.conflicts.map((line, j) => (
                  <span key={`c${j}`} className={styles.conflict}>
                    {line}
                  </span>
                ))}
              </div>
              <div className={styles.rules}>
                {row.rules.map((rule) => (
                  <span key={rule} className={styles.rule}>
                    {rule}
                  </span>
                ))}
              </div>
              <span className={`${styles.conf} ${CONFIDENCE[row.confidence]?.className ?? ""}`}>
                {CONFIDENCE[row.confidence]?.label ?? row.confidence}
              </span>
            </div>
          ))}
        </div>
      )}

      {ledger && !hasOtherEnd && (
        <div className={styles.callout}>
          <div className={styles.calloutText}>
            <span className={styles.strong}>Skema dan lokasi baru terkonfirmasi dari sisi lawan</span>
            <span className={styles.muted}>
              Tambahkan rekaman GI lawan: zona dan sinyal teleproteksi di sana menguatkan pembacaan skema, dan dua
              rekaman membuka lokasi dua ujung.
            </span>
          </div>
          <button type="button" className={styles.primary} onClick={onAddRecords}>
            Tambah rekaman
          </button>
        </div>
      )}
    </div>
  );
}

// --- Urutan sinyal -------------------------------------------------------------------------------

function SignalsTab({ incident, episodes, story }: Props) {
  const withSignals = incident.records.filter((r) => snapshot(r).reasoning?.signals);
  // The first fault's record until the user picks another one.
  const defaultId =
    episodes.map((e) => e.interpretation.reasoning?.fault_record_id).find((id) => id) ?? withSignals[0]?.incident_record_id;
  const [pickedId, setRecordId] = useState<string | null>(null);
  const [all, setAll] = useState(false);
  const recordId = pickedId ?? defaultId;

  const record = withSignals.find((r) => r.incident_record_id === recordId) ?? withSignals[0];
  const reasoning: RecordReasoning | undefined = record ? snapshot(record).reasoning : undefined;
  const signals = reasoning?.signals;
  const changed = new Map<string, string>();
  for (const ev of signals?.events ?? []) if (ev.role !== "Gelombang") changed.set(ev.channel, ev.role);

  if (!signals || !record) {
    return (
      <p className={styles.muted}>
        Urutan sinyal belum tersedia: rekaman dianalisa sebelum fitur ini ada. Muat ulang analisa rekaman dari tab
        Penalaran.
      </p>
    );
  }
  const fault = story?.faults.find((f) =>
    episodes.find((e) => e.episode_id === f.episodeId)?.interpretation.reasoning?.fault_record_id === record.incident_record_id,
  );
  const reference =
    signals.reference === "fault_start"
      ? `Waktu dihitung dari awal gangguan${fault?.time ? ` (${fault.time} ${story?.clockLabel ?? "jam DFR"})` : ""}.`
      : "Waktu dihitung dari awal rekaman.";

  return (
    <div className={styles.stack}>
      {withSignals.length > 1 && (
        <div className={styles.toolbar}>
          {withSignals.map((r) => (
            <button
              key={r.incident_record_id}
              type="button"
              aria-pressed={r === record}
              className={r === record ? styles.pillOn : styles.pill}
              onClick={() => setRecordId(r.incident_record_id)}
            >
              {recordName(r)}
            </button>
          ))}
        </div>
      )}
      <div className={styles.toolbar}>
        <button type="button" aria-pressed={!all} className={!all ? styles.pillOn : styles.pill} onClick={() => setAll(false)}>
          Kanal yang berubah ({changed.size})
        </button>
        <button type="button" aria-pressed={all} className={all ? styles.pillOn : styles.pill} onClick={() => setAll(true)}>
          Semua kanal ({signals.channel_count})
        </button>
        <span className={styles.muted}>{reference}</span>
      </div>

      {!all ? (
        <>
          <div className={styles.table}>
            <div className={`${styles.soeRow} ${styles.tableHead}`}>
              <span>Waktu (ms)</span>
              <span>Kanal</span>
              <span>Peran</span>
              <span>Perubahan</span>
            </div>
            {signals.events.map((ev, i) => (
              <div
                key={i}
                className={`${styles.soeRow} ${ev.role === "Gelombang" ? styles.waveRow : ""} ${ev.muted ? styles.mutedRow : ""}`}
              >
                <span className={styles.time}>{formatMs(ev.t_ms)}</span>
                <span className={styles.channel}>{ev.channel}</span>
                <span className={`${styles.role} ${ROLE_CLASS[ev.role] ?? styles.roleOther}`}>{ev.role}</span>
                <span>{ev.change}</span>
              </div>
            ))}
          </div>
          <div className={styles.box}>
            <span className={styles.strong}>Direkam, tidak pernah aktif</span>
            <span className={styles.muted}>
              Kanal yang ada di rekaman tapi tetap 0 juga bukti: misalnya kanal Send yang tidak aktif saat Z2 pickup
              berarti skemanya bukan POTT.
            </span>
            <div className={styles.chips}>
              {signals.silent.map((s) => (
                <span key={s.channel} className={styles.chipMono}>
                  {s.channel}
                </span>
              ))}
            </div>
          </div>
        </>
      ) : (
        <div className={styles.table}>
          <div className={`${styles.channelRow} ${styles.tableHead}`}>
            <span>Kanal</span>
            <span>Peran</span>
            <span>Status</span>
          </div>
          {[...changed.entries()].map(([channel, role]) => (
            <div key={channel} className={styles.channelRow}>
              <span className={styles.channel}>{channel}</span>
              <span className={`${styles.role} ${ROLE_CLASS[role] ?? styles.roleOther}`}>{role}</span>
              <span>Berubah</span>
            </div>
          ))}
          {signals.silent.map((s) => (
            <div key={s.channel} className={`${styles.channelRow} ${styles.mutedRow}`}>
              <span className={styles.channel}>{s.channel}</span>
              <span className={`${styles.role} ${ROLE_CLASS[s.role ?? ""] ?? styles.roleOther}`}>{s.role ?? "Lainnya"}</span>
              <span>Tidak pernah aktif</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// --- Rekaman ---------------------------------------------------------------------------------------

function RecordsTab({ incident, relationships, story, onDetach, onRelationshipChanged }: Props) {
  const rows = new Map((story?.records ?? []).map((r) => [r.recordId, r]));
  const names = new Map(incident.records.map((r) => [r.incident_record_id, recordName(r)]));
  return (
    <div className={styles.stack}>
      <div className={styles.tableScroll}>
        <div className={styles.table} style={{ minWidth: 760 }}>
          <div className={`${styles.recRow} ${styles.tableHead}`}>
            <span>#</span>
            <span>Rekaman</span>
            <span>Line dianalisa</span>
            <span>Peran</span>
            <span>Mulai</span>
            <span>Aksi</span>
          </div>
          {incident.records.map((record, i) => {
            const row = rows.get(record.incident_record_id);
            const quality = (snapshot(record).data_quality ?? {}) as { analog_channel_count?: number; status_channel_count?: number };
            const detail = [
              record.station_name,
              quality.analog_channel_count !== undefined ? `${quality.analog_channel_count} analog` : null,
              quality.status_channel_count !== undefined ? `${quality.status_channel_count} digital` : null,
            ].filter(Boolean);
            return (
              <div key={record.incident_record_id} className={styles.recRow}>
                <span className={styles.muted}>{i + 1}</span>
                <div className={styles.cell}>
                  <span className={styles.strong}>{recordName(record)}</span>
                  <span className={styles.muted}>{detail.join(" · ")}</span>
                </div>
                <span>{row?.line ?? "—"}</span>
                <span>{row ? `${row.roleLabel}${row.roleSuffix}` : "—"}</span>
                <span className={styles.time}>{row?.start ?? "—"}</span>
                <div className={styles.actions}>
                  <Link
                    className={styles.smallButton}
                    to={`/workspace/${(record.protection_type || "21").toUpperCase()}/${record.analysis_id}`}
                  >
                    Buka analisa
                  </Link>
                  <button type="button" className={styles.smallDanger} onClick={() => onDetach(record.incident_record_id)}>
                    Lepas
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className={styles.box}>
        <span className={styles.strong}>Hubungan antar rekaman</span>
        {relationships.length === 0 ? (
          <span className={styles.muted}>
            Belum ada. Hubungan (reclose, re-fault, sisi lawan) muncul di sini begitu ada lebih dari satu rekaman, dan
            setiap hubungan bisa dikoreksi.
          </span>
        ) : (
          <div className={styles.relationList}>
            {relationships.map((rel) => (
              <RelationshipItem
                key={rel.relationship_id}
                incidentId={incident.incident_id}
                relationship={rel}
                left={names.get(rel.left_record_id) ?? rel.left_record_id.slice(0, 8)}
                right={names.get(rel.right_record_id) ?? rel.right_record_id.slice(0, 8)}
                onChanged={onRelationshipChanged}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function relationshipFacts(rel: RecordRelationshipOut): string {
  const m = rel.metrics as Record<string, unknown>;
  const facts: string[] = [];
  const num = (v: unknown, digits = 1) =>
    typeof v === "number" ? v.toLocaleString("id-ID", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : null;
  if (typeof m.dead_time_s === "number") facts.push(`dead time ${num(m.dead_time_s)} s`);
  if (typeof m.seconds_after_reclose === "number") facts.push(`${num(m.seconds_after_reclose)} s setelah reclose`);
  if (typeof m.fault_start_difference_ms === "number") facts.push(`awal gangguan selisih ${num(m.fault_start_difference_ms)} ms`);
  else if (typeof m.gap_seconds === "number" && rel.relationship_type !== "REMOTE_END_CAPTURE") {
    facts.push(`jarak trigger ${num(m.gap_seconds)} s`);
  }
  return facts.join(" · ");
}

function RelationshipItem({
  incidentId,
  relationship,
  left,
  right,
  onChanged,
}: {
  incidentId: string;
  relationship: RecordRelationshipOut;
  left: string;
  right: string;
  onChanged: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [type, setType] = useState<RelationshipType>(relationship.relationship_type);
  const [reason, setReason] = useState("");
  const [saving, setSaving] = useState(false);
  const facts = relationshipFacts(relationship);

  async function save() {
    setSaving(true);
    try {
      await overrideRelationship(incidentId, relationship.relationship_id, {
        corrected_relationship: type,
        operator: "",
        reason,
      });
      setEditing(false);
      onChanged();
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className={styles.relation}>
      <div className={styles.relationHead}>
        <span className={styles.strong}>
          {left} → {right}
        </span>
        <span className={styles.relationType}>{RELATIONSHIP_LABEL[relationship.relationship_type] ?? relationship.relationship_type}</span>
        <span className={styles.muted}>{Math.round(relationship.confidence * 100)}%</span>
        {relationship.overridden && <span className={styles.countWarn}>dikoreksi</span>}
        <span className={styles.spacer} />
        <button type="button" className={styles.smallButton} onClick={() => setEditing((v) => !v)}>
          {editing ? "Batal" : "Koreksi"}
        </button>
      </div>
      {facts && <span className={styles.muted}>{facts}</span>}
      {editing && (
        <div className={styles.relationForm}>
          <select value={type} onChange={(e) => setType(e.target.value as RelationshipType)}>
            {(Object.keys(RELATIONSHIP_LABEL) as RelationshipType[]).map((t) => (
              <option key={t} value={t}>
                {RELATIONSHIP_LABEL[t]}
              </option>
            ))}
          </select>
          <input placeholder="Alasan koreksi" value={reason} onChange={(e) => setReason(e.target.value)} />
          <button type="button" className={styles.primary} disabled={saving || type === relationship.relationship_type} onClick={save}>
            Simpan
          </button>
        </div>
      )}
    </div>
  );
}

// --- Data insiden ----------------------------------------------------------------------------------

function MetaTab({ incident, onSave, onArchive }: Props) {
  const [draft, setDraft] = useState({
    title: incident.title,
    voltage_level_kv: incident.voltage_level_kv?.toString() ?? "",
    station_name: incident.station_name ?? "",
    bay_name: incident.bay_name ?? "",
    operator_notes: incident.operator_notes ?? "",
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const stationFromRecords = incident.records.some((r) => r.station_name);

  async function save() {
    setSaving(true);
    setSaved(false);
    try {
      await onSave({
        title: draft.title,
        voltage_level_kv: draft.voltage_level_kv ? Number(draft.voltage_level_kv.replace(",", ".").replace(/[^0-9.]/g, "")) : null,
        station_name: draft.station_name || null,
        bay_name: draft.bay_name || null,
        operator_notes: draft.operator_notes || null,
      });
      setSaved(true);
    } finally {
      setSaving(false);
    }
  }

  const field = (key: keyof typeof draft, label: string, placeholder?: string, warn?: string) => (
    <label className={styles.field}>
      <span className={styles.label}>{label}</span>
      <input
        value={draft[key]}
        placeholder={placeholder}
        className={warn ? styles.inputWarn : undefined}
        onChange={(e) => setDraft({ ...draft, [key]: e.target.value })}
      />
      {warn && <span className={styles.warnText}>{warn}</span>}
    </label>
  );

  return (
    <div className={styles.metaForm}>
      <div className={styles.metaGrid}>
        {field("title", "Judul")}
        {field("voltage_level_kv", "Tegangan (kV)", "mis. 150")}
        {field(
          "station_name",
          "GI",
          "mis. GI CIBATU",
          !draft.station_name && !stationFromRecords ? "Belum diisi: nama GI tidak ada di file CFG." : undefined,
        )}
        {field("bay_name", "Bay")}
      </div>
      <label className={styles.field}>
        <span className={styles.label}>Catatan</span>
        <textarea
          rows={4}
          value={draft.operator_notes}
          placeholder="Temuan patroli, koordinasi dengan GI lawan, dll."
          onChange={(e) => setDraft({ ...draft, operator_notes: e.target.value })}
        />
      </label>
      <span className={styles.muted}>
        GI insiden menentukan ujung yang dipakai sebagai acuan jam dan sudut cerita. Jenis aset, keluarga proteksi, dan
        kualitas jam diisi otomatis dari rekaman.
      </span>
      <div className={styles.toolbar}>
        <button type="button" className={styles.primary} disabled={saving} onClick={save}>
          {saving ? "Menyimpan…" : "Simpan"}
        </button>
        {saved && <span className={styles.muted}>Tersimpan.</span>}
        <button type="button" className={styles.dangerLink} onClick={onArchive}>
          Arsipkan insiden
        </button>
      </div>
    </div>
  );
}
