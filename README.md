# AI Analisis Gangguan DFR — Pipeline

Sistem klasifikasi penyebab gangguan transmisi berbasis COMTRADE IEEE C37.111.

Update terakhir: 3 Juni 2026

---

## Ringkasan Sistem

Pipeline ini menerima rekaman COMTRADE dari rele jarak, rele diferensial trafo, dan
DFR eksternal (Qualitrol, Toshiba, dll), lalu berperan sebagai alat bantu respon awal
sebelum tim lapangan tiba: probabilistik untuk gangguan penghantar (prediksi penyebab
fisik) dan evidence-based triage untuk gangguan trafo (interpretasi sinyal lokal +
pemetaan evidence yang masih dibutuhkan). Untuk gangguan penghantar, sistem saat ini
mengklasifikasikan penyebab fisik ke dalam 7 kelas:

| Kelas | Deskripsi |
|---|---|
| **PETIR** | Sambaran petir langsung atau induced overvoltage |
| **LAYANG-LAYANG** | Kontak layang-layang dengan konduktor |
| **POHON / VEGETASI** | Sentuhan pohon/ranting pada ROW |
| **HEWAN / BINATANG** | Kontak hewan (ular, burung, babi, tikus, dll.) |
| **BENDA ASING** | Benda asing non-hayati (aluminium foil, terpal, spanduk, dll.) |
| **KONDUKTOR / TOWER** | Kerusakan konduktor, joint, klem, atau struktur tower |
| **PERALATAN / PROTEKSI** | Gangguan pilot wire, teleprotection/PLCC, CT/VT, PMT, atau peralatan proteksi terkait |

---

## Alur Klasifikasi

```
INPUT: file .cff ABB atau .cfg + .dat (COMTRADE)
        │
        ▼
1. Parse COMTRADE  →  Record
2. determine_protection  →  ProtectionType (DISTANCE / 87T / 87L / OCR / REF / UNKNOWN)
3. detect_fault          →  FaultEvent (inception time, duration, reclose outcome)
        │
        ├── [87T] → Diff vs restraint plot + dual-slope evaluation
        │           (operated / not_operated / fast_operated). Tanpa verdict AI;
        │           klasifikasi event trafo (inrush/internal/through/dll.) ada
        │           offline di models/transformer_classifier.py — belum di-wire ke UI.
        │
        ├── [87L] → Line-differential branch
        │           differential vs restraint, harmonic morphology, AI fault analysis
        │
        ├── [21]  → Distance branch
        │           impedance locus R-X plane (DFT phasor + k0/CT-VT overrides,
        │           RIO/XRIO zone overlay) + line classifier
        │
        └── [OCR / REF / DFR generic] → Line classifier
                │
                ▼
        4. extract_distance_features  →  17-feature dict
        5. Tier 1 rules (rules.py)
           ├── fault_on_reclose_phase_change   → KONDUKTOR / TOWER (85%)
           ├── three_pole_failed_reclose        → GANGGUAN PERMANEN (75%)
           └── explicit_failed_reclose          → GANGGUAN PERMANEN (90%)
                │ tidak cocok
                ▼
        6. Tier 2 Multi-class ML — LightGBM (class_weight='balanced')
           → PETIR / LAYANG-LAYANG / POHON / HEWAN / BENDA ASING / KONDUKTOR / PERALATAN
           → probabilitas per kelas + evidence narrative ditampilkan di UI
                │
                ├── jika kelas teratas = PETIR:
                │   tambahkan PETIR sub-mechanism (Shielding Failure / Back-Flashover
                │   / Belum Konklusif) dengan reasoning data-driven (jumlah fasa, kA puncak)
                │
                ▼ model tidak tersedia
        7. Fallback → PERLU INVESTIGASI
```

### Catatan DFR Eksternal

Bila file berasal dari DFR eksternal (Qualitrol, Toshiba standalone) tanpa sinyal trip rele:
- `protection_type = UNKNOWN` namun klasifikasi **tetap berjalan** menggunakan fitur gelombang
- Fitur waveform (di/dt, peak-I, THD, inception angle) tidak bergantung pada jenis rele
- Channel status CB (52A, 52B, PMT BUKA) juga dideteksi untuk konfirmasi trip dan reclose
- Evidence panel menampilkan caveat `[DFR EKSTERNAL]` dan rekomendasi mengkonfirmasi dengan rekaman rele jika tersedia

---

## Komponen Utama

| File | Fungsi |
|---|---|
| `core/comtrade_parser.py` | Parse COMTRADE, perbaikan format CFG beragam |
| `core/channel_normalizer.py` | Normalisasi nama channel multi-merk (ABB, Siemens, NARI, GE, Alstom, Toshiba) |
| `core/protection_router.py` | Deteksi tipe proteksi, zona, trip type, reclose outcome |
| `core/fault_detector.py` | Deteksi inception, durasi, fault count, SOE |
| `core/feature_extractor.py` | Ekstraksi 17 fitur line/transmisi |
| `core/transformer_channel_mapper.py` | Pemetaan channel trafo HV/LV/diff/restraint |
| `core/transformer_feature_extractor.py` | Ekstraksi fitur trafo (H2, H5, slope, DC offset) |
| `models/rules.py` | Tier 1: aturan deterministik KONDUKTOR/PERMANEN (sekarang juga dipanggil dari UI endpoint) |
| `models/train.py` | Training **LightGBM** 7-kelas (input: labeled_features.csv); CV report otomatis |
| `models/calibrate.py` | Fit Platt/isotonic probability calibrator pada held-out split → `proba_calibrator.pkl`. Dipakai webapp inference jika tersedia, else fallback ke temperature T=1.5 |
| `models/predict.py` | Inference end-to-end + PETIR sub-mechanism (SF/BFO) classifier |
| `models/transformer_classifier.py` | Klasifikasi event trafo berbasis pengetahuan (6 kelas) — **standalone / batch only**; belum dipanggil dari FastAPI router |
| `models/fault_classifier.pkl` | Model LightGBM aktif (7 kelas line-fault) |
| `models/proba_calibrator.pkl` | (Opsional) calibrator hasil `models/calibrate.py`; auto-detected saat inference |
| `webapp/api/main.py` | FastAPI backend root, lifespan warmup (eager model preload), GZipMiddleware, mount semua router |
| `webapp/api/routers/` | Router per-rele: `upload`, `relay_21`, `relay_87l`, `relay_87t`, `relay_ocr`, `relay_ref` |
| `webapp/api/ml_predict.py` | Builder fitur 17-dim + Tier 1 wiring + LightGBM + calibration + structured evidence + introspection fields untuk React UI |
| `webapp/frontend/` | React (Vite) UI yang terhubung ke FastAPI via `/api/*` (proxy `:5173 → :8000`) |
| `webapp/frontend/src/components/relay/shared/AIFaultResultView.tsx` | Render AI fault result + Provenance panel (Tier 1, applied caps, model meta) + collapsible JSON API Inspector |
| `profiling/` | py-spy helper script (PS + sh) untuk capture flamegraph; output di `profiling/flamegraphs/` (gitignored). Lihat `profiling/README.md` |
| `batch_extract.py` | Ekstraksi fitur batch dari seluruh raw_data/ termasuk corpus kandidat 87L |
| `extract_all.py` | Ekstraksi arsip ZIP/RAR menggunakan 7-Zip |

---

## Cara Menjalankan

### Quick start (clone fresh)
```bash
git clone https://github.com/hafizna/base_ai_tfa.git
cd base_ai_tfa
pip install -r requirements.txt

# Backend FastAPI (terminal 1)
uvicorn webapp.api.main:app --reload --port 8000

# Frontend React/Vite (terminal 2)
cd webapp/frontend
npm install
npm run dev
# buka http://localhost:5173
```

Frontend Vite dev server proxies `/api/*` → `http://localhost:8000`,
so kedua proses harus berjalan bersamaan saat development.

### Ekstraksi arsip (ZIP/RAR)
```bash
# Instal 7-Zip terlebih dahulu: https://www.7-zip.org/
python extract_all.py
# --dry-run  : preview tanpa mengekstrak
# --force    : ekstrak ulang meski sudah ada marker
```

### Batch ekstraksi fitur
```bash
python batch_extract.py
# output: data/features/labeled_features.csv
#         data/features/labeled_features_87l.csv
#         data/features/extraction_errors.csv
```

### Training ulang model
```bash
python models/train.py
# membaca: data/features/labeled_features.csv
# output:  models/fault_classifier.pkl
```

### Fit probability calibrator (opsional)
```bash
python models/calibrate.py                    # Platt scaling (default)
python models/calibrate.py --method isotonic  # isotonic (butuh ≥30 sampel/kelas)
# output: models/proba_calibrator.pkl
# Webapp auto-detect saat startup; fall back ke temperature T=1.5 kalau tidak ada.
```

### Klasifikasi file tunggal
```bash
python models/predict.py path/to/file.cfg
```

### Production deployment
```bash
# start.sh menjalankan uvicorn dengan WEB_CONCURRENCY (default 2 worker).
# Tiap worker reload model bundle (~180 MB RSS), jadi tune sesuai RAM.
WEB_CONCURRENCY=4 ./start.sh

# FastAPI lifespan otomatis warmup model + sklearn/LightGBM saat boot,
# jadi request pertama tidak bayar cold-start ~30 ms pickle load.
# Status warmup expose di GET /api/health → "warmup": {...}.
```

### Retensi data training dari production
Production Docker Compose menyimpan upload mentah ke folder host `training-data/`
ketika `TRAINING_RETENTION_ENABLED=1` (default di `docker-compose.prod.yml`).
Folder ini sengaja di-ignore Git.

Di server EC2, buat token admin sekali:
```bash
cd ~/base_ai_tfa
printf 'TRAINING_ADMIN_TOKEN=%s\n' 'ganti-dengan-token-panjang' > .env
docker compose -f docker-compose.prod.yml up -d --build
```

Yang tersimpan:
- `training-data/raw/<timestamp>_<analysis_id>/` berisi `.cfg/.dat`, `.cff`, atau `.cdb` asli plus `metadata.json`.
- `training-data/labels/feedback.csv` dan `.jsonl` berisi koreksi label dari panel Training Feedback.

Export dari UI: isi token di panel **Training Feedback**, lalu klik **Download archive**.

Export dari SSH:
```bash
cd ~/base_ai_tfa
bash scripts/export_training_archive.sh
```

Setelah ZIP aman dipindah ke laptop, clear archive server:
```bash
cd ~/base_ai_tfa
bash scripts/clear_training_archive.sh --yes
```

Workflow retraining tetap lokal: download ZIP, kurasi label, regenerate
`data/features/labeled_features.csv`, jalankan `python models/train.py` dan
`python models/calibrate.py`, commit model baru, lalu deploy ulang EC2.

### Profiling (Opsi A step #1)
```powershell
# Install py-spy sekali (dev-only, tidak ada di requirements.txt)
python -m pip install --user py-spy

# Capture flamegraph 30 detik sambil eksekusi smoke request
./profiling/profile_request.ps1                    # Windows
./profiling/profile_request.sh                     # Linux/macOS
# output: profiling/flamegraphs/profile_<timestamp>.svg
```

---

## Model Saat Ini

| Parameter | Nilai |
|---|---|
| Algoritma | **LGBMClassifier** (LightGBM, `class_weight='balanced'`) |
| Kelas | PETIR, LAYANG, POHON, HEWAN, BENDA_ASING, KONDUKTOR, PERALATAN (7) |
| Sampel training | ~450+ baris (setelah quality filter + Tier 1 exclusion) |
| Fitur | **17** (lihat `models/train.py:FEATURE_COLS`) |
| CV F1 macro | 0.407 (LGBM) vs 0.352 (RF) — primary metric karena class imbalance |
| CV F1 weighted | 0.757 (LGBM) vs 0.738 (RF) |
| CV accuracy | 0.778 (LGBM) — kurang relevan karena imbalance |
| Kalibrasi | Default: temperature scaling T=1.5. Override otomatis bila `models/proba_calibrator.pkl` tersedia (Platt/isotonic dari held-out split, hasilkan via `python models/calibrate.py`). Tetap dilengkapi ceiling 92% + cap 0.65/0.72 saat voltage absent |
| Caps | Tercatat di response field `applied_caps[]`: `ceiling_92`, `transient_ambiguity`, `equipment_caution`, `petir_digital_caution`. Tiap entry menyimpan before/after + reason untuk audit |
| PETIR sub-mechanism | Shielding Failure / Back-Flashover / Belum Konklusif (rule-based, data-driven) |
| Tier 1 di UI | `models/rules.py:apply_rules()` dijalankan **sebelum** LightGBM. Jika fire → response berisi `tier1: {fired, rule_name, label, evidence}` dan LightGBM dilewati |
| Introspection fields | `raw_probabilities`, `calibrated_probabilities`, `feature_vector_used`, `applied_caps`, `meta.model_version` (timestamp+SHA), `meta.feature_version`, `meta.calibration_method_used` — semua diserialisasi dalam JSON response dan terlihat di **API & JSON Inspector** panel UI |
| Catatan | `POHON` dan `PERALATAN` tetap di taksonomi, tetapi hanya muncul sebagai prediksi bila data latih usable sudah mencukupi. Untuk RF vs LGBM benchmarking jalankan `compare_models.py`. |

---

## Output Klasifikasi

### Line / transmisi
- `PETIR` / `LAYANG-LAYANG` / `POHON / VEGETASI` / `HEWAN / BINATANG` / `BENDA ASING` / `KONDUKTOR / TOWER` / `PERALATAN / PROTEKSI`
- `GANGGUAN PERMANEN` (Tier 1 permanent fault rules)
- `KONDUKTOR / TOWER` (Tier 1 conductor fault rule)
- `PERLU INVESTIGASI` (fallback)

#### PETIR sub-mechanism (saat top-class = PETIR)
Evidence panel menambahkan satu baris yang membedakan dua mekanisme petir,
dengan reasoning data-driven (fasa terganggu + arus puncak kA):

| Subtype | Indikator |
|---|---|
| **Shielding Failure (SF)** | satu fasa, peak < 8 kA → konsisten dengan distribusi EGM (sambaran lolos kawat tanah) |
| **Back-Flashover (BFO)** | multi-fasa **atau** satu fasa dengan peak ≥ 30 kA → potensial tower naik melampaui BIL |
| **Belum Konklusif** | satu fasa, peak 8–30 kA (rentang tumpang tindih) atau skala CT tak dipercaya |

Catatan: keputusan SF/BFO yang lebih akurat memerlukan integrasi data PI
(tower footing resistance + BIL isolator) — saat ini sub-mechanism hanya
heuristik dari waveform lokal.

### Transformer differential (87T) — saat ini

Workspace 87T saat ini hanya menampilkan analisa rekaman COMTRADE,
**tanpa verdict AI**:

- Diagram **differential vs restraint** per fasa (L1, L2, L3)
- Evaluasi terhadap karakteristik dual-slope (`idiff_pickup`, `slope1`/`intersection1`,
  `slope2`/`intersection2`, `idiff_fast`)
- Status operasi:
  - `NOT_OPERATED`
  - `IDIFF_OPERATED` — operating point melewati karakteristik
  - `IDIFF_FAST_OPERATED` — Idiff melewati threshold fast (high-set)
- Daftar fasa yang operated

> **Catatan**: classifier 6-kelas (`INRUSH MAGNETISASI` / `GANGGUAN INTERNAL TRAFO` /
> `GANGGUAN EKSTERNAL (THROUGH)` / `TEGANGAN LEBIH OVEREKSITASI` / `KEMUNGKINAN MALOPERATE`
> / `PERLU INVESTIGASI`) tersedia di `models/transformer_classifier.py` untuk pemakaian
> standalone (`models/predict.py path/to/file.cfg`), tetapi **belum diakses dari React
> UI**. Lihat bagian "Filosofi Analisa Gangguan Trafo" untuk arah pengembangan.

### Distance (relay 21)
- AI fault analysis (memakai line classifier + PETIR sub-mechanism)
- Diagram R-X impedance locus (DFT phasor) dengan zone overlay dari RIO/XRIO
- Override CT primary, VT primary, dan k0 (residual compensation) per analisis

---

## Arah Berikutnya Sebelum Data Tambahan Datang

Beberapa pekerjaan penting masih bisa dikerjakan **tanpa menunggu data baru**:

1. Rapikan filosofi keputusan trafo menjadi output bertingkat, bukan satu label tunggal.
   Pisahkan:
   - `event_class` = inrush / through-fault / internal / overexcitation / maloperate / unknown
   - `fault_origin` = internal trafo / eksternal distribusi / eksternal transmisi / proteksi-peralatan / unknown
   - `protection_assessment` = proteksi benar / proteksi tidak selektif / evidence belum cukup

2. Tambahkan skema evidence multi-relay.
   Untuk kasus trafo, file satu relay sering tidak cukup. Pipeline perlu siap menyimpan evidence dari:
   - 87T / REF / GFR HV / SBEF / OCR incomer
   - relay sisi distribusi / outgoing feeder / incoming breaker
   - status PMT, LOR/lockout, close failure, auto transfer, dll.

3. Ubah UX dari "langsung penyebab akhir" menjadi "kesimpulan + alasan".
   Tampilkan:
   - apa yang terlihat dari rekaman lokal
   - apa yang belum terlihat
   - asumsi apa yang dipakai
   - data tambahan apa yang dibutuhkan untuk konfirmasi

4. Kumpulkan corpus terpisah untuk domain yang berbeda.
   - `labeled_features.csv` untuk line-distance / generic line events
   - `labeled_features_87l.csv` untuk line differential
   - corpus kurasi trafo untuk kasus 87T dan koordinasi eksternal-internal

5. Tambahkan state "unknown / insufficient evidence" secara eksplisit.
   Ini penting untuk rekaman chopped, rekaman satu sisi saja, atau kasus di mana root cause tidak bisa dipastikan hanya dari satu COMTRADE.

---

## Filosofi Analisa Gangguan Trafo (Roadmap)

> **Status implementasi**: bagian ini adalah **arah desain**, bukan apa yang sudah jalan.
> Saat ini workspace 87T hanya menampilkan diff/restraint plot dan operated status
> dari rekaman COMTRADE — belum ada hierarchical evidence reasoning, multi-relay
> evidence fusion, atau layered output. Section di bawah menjelaskan target
> arsitektur untuk iterasi berikutnya.

Untuk trafo PLN, filosofi identifikasi penyebab **tidak bisa disamakan** dengan line classifier multi-kelas biasa.

Pada praktik operasi, banyak kejadian trafo sebenarnya berasal dari **faktor eksternal**:
- fault sisi distribusi
- fault eksternal di feeder atau downstream
- gangguan sistem di luar trafo
- issue proteksi/peralatan bantu

Artinya, bila tidak ada indikasi kuat kerusakan internal trafo, asumsi kerja yang lebih aman adalah:

`trip trafo belum tentu = fault internal trafo`

Karena itu, pendekatan yang lebih tepat adalah **hierarchical evidence-based reasoning**, bukan langsung memaksa satu label sebab fisik dari waveform lokal saja.

### Prinsip Utama

1. Tentukan dulu **apa yang dilakukan proteksi**, bukan langsung "apa penyebab finalnya".
2. Bedakan antara:
   - **event type** yang terlihat di relay lokal
   - **lokasi/origin gangguan**
   - **root cause fisik**
3. Bila evidence internal trafo lemah, lebih aman memberi hasil:
   - `kemungkinan external through-fault`
   - `kemungkinan fault distribusi`
   - `kemungkinan issue proteksi/peralatan`
   daripada memaksa `internal fault`.

### Urutan Analisa Yang Disarankan

#### Layer 1 — Event interpretation lokal

Pertanyaan pertama:
- apakah ini inrush?
- apakah ini through-fault eksternal?
- apakah ada indikasi internal differential operate yang meyakinkan?
- apakah ini overexcitation?
- apakah justru maloperate / abnormal protection behavior?

Di layer ini, fitur ilmiah yang sudah ada tetap berguna:
- harmonics (H2/H5)
- differential vs restraint
- slope / operate region
- I0 / sequence behavior
- waveform current magnitude dan timing

Tetapi output layer ini sebaiknya dibaca sebagai:

`apa arti event ini di relay lokal`

bukan langsung:

`apa root cause final di lapangan`

#### Layer 2 — Fault origin / selectivity assessment

Setelah event lokal dibaca, pertanyaan berikutnya:
- apakah gangguan berasal dari internal trafo?
- apakah gangguan berasal dari sisi distribusi?
- apakah gangguan berasal dari jaringan eksternal dan trafo hanya ikut trip?
- apakah PMT / relay / communication / auxiliary system yang bermasalah?

Di sini evidence dari **relay lain** menjadi sangat penting:
- incomer trip atau tidak
- feeder/outgoing berhasil isolasi atau tidak
- HV side trip sebagai backup atau primary
- REF, GFR HV, SBEF, OCR, BF, lockout, LBB
- urutan waktu trip antar relay
- status breaker open/close dan dead time

Contoh filosofi operasi:
- Jika fault ada di sisi distribusi dan incomer/downstream berhasil mengisolasi fault, maka dari sudut pandang proteksi, kejadian itu lebih tepat dianggap **distribution fault**, bukan internal trafo.
- Jika NGR putus, maka pembacaan 87T saja tidak cukup; perlu melihat urutan **REF -> GFR HV -> SBEF** dan device lain yang relevan.

### Fault Localization vs Root Cause

Untuk domain trafo, perlu dibedakan dengan tegas antara:

- **fault localization / origin determination**
- **root cause identification**

Jika localization sudah cukup jelas dari relay coordination, maka pipeline **tidak perlu** memakai model ML besar untuk memutuskan origin fault.

Contoh:
- hanya `OCR LV` yang bekerja dan downstream/incomer berhasil mengisolasi fault
- `87T` tidak menunjukkan indikasi internal yang kuat
- HV side tidak trip atau hanya pickup sebagai backup

Dalam situasi seperti ini, kesimpulan engineering yang paling penting sebenarnya sudah bisa diperoleh dari logika proteksi:

`kejadian ini kemungkinan besar berasal dari sisi distribusi / eksternal`

bukan dari model ML.

Artinya:
- **relay coordination** adalah sumber kebenaran utama untuk localization
- **ML** hanya dipakai bila ada pertanyaan yang memang belum dijawab jelas oleh koordinasi relay

Sub-problem yang masih cocok dibantu ML:
- membedakan `inrush` vs `internal fault` vs `through-fault` dari waveform lokal
- memberi anomaly score untuk kemungkinan maloperate
- membantu ranking hipotesis saat evidence relay tidak lengkap

Sub-problem yang sebaiknya **tidak** diserahkan ke model besar:
- menentukan origin fault padahal sequence relay sudah jelas
- mengalahkan kesimpulan selectivity yang sudah kuat dari relay working order
- memaksa root cause final dari satu file lokal saja

#### Layer 3 — Root cause family

Baru setelah dua layer di atas cukup jelas, pipeline boleh memberi family penyebab:
- `INTERNAL_TRANSFORMER`
- `EXTERNAL_DISTRIBUTION`
- `EXTERNAL_TRANSMISSION`
- `PERALATAN / PROTEKSI`
- `EXTERNAL_OBJECT_NEAR_TRAFO` (mis. ular/burung pada bushing, CT conductor, jumper, dll.)
- `UNKNOWN`

Ini lebih realistis untuk domain trafo dibanding memaksa semua kasus ke label fisik yang sangat spesifik sejak awal.

### Apa Yang Perlu Dilihat Untuk Kasus Trafo

Checklist evidence yang ideal:

- 87T differential operate / restraint / slope
- harmonic content (H2/H5)
- REF / restricted earth fault
- GFR HV / OCR HV / SBEF / NGR-related indication
- incomer dan outgoing/distribution relay
- breaker status, lockout, close fail, breaker fail
- bus / feeder voltage collapse pattern
- SOE urutan trip antar bay
- hasil inspeksi lapangan
- catatan operator / gangguan distribusi / laporan patrol

### Saran Arsitektur Umum

Untuk domain trafo, pendekatan terbaik kemungkinan bukan satu model ML besar, tetapi:

1. **Rule-based evidence fusion** untuk menentukan event meaning dan fault origin.
2. **ML hanya untuk sub-problem yang stabil**, misalnya:
   - inrush vs internal fault vs overexcitation
   - local waveform anomaly scoring
   - ranking kemungkinan external-object signature jika nanti datanya cukup
3. **Uncertainty-aware output**.
   Pipeline harus boleh menjawab:
   - `indikasi external through-fault, butuh evidence relay distribusi`
   - `indikasi internal fault lemah`
   - `kemungkinan proteksi/peralatan`
   - `belum cukup evidence`

Dengan kata lain, untuk trafo:
- bila relay coordination sudah menjawab localization, ikuti hasil itu
- bila waveform lokal masih perlu diinterpretasi, gunakan rule + ML kecil pada layer event meaning
- bila evidence belum cukup, tampilkan `unknown / need more relays`, bukan pseudo-precision

### Implikasi Untuk Labeling Data Trafo

Sebelum data internal-fault trafo memadai, jangan buru-buru membuat taxonomy fisik yang terlalu halus.

Lebih aman mulai dari umbrella categories:
- `EXTERNAL_DISTRIBUTION`
- `EXTERNAL_SYSTEM`
- `INTERNAL_TRANSFORMER`
- `PERALATAN / PROTEKSI`
- `MALOPERATE / UNCONFIRMED`

Kasus seperti ular pada bushing, burung di CT conductor, atau benda asing di area terminal trafo bisa diletakkan dulu sebagai:

`EXTERNAL_OBJECT_NEAR_TRAFO`

atau sementara masuk:

`EXTERNAL_SYSTEM`

lalu dipecah lagi setelah datanya cukup.

### Kesimpulan Praktis

Untuk trafo, pertanyaan utamanya sebaiknya berubah dari:

`apa penyebab finalnya dari waveform ini?`

menjadi:

`apa yang bisa dipastikan dari relay lokal, apa yang paling mungkin sebagai origin fault, dan evidence tambahan apa yang masih dibutuhkan?`

Itu lebih dekat dengan praktik engineering PLN, lebih jujur terhadap keterbatasan data, dan lebih scalable untuk kasus-kasus kompleks.

---

## Struktur Folder

```text
base_ai_tfa/                          ← repo root
  core/
    comtrade_parser.py                  ← parser COMTRADE multi-merk
    channel_normalizer.py
    channel_mappings.json               ← pola channel per merk (NARI, ABB, Siemens, Toshiba, dll.)
    protection_router.py                ← deteksi tipe rele + zona + trip type
    fault_detector.py
    feature_extractor.py                ← 17 fitur line/transmisi
    differential_feature_extractor.py   ← fitur 87L (rise time, DWT energy, dll.)
    transformer_channel_mapper.py
    transformer_feature_extractor.py
    path_heuristics.py
    rio_parser.py                       ← parser file proteksi RIO/XRIO untuk overlay zona
  config/
    channel_mappings.json
    relay_lookup.json
  models/
    rules.py                            ← Tier 1 deterministic rules
    train.py                            ← LightGBM training (CV report)
    calibrate.py                        ← Fit Platt/isotonic calibrator (held-out)
    predict.py                          ← inference + PETIR SF/BFO sub-classifier
    transformer_classifier.py
    fault_classifier.pkl                ← model aktif (LightGBM, 7 kelas)
    proba_calibrator.pkl                ← (opsional) calibrator hasil calibrate.py
    petir_tree.pkl                      ← legacy alias
    stage3_petir_classifier.pkl
    stage3_feature_columns.pkl
  webapp/
    api/
      main.py                           ← FastAPI app root + lifespan warmup + GZipMiddleware
      ml_predict.py                     ← Tier 1 wiring + LightGBM + calibration + structured evidence + introspection
      schemas.py                        ← Pydantic request/response (AIFaultResult diperluas)
      storage.py                        ← session storage (CSV/Postgres fallback)
      routers/
        upload.py                       ← POST /api/upload, GET /api/analysis/{id} (gzipped)
        relay_21.py                     ← distance + impedance locus
        relay_87l.py                    ← line differential
        relay_87t.py                    ← transformer differential
        relay_ocr.py
        relay_ref.py
    frontend/
      package.json
      vite.config.ts                    ← proxy /api/* → :8000
      src/
        App.tsx
        main.tsx
        pages/                          ← Landing, Upload, Workspace
        components/
          panels/                       ← COMTRADEExplorer (per-side waveform), SOETimeline, dll.
          relay/relay21/                ← ImpedanceLocus, ElectricalParams21, AIFaultAnalysis21
          relay/relay87l/               ← DiffRestraintPlot, AIFaultAnalysis87L
          relay/relay87t/               ← FaultRecap87T
          relay/shared/AIFaultResultView.tsx  ← Verdict + cause ranking + structured evidence + Provenance + API/JSON Inspector
        api/client.ts                   ← axios client; baseURL = VITE_API_URL || ""
        context/AnalysisContext.tsx
  profiling/                            ← Opsi A step #1 helpers
    profile_request.ps1                 ← Windows PS helper (uvicorn + py-spy + smoke request)
    profile_request.sh                  ← POSIX helper untuk Linux/CI
    README.md                           ← apa yang dicari di flamegraph
    flamegraphs/                        ← gitignored; output SVG py-spy
  data/
    features/                           ← labeled_features.csv (training set, ~314 KB)
    labels/                             ← labels_from_folders.csv (folder-derived labels)
    predictions/                        ← gitignored; output batch_predict.py
  tests/                                ← pytest suite (78 tests)
  batch_extract.py                      ← batch ekstraksi fitur dari raw_data/
  batch_predict.py
  extract_all.py                        ← unzip arsip via 7-Zip
  Dockerfile                            ← multi-stage: vite build → python runtime
  Procfile                              ← web: ./start.sh
  start.sh                              ← uvicorn dengan WEB_CONCURRENCY env (default 2)
  nixpacks.toml
  railway.json
  requirements.txt
```

---

## Optimasi Performa (Opsi A)

Roadmap inkremental untuk menjaga app tetap di Python (tanpa rewrite ke Rust/Go).
Tujuan: kurangi p50 latency request tanpa kehilangan fleksibilitas eksperimen ML.
Profile dulu (#1), lalu putuskan step berikutnya berdasar data, bukan asumsi.

| # | Step | Status | Catatan |
|---|---|---|---|
| **1** | py-spy profiling helper + flamegraph capture | ✅ Selesai | `profiling/profile_request.{ps1,sh}` — spin up uvicorn isolated, eksekusi smoke request, output SVG. Lihat `profiling/README.md` |
| **2** | Eager model + LightGBM preload di startup | ✅ Selesai | FastAPI `lifespan` panggil `ml_predict.warmup()` — pickle load + sklearn/LightGBM import ditelan ~30 ms di boot. Status di `GET /api/health → warmup` |
| **5** | uvicorn `--workers` configurable via env | ✅ Selesai | `start.sh` baca `WEB_CONCURRENCY` (default 2). Tiap worker ~180 MB RSS; tune sesuai RAM |
| **α** | GZipMiddleware untuk response JSON besar | ✅ Selesai | `minimum_size=10KB`, `compresslevel=6`. 2–5× reduction untuk waveform JSON di `/api/analysis/{id}` dan `/api/recalculate-ratio`. Health checks bypass |
| **6** | mmap `.dat` di `core/comtrade_parser.py` | ⏸️ Re-evaluate dgn file production | Profile sintesis menunjukkan parser ~10% dari active CPU tapi file uji terlalu kecil (26 KB). Worth re-test dengan `.dat` produksi (MB-sized) sebelum decide |
| **3** | ONNX convert + `onnxruntime` inference | ❌ Skip | Profile membuktikan inference < 0.1% active CPU. Effort tinggi untuk savings ~0.7 ms; risiko numerical drift di kalibrasi Platt/isotonic. Drop kecuali ada bukti baru |
| **4** | Vectorize `_digital_sequence_features` loop | ❌ Skip | Tidak muncul di flamegraph; logika bounce rejection + 1.5-breaker sudah teruji. Effort:risk buruk |
| **β** | Binary endpoint (Float32 buffer) untuk `/api/analysis/{id}` | ❌ Skip | JSON serialization tidak terdeteksi sebagai hot path di profile. GZip + Float64 sudah cukup; binary path tidak akan terasa di p50 |
| **γ** | Per-channel lazy load (`/channels/{id}/samples`) | ❌ Skip | Bandwidth bukan masalah dengan GZip aktif; tidak ada data yang menunjukkan user load record besar selektif |

**Hasil profile (2026-05-21):** 30 s sample @ 6× upload synthetic INRUSH (26 KB). Total 5996 samples,
~98% idle (uvicorn waiting). Active CPU breakdown (top non-framework hits):

| Function | % Active CPU |
|---|---|
| `core/comtrade_parser.py::parse_comtrade` (sample loop + CFG load) | ~10% |
| `shutil.copy2` (multipart temp upload) | ~5% |
| `comtrade.py::load` (third-party lib) | ~4% |
| `_compute_locus`, `_compute_electrical_params`, `_digital_sequence_features`, LightGBM `predict_proba`, JSON serialization | tidak muncul di top — bukan bottleneck |

**Keputusan pasca-profile:** sistem sudah well-tuned. Step #1, #2, #5, α yang sudah dieksekusi
sudah meng-cover hot path utama. Optimasi tambahan (#3, #4, β, γ) marginal dan di-skip kecuali
ada bukti baru dari production traffic. Step #6 menunggu re-test dengan `.dat` produksi (>1 MB)
karena synthetic file terlalu kecil untuk men-eksekusi parser scaling. Fokus engineering bergeser
ke feature & correctness.

---

## Status Pengembangan

| Komponen | Status | Catatan |
|---|---|---|
| Parser COMTRADE multi-merk | Selesai | NARI, ABB, Siemens, GE, Alstom, Toshiba, Qualitrol, Reyrolle |
| Deteksi CB status (52A/52B, PMT) | Selesai | Digunakan untuk konfirmasi trip/reclose DFR tanpa sinyal rele |
| Tier 1 rule engine | Selesai | 3 aturan deterministik KONDUKTOR/PERMANEN + Rule 0 CT anomaly + Rule 1b SOE mismatch |
| **Tier 1 wired ke UI endpoint** | Selesai | `webapp/api/ml_predict.run_ml_prediction` panggil `apply_rules()` sebelum LightGBM; response berisi `tier1: {fired, rule_name, label, evidence}` |
| Multi-class fault classifier (line) | Selesai | **LightGBM 7-class**, F1-macro 0.407 / weighted 0.757; POHON butuh data lebih |
| **PETIR sub-mechanism (SF / BFO)** | Selesai | Heuristik data-driven (fasa + kA puncak), reasoning ditampilkan di evidence |
| **Probability calibration (Platt/isotonic)** | Selesai | `models/calibrate.py` hasilkan `proba_calibrator.pkl` dari held-out split; webapp auto-detect, fallback ke T=1.5 |
| **Structured evidence + Provenance panel** | Selesai | Evidence sekarang `{text, severity, weight, kind}`; UI render badge berwarna + tampilkan applied caps + Tier 1 + model metadata |
| **API & JSON Inspector di UI** | Selesai | Collapsible panel di bawah AI fault result — tampilkan endpoint, request/response JSON, latency, status |
| **Model versioning + introspection** | Selesai | `meta.model_version` = `<trained_at_date>+<sha8>`, `feature_version`, `model_sha256_prefix`, `calibration_method_used`, `class_counts` di-expose di response |
| **Eager model preload (lifespan)** | Selesai | Opsi A #2 — first request tidak bayar cold-start pickle load |
| **GZip response compression** | Selesai | Opsi A α — waveform JSON dikompres untuk endpoint besar |
| Transformer differential (87T) — diff/restraint visualisasi | Selesai | Dual-slope characteristic + operated/not_operated/fast_operated status per fasa |
| Transformer event AI classifier (inrush / internal / through / dll.) | Belum di-wire ke UI | `models/transformer_classifier.py` jalan dari CLI saja; React 87T workspace belum memanggilnya. Perlu integrasi + filosofi layered (event → origin → root cause) |
| Distance (relay 21) impedance locus | Selesai | DFT phasor R-X plane, k0 + CT/VT override, RIO/XRIO zone overlay |
| Line differential (87L) | Selesai | Diff vs restraint, AI fault analysis, rise time + DWT energy ratio |
| **Siemens 7UT side suffix (.a / .b / .c)** | Selesai | HV vs LV current panels terpisah di COMTRADE Explorer (bukan overlay per fasa) |
| Backend FastAPI + frontend React/Vite | Selesai | Migrasi dari Flask + Jinja templates (legacy app dihapus April 2026) |
| Batch extraction pipeline | Selesai | ZIP/RAR via 7-Zip, skip duplikat, error log, simpan corpus 87L |
| **Profiling instrumentation (py-spy)** | Selesai | Opsi A #1 — helper script + flamegraph output |
| Kurasi data stakeholder | Berlanjut | Isi `correct`/`notes` di `labeled_features.csv` |
| Data kelas POHON | Kurang | Perlu minimal 10+ rekaman berlabel POHON untuk training |

Panduan labeling ringkas tersedia di [LABELING_GUIDELINES.md](LABELING_GUIDELINES.md).
