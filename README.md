# AI Analisis Gangguan DFR — Pipeline

Sistem klasifikasi penyebab gangguan transmisi berbasis COMTRADE IEEE C37.111.

Update terakhir: April 2026

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
INPUT: file .cfg + .dat (COMTRADE)
        │
        ▼
1. Parse COMTRADE  →  Record
2. determine_protection  →  ProtectionType (DISTANCE / 87T / 87L / UNKNOWN)
3. detect_fault          →  FaultEvent (inception time, duration, reclose outcome)
        │
        ├── [87T] → Transformer classifier (inrush, internal fault, through fault, dll.)
        │
        └── [DISTANCE / generic trip / UNKNOWN] → Line classifier
                │
                ▼
        4. extract_distance_features  →  feature dict
        5. Tier 1 rules (rules.py)
           ├── fault_on_reclose_phase_change   → KONDUKTOR / TOWER (85%)
           ├── three_pole_failed_reclose        → GANGGUAN PERMANEN (75%)
           └── explicit_failed_reclose          → GANGGUAN PERMANEN (90%)
                │ tidak cocok
                ▼
        6. Tier 2 Multi-class ML (LightGBM/RandomForest, 17 fitur)
           → PETIR / LAYANG-LAYANG / POHON / HEWAN / BENDA ASING / KONDUKTOR / PERALATAN
           → probabilitas per kelas ditampilkan di UI
                │ model tidak tersedia
                ▼
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
| `models/rules.py` | Tier 1: aturan deterministik KONDUKTOR/PERMANEN |
| `models/train.py` | Training RandomForest 7-kelas (input: labeled_features.csv) |
| `models/predict.py` | Inference end-to-end untuk satu file .cfg |
| `models/transformer_classifier.py` | Klasifikasi event trafo berbasis pengetahuan |
| `models/fault_classifier.pkl` | Model terlatih aktif (jumlah kelas mengikuti data trainable) |
| `webapp/api/main.py` | FastAPI backend: upload, analysis, relay-specific endpoints |
| `webapp/frontend/` | React (Vite) UI yang terhubung ke FastAPI via `/api/*` |
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

### Klasifikasi file tunggal
```bash
python models/predict.py path/to/file.cfg
```

---

## Model Saat Ini

| Parameter | Nilai |
|---|---|
| Algoritma | RandomForestClassifier (300 trees) |
| Kelas | PETIR, LAYANG, POHON, HEWAN, BENDA_ASING, KONDUKTOR, PERALATAN |
| Sampel training | ~400 baris (setelah quality filter + Tier 1 exclusion) |
| Fitur | 13 (lihat `models/train.py:FEATURE_COLS`) |
| CV accuracy | 82.9% ± 2.6% (5-fold stratified) |
| CV F1 weighted | 78.3% ± 3.5% |
| Catatan | `POHON` dan `PERALATAN` tetap muncul di taksonomi, tetapi hanya menjadi kelas prediksi bila data latih usable sudah mencukupi |

---

## Output Klasifikasi

### Line / transmisi
- `PETIR` / `LAYANG-LAYANG` / `POHON / VEGETASI` / `HEWAN / BINATANG` / `BENDA ASING` / `KONDUKTOR / TOWER` / `PERALATAN / PROTEKSI`
- `GANGGUAN PERMANEN` (Tier 1 permanent fault rules)
- `KONDUKTOR / TOWER` (Tier 1 conductor fault rule)
- `PERLU INVESTIGASI` (fallback)

### Transformer differential (87T)
- `INRUSH MAGNETISASI`
- `GANGGUAN INTERNAL TRAFO`
- `GANGGUAN EKSTERNAL (THROUGH)`
- `TEGANGAN LEBIH / OVEREKSITASI`
- `KEMUNGKINAN MALOPERATE`
- `PERLU INVESTIGASI`

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

## Filosofi Analisa Gangguan Trafo

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
base_ai_tfa/                    ← repo root
  core/
    comtrade_parser.py
    channel_normalizer.py
    channel_mappings.json       ← pola channel per merk (NARI, ABB, Siemens, Toshiba, dll.)
    protection_router.py
    fault_detector.py
    feature_extractor.py
    transformer_channel_mapper.py
    transformer_feature_extractor.py
    path_heuristics.py
  models/
    rules.py
    train.py
    predict.py
    transformer_classifier.py
    fault_classifier.pkl        ← model aktif (5-class RandomForest)
    petir_tree.pkl              ← alias legacy (sama dengan fault_classifier.pkl)
  webapp/
    app.py
    templates/
      index.html
      results.html
      history.html
      browse.html
  data/
    features/                   ← di-gitignore; diisi oleh batch_extract.py
  batch_extract.py
  extract_all.py
  requirements.txt
```

---

## Status Pengembangan

| Komponen | Status | Catatan |
|---|---|---|
| Parser COMTRADE multi-merk | Selesai | NARI, ABB, Siemens, GE, Alstom, Toshiba, Qualitrol |
| Deteksi CB status (52A/52B, PMT) | Selesai | Digunakan untuk konfirmasi trip/reclose DFR tanpa sinyal rele |
| Tier 1 rule engine | Selesai | 3 aturan deterministik KONDUKTOR/PERMANEN |
| Multi-class fault classifier | Selesai | 5-class RF, accuracy 82.9%. POHON butuh data lebih |
| Transformer differential support | Selesai | H2/H5/slope/DC offset, 6 kelas event |
| Web app & browse | Selesai | Upload, browse raw_data, history |
| Batch extraction pipeline | Selesai | ZIP/RAR via 7-Zip, skip duplikat, error log, simpan corpus 87L |
| Line differential (87L) dataset lane | Selesai | Rekaman berlabel 87L tidak lagi dibuang saat ekstraksi |
| Kurasi data stakeholder | Berlanjut | Isi `correct`/`notes` di `labeled_features.csv` dan kurasi corpus `labeled_features_87l.csv` |
| Data kelas POHON | Kurang | Perlu minimal 10+ rekaman berlabel POHON untuk training |

Panduan labeling ringkas tersedia di [LABELING_GUIDELINES.md](LABELING_GUIDELINES.md).
