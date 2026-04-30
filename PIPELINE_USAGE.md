# Panduan Penggunaan Pipeline

## 1. Klasifikasi File Tunggal

```bash
cd pipeline/
python models/predict.py path/to/file.cfg
```

Contoh output (multi-class ML):
```
============================================================
  File    : PCS900_RCD_01016_20240518_044955_016.CFG
  Label   : PETIR
  Tier    : 2  (multiclass_random_forest)
  Conf.   : 78%
  Evidence: Classifier ML: PETIR (78%)  AR berhasil — gangguan terkonfirmasi transien.
            dur=57ms  peak_i=953A  i0/i1=1.227  zone=Z1
============================================================
  Station      : NR
  Relay        : LINE_DISTANCE_RELAY
  Zone         : Z1
  Trip type    : single_pole
  Phases       : C
  Duration     : 57 ms
  fault_count  : 1
  peak_I       : 953 A
  i0/i1        : 1.227
  voltage sag  : 0.107 pu
  Reclose ok   : True
```

---

## 2. Klasifikasi Batch

```bash
python batch_extract.py
```

- Memindai seluruh `raw_data/` secara rekursif
- Melewati folder `olah/`, `_extracted/`, `locus/`, `analisa/`
- Output:
  - `data/features/labeled_features.csv`
  - `data/features/labeled_features_87l.csv`
  - `data/features/extraction_errors.csv`

### Format Output CSV

| Kolom | Keterangan |
|---|---|
| `predicted_label` | Hasil klasifikasi |
| `confidence` | Kepercayaan model (0–1) |
| `tier` | 1 = aturan deterministik, 2 = ML, 0 = fallback |
| `rule_name` | Layer yang terpicu |
| `evidence` | Detail analisis + probabilitas per kelas |
| `folder_label` | Label dari nama folder (jika ada) |
| `correct` | **Diisi stakeholder** — apakah prediksi benar? |
| `notes` | **Diisi stakeholder** — catatan lapangan |

---

## 3. Web App

Stack: **FastAPI** backend (`webapp/api/main.py`) + **React / Vite** frontend (`webapp/frontend/`).

```bash
pip install -r requirements.txt

# Terminal 1 — backend
uvicorn webapp.api.main:app --reload --port 8000

# Terminal 2 — frontend
cd webapp/frontend
npm install
npm run dev
# Buka http://localhost:5173
```

Fitur:
- Upload file `.cfg` + `.dat` atau pilih dari `raw_data/`
- Hasil klasifikasi + evidence + probabilitas per kelas (cause ranking)
- Workspace per-rele (21 / 87L / 87T / OCR / REF)
- Locus impedansi R-X interaktif (relay 21)

Panduan labeling tim tersedia di [LABELING_GUIDELINES.md](LABELING_GUIDELINES.md).

---

## 4. Urutan Layer Klasifikasi

Sistem menjalankan layer secara berurutan. Layer pertama yang cocok mengembalikan hasil final.

```
INPUT: fault event terdeteksi
        │
        ▼
[Transformer?] → prot = 87T → transformer classifier (inrush / internal fault / dll.)
        │ bukan 87T
        ▼
Layer 1a: fault_on_reclose_phase_change
  Syarat: fault_count 2–20, fasa berbeda antar kejadian, dur >80ms, AR tidak berhasil
  → KONDUKTOR / TOWER (85%)
        │ tidak cocok
        ▼
Layer 1b: three_pole_failed_reclose
  Syarat: trip 3-fasa, AR gagal, peak >50A
  → GANGGUAN PERMANEN (75%)
        │ tidak cocok
        ▼
Layer 1c: explicit_failed_reclose
  Syarat: AR gagal secara eksplisit, dur >10ms, peak >100A
  → GANGGUAN PERMANEN (90%)
        │ tidak cocok
        ▼
Layer 2: Multi-class RandomForest (13 fitur)
  Kelas: PETIR, LAYANG-LAYANG, POHON, HEWAN, BENDA ASING, KONDUKTOR, PERALATAN / PROTEKSI
  → label + confidence + probabilitas tiap kelas
        │ model tidak tersedia
        ▼
Fallback → PERLU INVESTIGASI
```

### DFR Eksternal (tanpa sinyal rele)

Ketika `protection_type = UNKNOWN` (DFR berdiri sendiri, tanpa channel trip rele):
- Klasifikasi **tetap berjalan** — fitur waveform tidak bergantung pada tipe rele
- Channel CB (52A, 52B, PMT BUKA) dideteksi untuk konfirmasi trip dan reclose
- Evidence panel menampilkan caveat `[DFR EKSTERNAL]`
- Rekomendasi: konfirmasi penyebab menggunakan rekaman rele jarak jika tersedia

---

## 5. Melatih Ulang Model

```bash
python models/train.py
```

Proses:
1. Baca `data/features/labeled_features.csv`
2. Filter sampel berkualitas (`scaling_ok=True`, `duration_ok=True`)
3. Exclude sampel yang sudah ditangani Tier 1
4. Training RandomForestClassifier (300 trees, balanced_subsample)
5. Stratified k-fold cross-validation
6. Output: `models/fault_classifier.pkl` + `models/petir_tree.pkl` (alias)

Model saat ini: **5-class RandomForest** (~400 sampel, CV accuracy 82.9%)
Catatan: `data/features/labeled_features_87l.csv` sekarang dikumpulkan terpisah untuk corpus 87L, belum ikut training utama.

---

## 6. Ekstraksi Arsip

Arsip ZIP/RAR di `raw_data/` perlu diekstrak terlebih dahulu:

```bash
# Instal 7-Zip dulu: https://www.7-zip.org/
python extract_all.py
```

- Skip otomatis jika sudah diekstrak (ada folder sama atau marker `.extracted`)
- Opsi `--dry-run` untuk preview tanpa mengekstrak
- Opsi `--force` untuk ekstrak ulang

---

## 7. Kenapa File Gagal Diklasifikasikan?

| Error | Penyebab | Solusi |
|---|---|---|
| `COMTRADE parse failed` | File `.dat` hilang / format tidak standar | Cek kelengkapan file |
| `No fault detected` | Window rekaman tidak mengandung gangguan | Normal — bukan error |
| `Feature extraction failed` | Channel arus/tegangan tidak dikenali | Tambahkan pola di `core/channel_normalizer.py` |
| `DIFFERENTIAL is not supported` | File dari rele 87L | Inference masih fallback ke gelombang; untuk training batch sekarang masuk `labeled_features_87l.csv` |
| `DIRECTIONAL_EF is not supported` | File dari rele 67N | Belum didukung |

---

## 8. Menambahkan Pola Channel Baru

Jika merk relay baru menghasilkan nama channel yang tidak dikenali:
1. Tambahkan pola di `core/channel_mappings.json` (bagian yang sesuai per merk)
2. Untuk pola deteksi proteksi, tambahkan di `core/protection_router.py`

Contoh log warning:
```
Could not normalize channel 'IaR' (unit: A, mfr: UNKNOWN)
```
