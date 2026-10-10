# Feedback, konteks kejadian, dan training model

Prioritas analisis adalah konteks: urutan gangguan, trip, PMT membuka, reclose,
gangguan berulang, dan SOTF/TOR, beserta implikasi dan bukti tiap sinyal. Model
7 kelas penyebab adalah lapisan berbeda, bukan pengganti pembacaan konteks.

## Konteks dan pembelajaran konteks

`core/event_analysis.py`, `core/analog_trace.py` dan `core/record_sequence.py`
memakai segmentasi yang sama untuk rekaman lengkap dan analisis insiden.
Satu file dapat memuat lebih dari satu episode. Penutupan PMT adalah observasi
mekanis; keberhasilan pemulihan adalah kesimpulan lain. Refault dan SOTF sesudah
penutupan berarti pemulihan tidak bertahan. Dead time bukan durasi gangguan.
Pembacaan ini bersandar pada waveform/kontak PMT/sinyal proteksi, bukan nama folder.

Workspace dan PDF menampilkan konteks sebelum penyebab. Kesimpulan menyimpan
bukti, aturan, konflik, dan confidence. Aturan permanen yang tidak menentukan
penyebab tidak boleh menghasilkan probabilitas penyebab buatan.

Training konteks terpisah dari classifier penyebab:

```powershell
python -m models.context_training --build --training-dir training-data
```

Targetnya adalah jumlah episode, hasil pemulihan, kelas sequence, fasa, jalur
trip, scheme, dan SOTF sesudah reclose. Target hanya berasal dari lapisan yang
direview CONFIRMED/PROBABLE, atau anotasi konteks pada
`config/context_annotations.jsonl`. Label PETIR/HEWAN dari folder dan kesimpulan
yang belum direview bukan target konteks. Raw yang sama tetap satu rekaman.
Fitur konteks mencakup urutan relatif trip/close, zona, send/receive, refault,
SOTF, durasi per episode, serta perubahan arus/tegangan per fasa.

Learner konteks mengevaluasi tiap target pada grouped CV melawan baseline
aturan yang berjalan. Kandidat disimpan hanya bila lebih baik; ia tidak
mengganti fakta terukur atau aturan proteksi otomatis. Minimal diperlukan dua
outcome dengan dua kelompok kejadian independen per outcome. Satu kasus Rawalo
adalah regression/ground truth untuk dikumpulkan, bukan cukup data untuk
mengklaim model konteks telah belajar generalisasi.

Menambah anotasi atau memperbaiki aturan bukan retraining bobot secara otomatis.
Pipeline ini membuat pembelajaran konteks dapat dilakukan dan dievaluasi saat
contoh terverifikasi cukup. Sampai itu terjadi, pembacaan konteks yang sudah
dibetulkan tetap berasal dari engine observasi/penalaran yang transparan.

## Classifier penyebab: tetap 7 kelas

Perubahan kode analisis tidak otomatis melatih LightGBM. Panel feedback menyimpan
koreksi; dataset builder menerapkannya pada training run berikutnya. Model aktif
hanya boleh diganti setelah kandidat lolos evaluasi. Tujuh kelas tetap PETIR,
LAYANG, POHON, HEWAN, BENDA_ASING, KONDUKTOR, dan PERALATAN.

## Jalur yang sama untuk analisis dan training

- `webapp/api/record_payload.py` mengadaptasi COMTRADE ke payload yang sama dengan upload.
- `webapp/api/ml_predict.py:extract_ml_features` membaca fitur waveform, digital,
  pemilihan line, inception dan clearing untuk analisis maupun dataset builder.
- `models/feature_schema.py` mengatur urutan kolom, encoding, log scaling, serta
  penanganan nilai kosong/non-finite secara identik untuk training dan inference.
- Filter Tier 1 menggunakan `models/rules.py:apply_rules`, sama dengan inference.

Dataset menyimpan fingerprint kode reader, hash isi CFG/DAT atau CFF, asal label,
alias file, dan kelompok kejadian. Training menolak dataset yang stale setelah
kode reader berubah. Builder membaca ulang sumber tanpa mengubah file mentah.

## Cara memakai koreksi panel

1. Beri koreksi pada lapisan yang benar. Fasa R/S/T adalah fakta elektrikal;
   PETIR/LAYANG adalah label penyebab yang berbeda.
2. Tandai tingkat keyakinan **CONFIRMED** atau **PROBABLE**. POSSIBLE/UNKNOWN
   tetap tersimpan untuk review dan tidak menggantikan label/fitur training.
3. Isi sumber ground truth yang tersedia (misalnya LIGHTNING_DETECTION atau
   PATROL_REPORT). Sistem tidak menganggap prediksi AI sebagai label terverifikasi.
4. Download arsip dari panel dan ekstrak ke direktori kerja lokal. Raw files dan
   `labels/feedback.jsonl` harus tetap berada dalam struktur arsip yang sama.
5. Bangun dataset dengan perintah berikut dari root repo:

```powershell
python -m models.build_dataset --training-dir training-data
# Opsional: --corpus-root <direktori file COMTRADE tambahan>
# Default inventory: data/features/labeled_features.csv
# Output: data/features/labeled_features_v2.csv dan .audit.json
```

Builder mencocokkan feedback dengan hash bytes rekaman, bukan nama file atau
UUID session saja. Arsip lama tanpa fingerprint feedback tetap bisa dicocokkan
lewat metadata raw upload. Checksum raw files diverifikasi sebelum dipakai.
File yang sama di beberapa upload/folder tidak menjadi beberapa sampel training.

Aturan penerapan:

- Koreksi penyebab terverifikasi menggantikan label folder. Label folder tetap
  menjadi fallback untuk korpus lama dan ditandai sebagai `label_source=folder`.
  Label folder adalah asumsi dataset, bukan ground truth yang sudah dikonfirmasi.
- Koreksi fasa, zona, trip, reclose, jumlah episode, dan tipe gangguan diterapkan
  sesuai lapisannya. Koreksi inception/clearing memicu ekstraksi waveform pada
  window yang dikoreksi, bukan hanya mengganti angka durasinya setelah ekstraksi.
- Jika ada beberapa feedback, gunakan yang terverifikasi paling baru. Opt-out
  pada submission terbaru mengecualikan kasus meskipun confidence-nya UNKNOWN.
- Label penyebab di luar 7 kelas, konflik label duplikat tanpa koreksi penyebab,
  koreksi parsing/mapping yang belum terselesaikan, timing invalid, dan record
  tanpa bukti gangguan dikeluarkan dengan alasan dalam audit.
- PHASE-only feedback tidak mengubah label penyebab; prediction/canonical snapshot
  disimpan untuk audit dan tidak dijadikan target penyebab secara otomatis.

Koreksi panel yang tidak disubmit dan koreksi dalam percakapan tidak otomatis
menjadi baris feedback. Koreksi percakapan dapat memperbaiki reader/aturan;
dataset harus diekstrak ulang untuk membawa perubahan itu ke training.

## Kandidat, evaluasi, dan promosi

```powershell
python -m models.retrain
# Kandidat: models/candidates/fault_classifier.pkl
# Evaluasi: models/candidates/fault_classifier.evaluation.json
# Model aktif belum diganti.
```

`python batch_extract.py` juga memakai builder baru. `python models/train.py`
adalah entrypoint kompatibilitas untuk membuat kandidat dan evaluasi; tidak lagi
menimpa model aktif maupun legacy pickle tanpa pemeriksaan.

Evaluasi membandingkan kandidat dengan estimator baseline yang dilatih ulang
pada fitur historis, memakai **record, target label, dan fold yang sama**.
Prediksi model aktif pada korpus yang pernah dipakai melatihnya tidak dianggap
hasil out-of-sample. Laporan menamai baseline ini secara eksplisit; skor historis
0,407 dari pembagian lama tidak dibandingkan langsung dengan skor grouped CV.

Kelompok kejadian dan raw DAT identik harus tetap dalam fold yang sama.
`StratifiedGroupKFold` menggunakan sampai 5 fold, dibatasi dukungan kelas paling
jarang. Pemilihan split yang mencakup semua kelas hanya melihat label/group,
bukan skor model. Class weights dihitung dari training fold, bukan seluruh data.
Laporan menyimpan split configuration, hash record tiap fold, F1-macro per fold,
dan precision/recall/F1/support per kelas.

Gate menolak promosi jika:

- F1-macro rata-rata kandidat tidak lebih tinggi dari baseline;
- F1 salah satu kelas turun lebih dari 0,05 pada pooled out-of-fold predictions;
- suatu kelas tidak mempunyai cukup kelompok independen untuk evaluasi;
- ada record baru yang belum memiliki fitur baseline sebanding;
- dataset stale atau checksum kandidat/model aktif berubah setelah evaluasi.

Jika gate lolos:

```powershell
python -m models.retrain --promote
# Mengevaluasi kembali, memeriksa gate, menyimpan backup model lama,
# lalu mengganti models/fault_classifier.pkl secara atomik.
```

Sesudah promosi, commit model/kode yang sesuai dan deploy/restart proses agar
cache model memuat versi baru. Calibrator dari model berbeda diabaikan; fallback
temperature scaling tetap tersedia. Jangan melatih calibrator pada data yang
juga dipakai melatih classifier lalu menyebutnya validasi independen.

Peningkatan CV bukan bukti penyebab fisik untuk setiap kasus. PETIR dan LAYANG
dapat berbagi pola transient dan reclose; label yang dikonfirmasi melalui sumber
di luar rekaman tetap diperlukan untuk memperbaiki kualitas target training.

Referensi metode: [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
dan [LightGBM classifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html).
