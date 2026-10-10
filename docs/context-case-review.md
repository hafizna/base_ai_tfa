# Review konteks rekaman — 11 Oktober 2026

Ketiga pekerjaan review selesai: reclose bertahan, rekaman lengkap versus
capture terpisah, dan jalur PUTT/POTT. Seleksi menghasilkan 60 kandidat unik;
55 dapat dibaca, lima gagal parsing. Kandidat dipilih dari korpus lama, bukan
upload baru. Hasil seleksi otomatis tidak dijadikan ground truth.

## Reclose yang bertahan

| Rekaman | Bukti PMT menutup | Setelah close hingga akhir rekaman | Hasil review |
|---|---|---|---|
| Rawalo–Purbalingga 2, 26 Agustus 2023 | Kontak A kembali 1566,04 ms; Any Pole Dead reset 1567,71 ms | 930,63 ms; arus kembali, tidak ada fault/trip/SOTF berikutnya | Berhasil selama interval yang terekam |
| Rawalo–Purbalingga 1, 11 Desember 2024 | Kontak C kembali 1471,08 ms; Any Pole Dead reset 1472,74 ms | 1205,35 ms; tidak ada fault/trip/SOTF berikutnya | Berhasil selama interval yang terekam |

Kedua label CONFIRMED hanya menyatakan hasil dalam rekaman. Kondisi operasi
setelah file berakhir memerlukan SOE atau informasi operasi lain. Receive
distance yang datang sesudah trip Z1 bukan penyebab trip Z1 tersebut.

## Rekaman lengkap dibanding tiga capture

Rawalo `pbg 2.CFG` dipartisi pada 1,20 dan 1,55 detik. Sampel dan jarak waktu
dipertahankan; percobaan ini tidak menambah contoh training.

Sebelum perbaikan, insiden rekaman lengkap disebut `SINGLE_FAULT_FAILED_RECLOSE`;
tiga capture disebut `MULTIPLE_EPISODES_MIXED_RELATIONSHIP`. Capture terakhir
bermula saat dead time, sehingga waveform tidak memiliki referensi sehat untuk
menentukan onset refault. SOTF sesudah close tetap terlihat.

Sesudah perbaikan, keduanya memberikan `RECLOSE_REFAULT_SOTF`, pemulihan gagal,
dan dua episode gangguan. Penutupan mekanis tetap dibuktikan kontak/status,
bukan hanya perintah AR. SOTF Enable tidak dihitung sebagai SOTF Trip.
Timeline rekaman lengkap menampilkan kedua fault; capture dead time tidak
diberi inception palsu di t=0. Narasi insiden memakai konteks SOTF yang sama.

Jumlah grup capture tetap dapat berbeda: satu grup pada rekaman lengkap dan
dua pada percobaan tiga potongan. Grup capture bukan jumlah fault fisik.
Pada potongan tanpa referensi waveform sehat, SOTF menjadi bukti refault,
tetapi onset/FCT refault dibiarkan tidak terukur. Pada rekaman lengkap,
onset waveform dapat mendahului perubahan kontak sekitar satu siklus.

## PUTT/POTT dan implikasi urutan

| Kasus | Urutan yang direview | Kesimpulan |
|---|---|---|
| Cibatu–Mekarsari 2, 26 April 2024 | Z1/Send distance diam; Z2 3506,93 ms → Receive + trip 3516,93 ms | Kemungkinan PUTT; trip 10 ms setelah Z2 mendukung percepatan, bukan timer Z2 |
| Ungaran–Krian, 20 Mei 2022 | Z2 C 489,98 ms → Aid 1 distance trip + Receive 498,73 ms; Aid 1 Send diam | Kemungkinan PUTT; Send Aid 2 tidak membuktikan POTT pada jalur Aid 1 |
| Majenang–Banjar 1, 7 Mei 2025 | Send distance 795,64 ms → Receive + Z2 trip 808,98 ms | Kemungkinan POTT; Send mendahului Receive/trip 13,34 ms |
| Parakan–Kadipaten 1, 23 Januari 2023 | DEF Receive saat trip 3469,54 ms; distance Receive baru 3624,29 ms | Label POTT lama ditarik; receive DEF tidak membuktikan izin distance |

Skema mendapat confidence PROBABLE, bukan CONFIRMED. Pemetaan kanal dan setting
relay belum tersedia. Keheningan Send mendukung PUTT hanya jika kanal memang
terhubung. Trip cepat tanpa Receive belum membuktikan blocking/DCB. Pembeda
PUTT/POTT adalah sumber pengiriman izin underreach/overreach, sesuai
[penjelasan teknis SEL](https://selinc.com/api/download/6382/).

## Dampak pada training

Lima anotasi ditambahkan pada Rawalo gagal yang sudah ada: total enam rekaman
unik, tanpa menggandakan raw/capture. Bukti ringkas disimpan di
[`config/context_case_reviews.json`](../config/context_case_reviews.json);
target eksplisit di [`config/context_annotations.jsonl`](../config/context_annotations.jsonl).

Fitur kini menyimpan urutan Send → Receive → Trip sebagai selisih waktu dengan
indikator ketersediaan. Sinyal hilang berbeda dari dua sinyal yang simultan.
Receive/Send mengikuti jalur yang sama dengan reasoning, bukan digabung dari
semua kanal. Tujuh kelas penyebab tetap dipertahankan.

Dataset konteks dibangun ulang dan gate evaluasi dijalankan. Belum ada kandidat
bobot: outcome gagal mempunyai satu kelompok, berhasil dua; POTT satu, PUTT
dua. Jalur trip terlabel baru memiliki satu outcome (`accelerated`). Minimal
dua kelompok independen per outcome dan dua outcome tetap berlaku. Perbaikan
yang langsung berlaku berasal dari pembaca/aturan; anotasi ini menjadi target
untuk pembelajaran konteks berikutnya.

## Mengulang review

```powershell
python -m models.review_context_cases --limit 40
python -m models.review_context_cases --path 'path/to/case.cfg'
python -m models.context_capture_comparison 'path/to/pbg 2.CFG' --cuts 1.2 1.55
python -m models.context_training --build
```

Output rinci lokal berada di `data/training-runs/` dan tidak dikomit sebagai
raw/dataset baru. Uji regresi mencakup isolasi DEF/distance/Aid, urutan fitur,
SOTF Enable versus Trip, dan kesetaraan konteks complete/partitioned.
