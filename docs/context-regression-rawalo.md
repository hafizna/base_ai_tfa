# Rawalo: konteks satu COMTRADE lengkap

Review 11 Oktober 2026, berdasarkan CFG/DAT dan laporan yang diberikan pengguna.
Anotasi review dipisahkan dari kode inference di
[`config/context_annotations.jsonl`](../config/context_annotations.jsonl).
Tidak ada kondisi khusus nama Rawalo di detector.

## Urutan yang harus terbaca

1. Gangguan awal S-N; trip awal satu pole.
2. Arus episode pertama padam. Pemutusan arus pertama tidak berarti penyebab
   gangguan fisik sudah hilang.
3. A/R Close memerintahkan penutupan; kontak PMT mengonfirmasi penutupan kembali.
4. Arus gangguan muncul kembali dalam episode kedua.
5. SOTF/TOR Trip bekerja setelah reclose dan PMT membuka kembali.

Penutupan mekanis terjadi, tetapi pemulihan sistem tidak bertahan. Ini berbeda
dari reclose berhasil dengan beban normal yang bertahan. Urutan tersebut tidak
dengan sendirinya membuktikan PETIR, HEWAN, atau penyebab fisik lain.

Pembacaan file asli setelah perbaikan:

| Observasi | Hasil |
|---|---|
| Jumlah episode | 2 |
| Durasi episode awal | 68,2 ms |
| Durasi episode berikutnya (waveform) | sekitar 77,3 ms |
| Dead time dari kontak PMT | 1.051,6 ms |
| Kontak PMT menutup | sekitar 1.599,1 ms pada time axis rekaman |
| SOTF/TOR Trip aktif | sekitar 1.610,8 ms pada time axis rekaman |
| Hasil pemulihan | gagal / refault setelah reclose |

Onset waveform yang dihitung dengan window dapat mendahului transisi kontak
digital sekitar satu siklus. Waktu arus kembali dan kontak bantu tidak dianggap
sebagai pengukuran instantaneous yang identik.

## Kesalahan yang dikoreksi

- `52-A` tidak dikenali sebagai `52A`, sehingga kontak closed-state justru dibaca
  sebagai open-state. Open/close terbalik menghasilkan dead time palsu sekitar
  52 ms dan klaim AR berhasil.
- Clearing SOTF yang lebih belakangan ikut digabung ke fault awal, sehingga
  dead time dan dua fault sempat dibaca sebagai satu gangguan sekitar 1,17 s.
- Windowed waveform onset pada fault kedua beberapa sampel lebih awal dari
  current-return edge, sehingga summary refault sempat hilang.
- Rule gangguan permanen sempat menghasilkan ranking penyebab buatan. Kini
  kesimpulan kejadian dipisahkan dari penyebab yang belum dapat ditentukan.

## Verifikasi dan status training

Regression memakai waveform sintetis dengan jawaban yang diketahui untuk
dua fault, closed-state `52-A`, reclose command tanpa contact return, serta
SOTF setelah reclose. File Rawalo asli juga dianalisis langsung. Workspace dan
PDF menampilkan konteks sebelum penyebab; diagram laporan mencakup kedua episode.

Korpus penyebab lama diekstrak ulang: 493 entri inventaris, 462 pasangan unik
yang dapat diidentifikasi, 433 rekaman yang dapat dianalisis, dan 394 yang lolos
filter model pada reader final. Dataset yang sama tidak diperbesar oleh raw
upload duplikat. Promosi 7 kelas ditolak karena PERALATAN hanya mempunyai satu
kelompok kejadian independen pada subset evaluasi.

Dataset konteks memuat satu review Rawalo: jumlah episode, kelas sequence,
restoration outcome, dan SOTF-after-reclose. Ini belum cukup untuk melatih dan
memvalidasi model konteks; tidak ada bobot konteks baru yang diklaim terlatih.
Engine pembacaan/penalaran telah diperbaiki dan pipeline feedback konteks siap
menerima review kasus lain. Lihat [`TRAINING_PIPELINE.md`](../TRAINING_PIPELINE.md).
