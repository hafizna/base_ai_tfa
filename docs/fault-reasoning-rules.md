# Rantai reasoning per gangguan — katalog aturan

> **Status: disepakati 9 Oktober 2026.** Kode dan tes merujuk ke ID aturan yang sama.
>
> Implementasi dikerjakan bertahap, satu PR per tahap:
> 1. penelusuran sinyal analog per rekaman;
> 2. rekombinasi rekaman insiden di satu sumbu waktu;
> 3. rantai aturan dan Detail teknis;
> 4. laporan TWS dan DE-FL.
>
> Yang sudah masuk: sinkronisasi DE-FL (F2.1, PR #35).
>
> Fasa ditulis dengan nama PLN R, S, T. Di kode namanya A, B, C.

## Kenapa dokumen ini ada

Sekarang satu kesimpulan yang sama (fasa terganggu, FCT, zona) dihitung di tiga sampai empat tempat dengan aturan berbeda, dan hasilnya saling bertentangan. Contoh dari dua rekaman nyata:

| Kesimpulan | Jalur sekarang | Cibatu–Mekarsari 2 (aktual R-N) | Bringin ZQ6D (aktual S-T) |
|---|---|---|---|
| Fasa | Narasi AI (`ml_predict.extract_ml_features`) | R+T-N ✗ | S-T ✓ |
| Fasa | Panel Jenis Gangguan (`relay_21._evidence_based_fault_phases`) | R-N ✓ | R-S-T (3Ph) ✗ |
| Fasa | Event window (`core/fault_detector`) | R ✓ | S-T ✓ |
| FCT | Event window, dipakai halaman Insiden | 53 ms ✗ | 65 ms ✗ |
| FCT | Narasi AI / Parameter Elektrikal | 92 ms | 115 ms ✗ |
| FCT aktual | Arus fasa terganggu benar-benar berhenti | ±82 ms | ±77 ms |
| Zona | Narasi AI | "Zona Z2 bekerja mentrigger TRIP" ✗ | "Zona Z1 …" ✓ |

Dokumen ini menetapkan **satu rantai reasoning** berisi sembilan langkah berurutan. Tiap langkah menghasilkan satu kesimpulan dari aturan yang eksplisit. Semua tampilan membaca hasil yang sama: halaman relay 21, narasi AI, halaman Insiden, dan PDF.

## Cara membaca

Setiap aturan punya ID (mis. **F4.5**) dan status:

- **VALID**: sudah benar, dipertahankan.
- **PERBAIKI**: idenya benar, implementasinya salah.
- **GANTI**: aturannya sendiri tidak valid, jadi dihapus atau diganti.
- **BARU**: belum ada di aplikasi.
- **KONFIRMASI**: butuh keputusan Anda.

Setiap kesimpulan yang dihasilkan rantai ini membawa lima hal:
- nilainya;
- ID aturan yang dipakai;
- buktinya (kanal, waktu, besaran);
- tingkat keyakinan;
- konflik, bila ada.

## Prinsip

- **P1. Fakta dulu, tafsiran kemudian.** Langkah 1–7 hanya memakai fakta terukur: kanal status, tegangan, arus, impedansi. Penyebab (langkah 8) dibangun di atasnya dan tidak boleh mengubah fakta.
- **P2. Satu kesimpulan, satu tempat hitung.** `build_event_window` sejak awal dimaksudkan sebagai sumber tunggal (lihat docstring-nya), tapi `ml_predict` dan `relay_21` menghitung ulang. Rantai ini menutup celah itu.
- **P3. Urutan kekuatan bukti:** keputusan relay (kanal status), lalu fisika gelombang (V, I, Z), lalu statistik (AI). Keputusan relay bisa salah (maloperasi). Jadi bila fisika bertentangan dengan relay, konfliknya ditampilkan, bukan dipilih diam-diam.
- **P4. Kanal yang direkam tapi tetap 0 adalah bukti. Kanal yang tidak direkam berarti "tidak diketahui".**
  - Contoh: `DIST Sig. Send` direkam dan tidak pernah aktif, jadi itu bukti untuk PUTT.
  - Bila kanal Send tidak direkam, skemanya tidak dapat ditentukan.
- **P5. Konflik ditampilkan.** Kesimpulan yang bertentangan tidak dirata-rata.
- **P6. Keyakinan:**
  - tinggi bila dua sumber independen atau lebih sepakat;
  - sedang bila hanya satu sumber;
  - rendah bila bukti bertentangan atau data kurang.

## Langkah 1 — Apakah ada gangguan?

**F1.1 — Gerbang no-fault.** **VALID.** Gangguan dianggap ada bila salah satu terpenuhi:
- proteksi beroperasi;
- ada lonjakan arus (peak ≥2× dan RMS ≥1,5× prefault);
- ada sag ≥25% disertai unbalance kuat.

Kode: `webapp/api/fault_detection.py`.

**F1.2 — Rekaman reclose.** **VALID.** Rekaman yang dimulai saat PMT terbuka (dead time) hanya menangkap reclose, jadi penyebab tidak diklasifikasi dari rekaman ini. Kode: `ml_predict._reclose_capture_gate`.

## Langkah 2 — Line mana yang terganggu?

**F2.1 — DFR dua line dalam satu CFG.** **VALID.** Line yang terganggu dipilih dari bukti. Kode: PR #29, `core/line_selection.py`.
- Aturan ini harus dipakai di semua jalur.
- Grafik sinkronisasi di halaman DE-FL dulu belum memakainya: halaman itu mengambil kanal `IA` pertama di tiap rekaman. Untuk Bringin + Mojosongo, yang terambil adalah line yang padam (MJSNG1 dan BRINGIN 1). **Diperbaiki di PR #35.**
  - Grafik kini menampilkan fasa terganggu yang step-nya paling jelas di kedua ujung, pada line yang terganggu.
  - Estimasi geser kini menyejajarkan inception kedua rekaman, lalu memeriksanya dengan jam. Jangkar jam memakai sampel pertama, dan selisih zona waktu (Qualitrol UTC) dibuang sebelum dibandingkan.

**F2.2 — Dua sirkit terganggu bersamaan.** **BARU (nanti).** Kedua line dianalisa. Langkah ini dibutuhkan F8.3 sebagai indikasi BFO.

## Langkah 3 — Kapan gangguan mulai dan kapan padam?

**F3.1 — Inception.** **VALID.** Memakai event window kanonik, yang menyelaraskan kanal status dengan gelombang.

**F3.2 — FCT = inception sampai arus gangguan benar-benar padam.** **GANTI.** Dua implementasi sekarang salah:
- **Event window** memakai lebar pulsa kontak trip (trip naik sampai trip turun) sebagai durasi gangguan.
  - Cibatu: 90,0 − 36,7 = 53,3 ms. Bringin: 109,9 − 45,1 = 64,8 ms.
  - Itu durasi kontak trip, bukan FCT. Kode: `core/fault_detector._detect_from_status_channels`.
- **`ml_predict` dan `relay_21`** memakai ambang RMS 0,6 × max(prefault, 5% puncak) pada satu fasa.
  - Di Bringin, ekor arus CT yang meluruh setelah PMT membuka (CT subsidence) ikut terhitung, sehingga hasilnya 115 ms.

**Aturan baru.** Definisinya sama dengan Grid Code (Permen ESDM 20/2020, CCA1 2.2): waktu pemutusan gangguan dihitung "mulai dari saat terjadi gangguan sampai dengan padam busur listrik oleh terbukanya PMT". Titik padam adalah zero crossing terakhir arus fasa terganggu sebelum magnitudo fundamental 50 Hz-nya (DFT satu siklus) jatuh di bawah ±10% arus gangguan dan bertahan di bawah itu minimal satu siklus.
- PMT memutus di zero crossing.
- Ekor CT subsidence tidak punya zero crossing 50 Hz, jadi tidak ikut terhitung.

**F3.3 — Waktu interupsi PMT.** **BARU.** Selisih padam − trip adalah waktu interupsi PMT, wajarnya ±1,5–4 siklus (30–80 ms).
- Di luar rentang itu, kesimpulannya ditandai. Kemungkinannya: PMT lambat, pole macet, atau kanal trip bukan output relay.
- Cibatu 45 ms ✓, Bringin 32 ms ✓.

**F3.4 — Urutan kontak bantu PMT.** **BARU.** Kontak bantu PMT (CB Aux) berubah setelah arus padam. Dipakai untuk cek urutan, bukan untuk menghitung FCT. Cibatu: padam ±82 ms, CB Aux R +86,6 ms ✓.

**F3.5 — Batas FCT proteksi utama.** **BARU.** Sumber: Grid Code, Permen ESDM 20/2020, CCA1 2.2.

| Tegangan | FCT maksimum |
|---|---|
| 500 kV | 90 ms |
| 275 kV | 100 ms |
| 150 kV | 120 ms |
| 66 kV | 150 ms |

- Batas ini berlaku untuk trip proteksi utama: Z1, aided trip, dan 87L. FCT yang melewati batas ditandai.
- Trip waktu tunda Z2/Z3 adalah proteksi cadangan, jadi batas ini tidak berlaku. Pasal yang sama mengatur proteksi cadangan sisi pemakai jaringan <400 ms, dan CBF men-trip PMT sekitarnya dalam 200–250 ms.
- Cibatu 82 ms ✓, Bringin 77 ms ✓.

## Langkah 4 — Fasa mana yang terganggu, dan apakah ke tanah?

**F4.1 — Ke tanah bila I0/I1 > 0,2.** **VALID** (sudah ada). Catatan: BFO dua fasa pada satu tower dengan tahanan kaki tinggi bisa memberi I0 kecil, sehingga terbaca seperti LL.

**F4.2 — Fasa terganggu dari tegangan.** **BARU** sebagai bukti utama.
- Fasa terganggu adalah fasa yang tegangannya jatuh jelas lebih dalam dari fasa lain, mis. <0,8 pu sementara yang lain ≥0,9 pu.
- Untuk gangguan jauh dengan sag dangkal, pakai urutan relatif antarfasa.
- Cibatu: R 0,46 / S 0,91 / T 0,90 pu. Bringin: R 0,97 / S 0,66 / T 0,64 pu.

**F4.3 — Impedansi loop.** **PERBAIKI.**
- Loop yang melihat gangguan punya |Z| terkecil; loop sehat 2–3× lebih besar atau lebih.
- Loop fasa-fasa selalu dibandingkan. Loop fasa-tanah (dengan K0) hanya dipakai bila F4.1 menyatakan ada arus tanah, seperti relay yang baru melepas loop tanah bila 3I0 cukup besar.
- Sekarang hanya loop fasa-tanah yang dipakai, di panel Jenis Gangguan.
- Cibatu: loop R-N 2,2 Ω, loop lain ≥9 Ω.
- Bringin: loop S-T 6,3 Ω, sedangkan R-S 20,5 Ω dan T-R 26,6 Ω.

**F4.4 — Phase selection relay.** **VALID.**
- Bukti fasa diambil dari kanal start/phase-select per fasa, atau dari trip 1-pole fasa X.
- Trip 1-pole fasa X berarti gangguan X-N, karena relay hanya trip 1-pole untuk gangguan satu fasa.
- Ini override single-pole yang sedang dikerjakan di checkout utama.

**F4.5 — Trip 3-pole tidak membawa informasi fasa.** **GANTI.**
- Fasa yang trip pada trip 3-pole tidak boleh dihitung sebagai bukti fasa terganggu.
- Sekarang panel Jenis Gangguan menghitungnya, sehingga Bringin yang S-T terbaca R-S-T (3Ph).

**F4.6 — "Arus fasa naik di atas 3× prefault (atau >10% puncak) berarti fasa terganggu."** **GANTI.**
- Saat gangguan satu fasa, fasa sehat ikut membawa arus karena pembagian arus urutan nol dan positif tidak sama.
- Cibatu: arus T 2,1 kA (16% arus R), padahal tegangannya 0,90 pu.
- Aturan ini dipakai narasi AI, sehingga muncul "R+T-N (DLG)" dan mekanisme BFO di atasnya.

**F4.7 — Cek konsistensi arus.** **BARU** (aturan dari Anda).
- Gangguan dua fasa membuat arus kedua fasa sama-sama tinggi. Untuk LL, arusnya bahkan sama besar dan berlawanan arah.
- Bringin: S 8,4 kA, T 7,6 kA, beda sudut 180° ✓.
- Bila satu fasa jauh lebih kecil (Cibatu: T = 16% R), itu bukan gangguan dua fasa.

**F4.8 — Konflik gelombang vs relay.** **BARU.** Kasus yang ditandai:
- gelombang menunjukkan multi-fasa tapi relay trip 1-pole;
- atau sebaliknya.

Kemungkinannya: maloperasi, gangguan berkembang, atau masalah CT/VT.

**F4.9 — Gangguan berkembang.** **BARU (nanti).** Fasa yang terlibat berubah dalam satu kejadian (mis. R-N menjadi R-S-N), jadi fasa dievaluasi per jendela waktu.

## Langkah 5 — Proteksi apa yang trip, dan lewat jalur apa?

**F5.1 — Pickup ≠ trip.** **BARU.**
- Kanal zona yang aktif berarti elemen itu melihat gangguan.
- Elemen yang men-trip ditentukan dari waktu dan sinyal lain (F5.2–F5.5).
- Aturan ini menggantikan aturan lama yang menulis "Zona Zx bekerja mentrigger TRIP" untuk zona mana pun yang pernah aktif.

**F5.2 — Waktu trip dibandingkan dengan timer zona.** **BARU.** Nilai PLN dari Anda:

| Elemen | Waktu trip normal, dihitung dari zona start |
|---|---|
| Z1 | Seketika (waktu relay ±20–50 ms) |
| Z2 | 0,4 s atau 0,8 s |
| Z3 | 1,2 s atau 1,6 s |

Bila file setting (RIO) tersedia, timer diambil dari file itu. Toleransinya **±10%** dari timer (disepakati 9 Okt 2026): Z2 0,4 s → ±40 ms, Z2 0,8 s → ±80 ms, Z3 1,2 s → ±120 ms, Z3 1,6 s → ±160 ms.

**F5.3 — Hanya Z2 yang dipercepat teleproteksi.** **BARU** (dari Anda).
- Z3 tidak pernah dipercepat teleproteksi.
- Trip jauh lebih cepat dari timer Z2, sementara hanya Z2 yang pickup (Z1 tidak), berarti trip dipercepat.
- Jenis percepatannya ditentukan F5.4 dan F5.5.

**F5.4 — Skema dibaca dari rekaman, bukan diasumsikan per bay.** **BARU.** Skema bisa PUTT, POTT, atau blocking, tergantung kondisi teleproteksi dan sistem, jadi pembacaannya adaptif:

| Yang terlihat di rekaman (Z2 pickup, Z1 tidak) | Pembacaan |
|---|---|
| Trip sebelum timer Z2; ada receive saat trip; Send lokal direkam tapi tidak pernah aktif | Aided trip, **PUTT** |
| Trip sebelum timer Z2; ada receive; Send lokal aktif saat Z2 pickup | Aided trip, **POTT** |
| Trip sebelum timer Z2; kanal receive direkam tapi tidak aktif; trip menyusul Z2 pickup dengan jeda pendek (puluhan ms) | **Blocking (DCB)**: tidak ada sinyal blok, jadi trip diizinkan. Bisa juga unblocking dengan loss-of-guard. |
| Trip sebelum timer Z2, tepat setelah PMT menutup (reclose/close) | **SOTF/TOR** (F5.5) |
| Trip sebelum timer Z2; kanal receive/send tidak direkam | Trip dipercepat; skema **tidak dapat ditentukan** |
| Trip ≈ 0,4/0,8 s setelah Z2 pickup | **Z2 waktu tunda**: teleproteksi tidak membantu atau tidak ada |
| Z2 pickup lalu hilang **tanpa trip lokal** | Gangguan di luar line, di balik GI lawan, dan diputus proteksi lain. Z2 hanya standby. |

Catatan untuk pola "Z2 aktif lalu tiba-tiba hilang":
- Z2 hilang karena arus gangguan berhenti. Artinya gangguan padam sebelum timer Z2.
- Pola ini belum menentukan skema. Yang menentukan adalah ada tidaknya trip lokal, receive, dan send (tabel di atas).

**F5.5 — SOTF/TOR.** **BARU.**
- Setelah PMT menutup (manual atau reclose), Z2/Z3 atau elemen SOTF bisa men-trip seketika tanpa receive. Trip seperti itu dibaca sebagai SOTF/TOR, bukan teleproteksi.
- Aturan ini penting untuk re-fault setelah reclose.
- Bila kanal SOTF/TOR direkam (mis. `SOTF/TOR Trip` di rekaman MiCOM), kanal itulah yang dipakai.
- Bila kanal itu tidak ada, dipakai pembacaan waktu. Trip yang terjadi ≤1 s setelah PMT menutup, lebih cepat dari timer zonanya, dan tanpa receive dibaca "kemungkinan TOR".
  - Batas 1 s adalah default. Di MiCOM P443, lama mode ini aktif diatur setting `TOC Reset Delay`; nilai dari file setting dipakai bila tersedia.
- Konsistensi: trip SOTF/TOR selalu 3-pole dan memblok AR, jadi tidak ada reclose sesudahnya.
- Rekaman Cibatu punya kanal `SOTF/TOR Trip`, dan kanal itu tidak aktif.

**F5.6 — Zona bersarang.** **BARU** (prinsip dari Anda).
- Zona forward bersarang: Z1 ⊂ Z2 ⊂ Z3. Bila start Z2 aktif, start Z3 seharusnya ikut aktif, asal tiga syarat terpenuhi:
  1. Z3 forward (tidak reverse, offset, atau disabled);
  2. Z3 menutupi Z2 di seluruh karakteristik, termasuk jangkauan resistif;
  3. kanal "Z3" di DFR adalah sinyal start, bukan output trip.
- Bila Z3 tidak ikut aktif, kesimpulannya ditandai "perlu dicek" beserta kemungkinan yang masih tersisa:
  - Z3 diset reverse ("zona 3 belakang"), offset, atau disabled;
  - jangkauan resistif Z3 dipotong load blinder, atau karakteristiknya berbeda (mis. Z2 quad, Z3 mho). Kemungkinan ini gugur bila impedansi loop kecil (gangguan tidak resistif);
  - kanal Z3 adalah output trip, yang baru aktif setelah timer 1,2/1,6 s;
  - relay hanya mengeluarkan indikasi zona eksklusif ("gangguan di zona X"), bukan start tiap zona;
  - kanal Z3 tidak dipetakan.
- Bila file setting (RIO) tersedia, arah dan jangkauan Z3 dibaca dari sana, sehingga kemungkinannya bisa dipersempit.
- Cibatu: Z2 pickup, Z3 dan Z4 tidak.
  - Loop R-N hanya 2,2 Ω, jadi penjelasan resistif/blinder gugur.
  - Z4 tidak pickup itu wajar, karena di MiCOM Z4 umumnya reverse.

**F5.7 — Implikasi lokasi dari zona.** **BARU.**
- Bila Z1 lokal tidak pickup dan terjadi aided trip, gangguan ada di luar jangkauan Z1 lokal (umumnya 80% panjang line), dekat GI lawan.
- GI lawan seharusnya melihatnya di Z1 (PUTT) atau di zona overreach (POTT).
- Ini dikonfirmasi dengan rekaman sisi lawan (F7.3).

**F5.8 — Analisa 87L hanya bila rekaman punya bukti 87L.** **GANTI.**
- Bukti 87L berupa kanal operate diferensial atau arus ujung remote.
- Sekarang panel 87L muncul karena pilihan menu LINE, bukan karena isi rekaman.

## Langkah 6 — Mode trip dan reclose

**F6.1 — Mode trip.** **VALID.** Dibaca dari status pole PMT (CB Aux per fasa) dan kanal trip per fasa.

**F6.2 — Ekspektasi pola trip dan AR.** **BARU** sebagai aturan ekspektasi. Dasarnya SPLN T5.002:2021 dan praktik PLN:
- Gangguan satu fasa-tanah: trip 1-pole, lalu SPAR single shot (dead time ±0,9–1 s).
- Gangguan multi-fasa: trip 3-pole, lalu TPAR atau lockout sesuai setting bay. Bringin memakai reclose 3-pole setelah 5,0 s.
- SUTT/SUTET yang tersambung ke pembangkit memakai **SPAR single shot** (SPLN T5.002:2021).
- Grid Code (CCA1 2.3.1) mewajibkan setiap proteksi utama di terminal SUTT mampu tripping dan reclosing 1 fasa dan 3 fasa. Reclosing 3 fasa harus melalui synchro check. Jadi bay mana pun bisa memakai SPAR atau TPAR, tergantung setting-nya.
- A/R 3 fasa kecepatan tinggi di GI pembangkit atau di dekatnya hanya dipakai setelah dipastikan aman bagi poros dan belitan generator. Reclose dilakukan berurutan, dimulai dari PMT yang jauh dari pembangkit.
- AR di-inisiasi oleh trip seketika atau aided (Z1, Z1+aided, DEF+aided), bukan oleh trip waktu tunda Z2/Z3.

Penyimpangan dari ekspektasi ini ditandai "cek setting AR bay ini", bukan dianggap kesalahan.

**F6.3 — Konsistensi trip/reclose.** **BARU.** Kasus yang ditandai:
- trip 1-pole untuk gangguan multi-fasa (konflik);
- trip 3-pole untuk gangguan satu fasa di bay SPAR. Kemungkinannya: AR tidak siap atau diblok, gangguan berkembang, atau bay TPAR.
- reclose setelah trip waktu tunda Z2/Z3.

**F6.4 — Dead time.** **VALID** (PR #30). Diukur dari PMT buka sampai PMT tutup dengan kontak stabil; bounce diabaikan.

**F6.5 — Reclose berhasil.** **VALID** (PR #30). PMT menutup, line bertegangan, dan tidak ada arus gangguan.

**F6.6 — Re-fault setelah reclose.** **VALID** (PR #30). Gangguan 0,5–60 s setelah reclose berhasil. Tambahan: trip kedua dicek terhadap F5.5 (TOR).

## Langkah 7 — Di mana lokasinya?

**F7.1 — Lokasi single-ended dari impedansi.** **VALID**, dengan catatan: akurasinya turun karena tahanan gangguan dan infeed dari ujung lain.

**F7.2 — Implikasi zona.** **BARU.** Lihat F5.7.

**F7.3 — Rekaman dua sisi dalam satu insiden.** **BARU** sebagai aturan insiden. Dua hal dilakukan:
- lokasi dua ujung (DE-FL);
- konfirmasi skema dari sisi lawan, mis. Z1 + Send untuk PUTT.

DE-FL sendiri sudah ada di halaman terpisah.

**F7.4 — DE-FL hanya sah pada loop yang membawa arus gangguan.** **BARU**, karena ditemukan saat menguji Bringin + Mojosongo.
- Halaman DE-FL memilih loop dari klasifikasi fasa lama. Bringin S-T terbaca 3 fasa (F4.5), sehingga loop yang dipilih **ZA**, padahal fasa R sehat.
- Hasilnya 19,79 km dengan "arus gangguan" 0,18 kA, yang sebenarnya arus beban. KVL residual-nya 0,001, sehingga tampak andal.
- Penyebabnya: pada loop yang hanya dilalui arus beban, persamaan dua ujung terpenuhi oleh jarak berapa pun. Residual kecil di loop seperti itu tidak berarti hasilnya benar.
- Dengan loop yang benar (ZBC), hasilnya 16,71 km dengan arus 14,5 kA, tapi residual-nya 0,29 karena kedua rekaman belum sinkron.
- Aturan baru:
  - loop diambil dari langkah 4;
  - arus loop harus jelas di atas arus beban (mis. ≥3× prefault), kalau tidak hasilnya ditolak;
  - sinkronisasi waktu kedua rekaman harus terkonfirmasi sebelum hasil dilaporkan.
- Setelah PR #35, geser yang benar untuk pasangan ini adalah −50,5 ms, dan jam kedua DFR sepakat dalam 0,02 ms. Residual loop S-T tetap 0,26–0,37, karena:
  - panjang line dan R1/X1 masih nilai uji;
  - Mojosongo adalah ujung weak infeed: arus gangguannya hanya 0,4–0,8 kA, setara arus beban, sementara Bringin ±6 kA;
  - arus Mojosongo berhenti ±20 ms lebih dulu daripada Bringin.
- Aturan tambahan **BARU**:
  - ujung dengan kontribusi arus gangguan < 2× beban ditandai weak infeed, dan keandalan DE-FL diturunkan;
  - jendela evaluasi hanya dipakai selama kedua ujung masih mengalirkan arus gangguan.

## Langkah 8 — Apa penyebabnya?

**F8.1 — AI per rekaman.** **VALID.** Ini bacaan statistik, dilaporkan apa adanya, dan tidak pernah mengubah fakta langkah 1–7.

**F8.2 — Pola urutan.** **VALID** sebagai bukti pola dengan kekuatan sedang. Re-fault di fasa yang sama setelah reclose berhasil menunjukkan kontak fisik (pohon atau benda asing).

**F8.3 — Sub-mekanisme petir (SF/BFO).** **GANTI.**
- **Yang dihapus:** perbandingan arus gangguan 50 Hz di DFR dengan batas arus sambaran model EGM (`models/predict._classify_petir_subtype`). Arus gangguan ditentukan kekuatan sumber dan lokasi gangguan, bukan arus sambaran.
- **Tanpa LDS:** hasilnya "Sub-mekanisme tidak dapat ditentukan tanpa data LDS", ditambah indikasi dari COMTRADE:
  - flashover multi-fasa ke tanah, atau dua sirkit terganggu bersamaan (F2.2), menjadi indikasi BFO;
  - flashover satu fasa bisa SF atau BFO.
- **Dengan LDS (input opsional):** LDS digabung dengan pola arus dari COMTRADE.
  - Isian per sambaran, mengikuti format keluaran jaringan deteksi petir:
    - waktu UTC (presisi µs);
    - lintang dan bujur;
    - arus puncak dalam kA, bertanda sesuai polaritas;
    - jenis: CG (ke tanah) atau IC (awan);
    - elips galat 50%: sumbu semi-mayor, sumbu semi-minor, orientasi;
    - multiplisitas;
    - jumlah sensor.
  - Pencocokan:
    - jendela waktu ±1 s dari inception, setelah koreksi jam DFR. Studi korelasi gangguan line dengan data petir mendapati ±90% kejadian terkorelasi berada dalam ±1 s.
    - jarak ke koridor line sampai 5 km.
    - kandidat diurutkan menurut selisih waktu, lalu jarak.
    - Bila jam DFR belum sinkron GPS, selisih jam harus diisi dulu. Tanpa itu, pencocokan waktu ditandai "tidak dapat dipastikan".
  - Penilaian: arus puncak sambaran dibandingkan dengan batas SF (EGM) dan arus kritis BFO. Arus kritis BFO butuh data CFO isolator dan tahanan kaki tower. Bila data tower belum ada, hasilnya kualitatif: SF hanya mungkin untuk sambaran berarus kecil, sedangkan BFO butuh sambaran berarus besar.

**F8.4 — "Sudut inception dekat puncak tegangan berarti tipikal petir."** **GANTI** menjadi bukti lemah yang tidak khas petir. Kegagalan isolasi karena sebab apa pun (pohon, kontaminasi) juga cenderung terjadi dekat puncak.

**F8.5 — Aturan Tier-1 (`models/rules.py`).**
- **Rule 0, anomali CT/pengukuran:** **VALID**.
- **Rule 1, "tepat dua fasa berarti perubahan fasa saat reclose":** **GANTI**. Perubahan fasa harus dibandingkan antarkejadian (gangguan #1 vs #2), bukan dari jumlah fasa satu kejadian. Gangguan S-T biasa pun memenuhi syarat aturan lama.
- **Rule 2 dan 3, "reclose gagal berarti gangguan permanen":** **VALID** sebagai sifat gangguan, bukan sebagai penyebab.

**F8.6 — Pengali heuristik penyebab.** **GANTI.**
- Lokasinya `models/predict._compute_cause_scores`, dipakai pipeline batch dan tidak dipakai webapp.
- Contohnya: arus >10 kA menaikkan skor petir ×2,2.
- Besar arus mencerminkan kekuatan sumber dan lokasi, bukan penyebab.

## Langkah 9 — Cek konsistensi akhir

Cek silang yang dijalankan setelah langkah 1–8 (semuanya sudah disebut di atas):

- F3.3: selisih padam − trip ada dalam rentang waktu PMT;
- F4.7: pola arus cocok dengan jenis gangguan;
- F4.8: jenis gangguan cocok dengan mode trip relay;
- F5.6: zona bersarang;
- F6.3: mode trip/reclose cocok dengan jenis gangguan dan jalur trip;
- antarrekaman dalam satu insiden: fasa sama atau beda, dan skema dua sisi (F7.3).

## Contoh 1 — Cibatu–Mekarsari 2 (bay MKSRI 2 di GI Cibatu, 26 Apr 2024)

| Langkah | Kesimpulan | Aturan | Bukti |
|---|---|---|---|
| 1 | Ada gangguan | F1.1 | Proteksi operate; arus R 20,7 kA puncak |
| 3 | Padam ±82 ms; PMT ±45 ms | F3.2–F3.4 | Trip +36,7 ms; arus R jatuh pada jendela 80–90 ms; CB Aux R +86,6 ms |
| 4 | R-N | F4.1–F4.4, F4.7 | I0/I1 0,78; V 0,46 / 0,91 / 0,90 pu; loop R-N 2,2 Ω; trip 1-pole R; arus T = 16% R |
| 5 | Aided trip, PUTT | F5.1–F5.4 | Z2 start +26,7 ms. Trip R + Chan Recv +36,7 ms, yaitu 10 ms setelah Z2, jauh di bawah 0,4 s. `Sig. Send` direkam tapi tidak pernah aktif; Z1 tidak pickup |
| 5 | Ditandai: Z3 tidak pickup | F5.6 | Z2 aktif, Z3 dan Z4 tidak |
| 5 | Gangguan dekat GI Mekarsari | F5.7 | Z1 Cibatu tidak pickup |
| 6 | Trip 1-pole R, SPAR, dead time 1,0 s, berhasil | F6.1–F6.5 | A/R 1P In Prog +36,7 sampai +1029,6 ms; CB Aux R 86,6 sampai 1089,6 ms |
| 8 | AI: Petir 92%. Sub-mekanisme tidak dapat ditentukan tanpa LDS | F8.1, F8.3 | — |

Yang perlu dicek:
- rekaman sisi Mekarsari (Z1 + Send?);
- setting Z3 Cibatu.

## Contoh 2 — Bringin ZQ6D, gangguan #1 (21 Agu 2023, label POHON)

| Langkah | Kesimpulan | Aturan | Bukti |
|---|---|---|---|
| 2 | Line MJSNG2; MJSNG1 tidak bertegangan | F2.1 | PR #29 |
| 3 | Padam ±77 ms; PMT ±32 ms | F3.2, F3.3 | Trip +45 ms; arus S/T jatuh pada jendela 70–80 ms; ekor CT subsidence sesudahnya tidak dihitung |
| 4 | S-T, tidak ke tanah | F4.1–F4.3, F4.5, F4.7 | I0/I1 0,06; V 0,97 / 0,66 / 0,64 pu; loop S-T 6,3 Ω; IS 8,4 kA dan IT 7,6 kA berlawanan arah; trip 3-pole tidak dipakai sebagai bukti fasa |
| 5 | Z1, seketika | F5.1, F5.2 | TRIP Z1 +48 ms. DFR eksternal tidak merekam kanal teleproteksi, jadi skema tidak dapat ditentukan (dan tidak diperlukan untuk trip Z1) |
| 6 | Trip 3-pole, konsisten dengan multi-fasa. Reclose 3-pole setelah 5,0 s (rekaman ZQ6E) | F6.1–F6.4 | TRIP R/S/T +45–48 ms |
| 8 | AI: Petir 92% (dipotong ke 82% di insiden). Pola re-fault di fasa sama setelah reclose menunjukkan kontak fisik, cocok dengan label POHON. Aturan BFO lama keliru menyimpulkan BFO | F8.1–F8.3 | — |

## Keputusan (9 Okt 2026)

1. **Toleransi timer zona (F5.2):** ±10% dari timer. Disepakati.
2. **Jendela TOR (F5.5):**
   - kanal SOTF/TOR dipakai bila direkam;
   - bila tidak, dipakai default 1 s setelah PMT menutup.
   - Nilai dari setting relay menggantikan default bila tersedia.
3. **Batas FCT (F3.5):** 150 kV ≤120 ms. Sumbernya Grid Code, Permen ESDM 20/2020, CCA1 2.2.
4. **Isian LDS (F8.3):**
   - field standar keluaran jaringan deteksi petir;
   - pencocokan ±1 s dan ≤5 km dari line.

Butir 2 dan 4 diputuskan dari literatur karena tidak ada praktik PLN yang spesifik. Bila ada setting atau format data LDS PLN yang berbeda, nilainya diganti.

## Peta kode sekarang

| Langkah | Dihitung di | Dibaca oleh |
|---|---|---|
| 3 — FCT | `core/fault_detector` (event window); `ml_predict.extract_ml_features`; `relay_21._compute_electrical_params` | Halaman Insiden; narasi AI; Parameter Elektrikal; PDF |
| 4 — Fasa | `ml_predict.extract_ml_features`; `relay_21._evidence_based_fault_phases`; `core/fault_detector`; `core/protection_router` | Narasi AI, Tier-1, BFO; Jenis Gangguan; Insiden |
| 5 — Zona/skema | `ml_predict._digital_sequence_features` + loop fallback zona. `core/protection_router` sudah punya deteksi receive dan skema, tapi tidak dipakai halaman 21. `record_facts.protection_operations` berisi fakta (PR #34) | Narasi AI; Jenis Gangguan; Insiden |
| 6 — Trip/AR | `ml_predict._digital_sequence_features`; `core/fault_detector`; `incidents/relationships` | Narasi AI; Insiden |
| 8 — Penyebab | `ml_predict.run_ml_prediction`; `models/rules.py`; `models/predict._classify_petir_subtype`; narasi/pola insiden | Panel AI; Insiden |

Targetnya satu modul rantai reasoning:
- menghasilkan semua kesimpulan di atas sekali per rekaman;
- hasilnya disimpan di snapshot rekaman dan dibaca semua tampilan;
- narasi disusun dari kesimpulan itu, bukan dihitung ulang.

## Sumber aturan PLN

- Grid Code, Permen ESDM No. 20 Tahun 2020 *Aturan Jaringan Sistem Tenaga Listrik*:
  - **CCA1 2.2:** waktu pemutusan gangguan, dihitung dari gangguan terjadi sampai busur padam oleh terbukanya PMT: 500 kV 90 ms, 275 kV 100 ms, 150 kV 120 ms, 66 kV 150 ms. Proteksi cadangan sisi pemakai jaringan <400 ms. CBF 200–250 ms.
  - **CCA1 2.3.1:** proteksi saluran 150 kV dan 66 kV.
    - Saluran pendek: LCD dengan fungsi distance.
    - Saluran sedang dan panjang: LCD, atau distance dengan transfer trip permissive underreach/overreach, termasuk zone-2 dan zone-3 waktu tunda.
    - Setiap proteksi utama di terminal SUTT mampu tripping dan reclosing 1 dan 3 fasa; reclosing 3 fasa lewat synchro check.
- MiCOM P443 (SOTF/TOR):
  - selama mode SOTF/TOR aktif, zona yang dipilih atau detektor *Current No Volt* men-trip 3-pole seketika dan memblok AR;
  - lama aktifnya diatur `TOC Reset Delay`.
- Data petir: field standar NLDN/LLS, yaitu waktu, lokasi, arus puncak bertanda polaritas, dan elips galat 50%. Jendela korelasi ±1 s berasal dari studi korelasi gangguan line dengan data petir.
- SPLN T5.002:2021 *Pola Proteksi Transmisi*: SUTT/SUTET yang tersambung ke pembangkit memakai SPAR single shot. Dikutip dari deck pelatihan "KS Sistem Proteksi Saluran Transmisi", slide 22. Teks SPLN lengkapnya belum dicek langsung.
- Materi pelatihan "Dasar Sistem Proteksi TT":
  - A/R 3 fasa kecepatan tinggi di GI pembangkit atau di dekatnya hanya dipakai setelah dipastikan aman bagi mesin;
  - reclose berurutan dimulai dari PMT yang jauh dari pembangkit.
- Contoh setting SPAR pada line 150 kV Menggala–Gumawang 2 (makalah analisa auto recloser):
  - di-inisiasi oleh Z1, Z1+aided, dan DEF+aided;
  - dead time 900 ms;
  - reclaim 40 s.
- Timer zona Z1 seketika, Z2 0,4/0,8 s, Z3 1,2/1,6 s: konfirmasi pengguna, 9 Okt 2026.
