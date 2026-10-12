# Baseline performa frontend — 12 Oktober 2026

Tahap pertama mengukur build lokal dan memisahkan loading halaman. Tahap kedua
membatasi runtime Plotly pada trace yang dipakai aplikasi. Pengukuran ini bukan
latency produksi atau benchmark analisis COMTRADE.

## Metode

Build sebelum dan sesudah menggunakan dependency yang sama, dengan
`npm.cmd run build -- --manifest` dari `webapp/frontend`. Jalankan
`node profiling/frontend_bundle.mjs` dari root repo untuk menghitung JavaScript
entry beserta seluruh static imports-nya. Dynamic imports tidak dihitung sebagai
loading awal. Gzip dihitung per file dengan default `node:zlib`, sehingga dapat
berbeda dari estimasi Vite dan konfigurasi kompresi server produksi.

## Hasil

| Metrik JavaScript | Sebelum | Lazy routes | Lazy routes + custom Plotly |
|---|---:|---:|---:|
| Loading awal, byte tanpa kompresi | 5.251.341 | 238.728 | 238.900 |
| Loading awal, byte estimasi gzip | 1.563.000 | 76.695 | 76.753 |
| Total seluruh chunk, byte tanpa kompresi | 5.251.341 | 5.252.805 | 2.074.407 |
| Total seluruh chunk, byte estimasi gzip | 1.563.000 | 1.570.424 | 685.416 |

JavaScript awal turun sekitar 95%. Lazy routes menunda pemuatan; custom Plotly
juga mengurangi total kode menjadi sekitar 2,07 MB. CSS, font, gambar, dan HTML
tidak termasuk angka tersebut.

Landing tetap dimuat langsung. Upload, simulator, daftar insiden, workspace,
workspace insiden, TWS, dan double-ended fault locator dimuat saat rutenya dibuka.
Provider analisis berada di luar Suspense agar pilihan relay tetap tersedia.
Halaman yang sedang dimuat menampilkan status `Loading page…`.

HTML entry hanya memuat entry dan shared JSX runtime, tanpa preload Plotly.
Upload menambah sekitar 53 KB JavaScript (19,5 KB gzip) setelah entry.

## Runtime grafik yang lebih kecil

Inventarisasi trace menemukan `scatter` (waveform, digital fill, locus,
differential, OCR), `histogram` (DE-FL), dan `scattergl` (TWS). Runtime memakai
`plotly.js/lib/core`, yang sudah memuat scatter, lalu mendaftarkan histogram dan
scattergl. React memakai factory agar tidak ikut menarik bundle Plotly lengkap.
API `window.Plotly.toImage` tetap disediakan bagi export chart untuk laporan.

| Tambahan JS pada kunjungan pertama | Lazy routes | Dengan custom Plotly |
|---|---:|---:|
| Chunk Plotly, tanpa kompresi | 4,65 MB | 1,47 MB |
| Workspace beserta static dependencies, tanpa kompresi | 4.879.513 byte | 1.700.943 byte |
| Workspace beserta static dependencies, estimasi gzip | 1.449.517 byte | 564.466 byte |

Chunk grafik turun sekitar 68%; tambahan transfer workspace menurut estimasi
gzip turun sekitar 61%. Perhitungan waveform tidak berubah. Jalur WebGL memakai
polyfill `buffer` dan alias `globalThis`; build final tidak lagi mengeksternalkan
Buffer sebagai modul Node yang tidak tersedia di browser.

Smoke check pada JavaScript production di JSDOM/VM, tanpa global Node Buffer
atau global, berhasil memuat chunk serta memeriksa registry trace, render
scatter/histogram/shape, zoom, dan export SVG. Ini simulasi DOM, bukan pengujian
GPU atau browser nyata. Setelah pengguna memilih Playwright, pemeriksaan
Chromium headless juga dijalankan; hasilnya dicatat di bawah.

## Batas verifikasi dan pekerjaan berikutnya

- Build production berhasil. Suite frontend sesudah perubahan lulus 47 tes
  dalam 12 file, termasuk tiga tes tambahan untuk navigasi lazy upload dengan
  pilihan relay, direct link workspace insiden, dan redirect URL yang tidak
  dikenal. Lint seluruh file kode/config yang diubah lulus; lint seluruh proyek masih memiliki
  26 error dan enam warning di file yang tidak diubah dalam tahap ini.
- Dua tes runtime tambahan memeriksa registry/API export, render waveform,
  digital fill, histogram, shape/annotation, zoom, dan export SVG di JSDOM.
- Seluruh perubahan masih lokal, tanpa commit, push, atau deployment.
- Cold/warm load, transfer aktual, dan JS heap sudah diukur dalam harness
  Chromium headless lokal. RAM seluruh proses, GPU perangkat pengguna, dan
  latency p50/p95 produksi belum diukur.
- Percobaan awal Python dalam sandbox gagal. Eksekusi Python di luar sandbox
  kemudian berhasil; backend lokal untuk validasi sudah dapat dijalankan.
  Pengukuran latency produksi/backend lengkap masih belum dilakukan.
- Berikutnya: ukur ulang jalur live API dengan profiling backend, terutama
  parsing/komputasi berulang dan pemuatan waveform lengkap. Penurunan byte
  frontend belum menyelesaikan seluruh biaya analisis.

Bobot model dan perhitungan waveform tidak berubah dalam tahap ini.

## Persiapan validasi browser lanjutan

Build committed App/PlotlyChart dari commit
`ce823d9eaa308bed13d3848f0a360e14585b4998` berhasil direproduksi terpisah di
`tmp/performance-before`; entry JavaScript-nya tetap 5.251.341 byte. Working tree
dan build optimasi tidak ditimpa.

Seleksi read-only pada korpus lokal menemukan 1.125 pasangan CFG/DAT, lalu
memilih tiga rekaman nyata berikut. Raw tetap pada lokasi asal; manifest dan
session pengujian tersimpan hanya di `tmp/browser-validation`.

| Raw CFG+DAT | Sampel | Kanal analog | Kanal digital | Upload pertama lokal |
|---:|---:|---:|---:|---:|
| 1.002.366 byte | 3.601 | 27 | 77 | 412 ms |
| 5.002.687 byte | 20.482 | 18 | 64 | 1.468 ms |
| 9.980.789 byte | 46.848 | 18 | 32 | 2.336 ms |

Ketiganya berhasil diparse melalui API upload sebenarnya pada backend loopback
terisolasi. Angka upload adalah satu observasi per file, termasuk pengiriman dan
respons, bukan p50/p95 atau hasil perbandingan optimasi. Retensi training
dimatikan; tidak ada penambahan contoh training. Kedua build memakai ID
session yang sama agar waveform input identik.

## Hasil Chromium headless

Playwright 1.58.2 memakai Chromium terpasang versi 153.0.8010.12, viewport
1440 x 1000, dan ANGLE SwiftShader untuk pemeriksaan WebGL. Server frontend
loopback menyediakan gzip dan cache asset yang sama bagi kedua build. Setiap
skenario diulang tiga kali: context browser baru untuk cold load, lalu reload
dengan cache untuk warm load. Urutan before/after dibalik pada pengulangan kedua.
Total 48 observasi readiness, tanpa JavaScript page error.

Respons read-only API direkam dari backend sebenarnya lalu dibekukan untuk
perbandingan frontend. Analysis POST diprime sebelum pengukuran dan dibagikan
ke kedua build. Ini mengisolasi biaya transfer/parse/render frontend; bukan
benchmark backend live atau produksi. Request PDF tetap diteruskan ke backend.
`readyMs` adalah waktu dari navigasi sampai grafik dan indikator loading selesai,
font siap, serta network idle, termasuk gate idle Playwright. Bukan LCP atau
waktu komputasi analisis. Sampel tiga cukup untuk median/rentang lokal, bukan p95.

| Skenario | Cold readiness median sebelum -> sesudah | Warm median sebelum -> sesudah | Cold JS heap median sebelum -> sesudah |
|---|---:|---:|---:|
| Landing | 1.233 -> 814 ms | 914 -> 616 ms | 9,54 -> 2,04 MB |
| Rekaman 1 MB, OCR | 1.866 -> 1.768 ms | 1.393 -> 1.339 ms | 23,38 -> 18,67 MB |
| Rekaman 5 MB, distance | 2.348 -> 2.358 ms | 2.384 -> 1.933 ms | 33,94 -> 29,24 MB |
| Rekaman 10 MB, distance | 3.115 -> 2.436 ms | 2.724 -> 2.506 ms | 46,55 -> 41,96 MB |

Heap diambil lewat CDP setelah garbage collection; ini bukan RSS seluruh
browser. Waktu workspace bervariasi: cold 5 MB sebelum 2.135-4.000 ms dan
sesudah 2.275-2.705 ms; tidak ada bukti percepatan cold load kasus tersebut.
Landing cold lebih stabil: sebelum 1.213-1.257 ms, sesudah 775-829 ms.

Transfer JavaScript cold yang terukur pada resource timing:

- Landing: 1.563.300 -> 77.353 byte, turun sekitar 95%.
- Workspace tiap rekaman: 1.563.300 -> 643.019 byte, turun sekitar 59%.
- Warm reload tidak mentransfer ulang JS karena cache browser.
- Transfer API tetap sama: sekitar 154 KB, 1,62 MB, dan 3,01 MB bagi masing-masing
  kasus. Karena itu, payload API masih dominan pada rekaman besar.

Pemeriksaan fungsi:

- Fingerprint seluruh data trace (x/y/type/name/fill) dan shape sama untuk
  23 grafik kasus 1 MB, 17 grafik kasus 5 MB, dan 19 grafik kasus 10 MB.
- Hover pointer pada waveform nyata serta zoom berjalan pada ketiga kasus
  di kedua build.
- Histogram DE-FL dan trace WebGL TWS diuji memakai fixture deterministik.
  Hover dan zoom lulus; capture PNG memuat pixel trace, bukan hanya canvas
  kosong. Ukuran PNG dan jumlah pixel berwarna sama sebelum/sesudah.
- Download PDF OCR live lulus setelah Plot TCC: request berisi PNG
  `overcurrent_overlay`, PDF valid 374.058 byte pada kedua build. Review visual
  seluruh tiga halaman menunjukkan render yang sama. Pemisahan judul TCC di
  akhir halaman 2 dari grafik di halaman 3 sudah ada pada kedua build; ini
  masalah pagination terpisah, bukan regresi perubahan bundle.
- Workflow upload .cdb TWS dan PDF distance/DE-FL belum diuji end-to-end.
  Pemeriksaan ini memastikan runtime trace/export, bukan seluruh alur tersebut.

Run awal memakai API live tanpa replay melewati timeout 120 detik saat mencoba
PDF rekaman 10 MB pada versi sebelum optimasi; respons report akhirnya tercatat
200 di backend setelah browser sudah timeout. Tes tersebut tidak membuktikan
regresi optimasi. Perlu profiling live API dan request bersamaan sebelum
menyimpulkan penyebab pasti; jangan menganggap hasil replay menyelesaikannya.

Kesimpulan: penurunan transfer dan JS heap frontend terbukti, serta pemeriksaan
kompatibilitas di atas lulus. Percepatan semua analisis belum terbukti. Semua
perubahan tetap lokal tanpa commit/push/deployment.

Hasil rinci, screenshot, PDF uji, serta snapshot API ada di ignored
`tmp/browser-validation`; tidak dimasukkan ke training atau commit.
