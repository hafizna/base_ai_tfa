# Rencana Perbaikan Performance dan Tech Stack

Tanggal review: 10 Oktober 2026. Status: rencana; belum diimplementasikan.

Tujuan: mempercepat pembukaan web dan analisis COMTRADE berukuran hingga sekitar
10 MB, dengan menjaga akurasi analisis dan detail waveform saat investigasi.

## Temuan dan batas bukti

| Area | Bukti dari repo | Implikasi yang perlu diukur |
|---|---|---|
| Frontend | Artifact build lokal memiliki JavaScript 5.230.861 byte sebelum kompresi; `App.tsx` mengimpor halaman langsung | Pembukaan awal dapat memuat kode halaman dan grafik yang belum diperlukan; ukuran production belum diverifikasi |
| Grafik | `PlotlyChart.tsx` memakai `react-plotly.js`; explorer sudah membatasi titik plot | Periksa jenis trace yang digunakan sebelum memilih bundle lebih kecil; pembatasan plot belum mengurangi transfer data lengkap |
| Waveform | Workspace memanggil `/api/analysis/{id}`, yang mengembalikan semua sampel analog, digital, dan waktu | Transfer, parsing JSON, serta RAM browser dapat meningkat untuk rekaman besar |
| Storage | Fallback filesystem menyimpan dan membaca payload JSON lengkap | Endpoint berbeda dapat mengulang pembacaan dan parsing payload yang sama |
| Backend | Sejumlah endpoint `async` memanggil `load_analysis` secara sinkron sebelum menjalankan komputasi di executor | Pembacaan/parsing besar berpotensi menahan event loop |
| Deployment | `docker-compose.prod.yml` mengatur `WEB_CONCURRENCY=1`, sedangkan default `start.sh` adalah 2 | Konfigurasi repo belum membuktikan konfigurasi live; kapasitas dan antrean perlu diukur |
| Training archive | Badge menghitung seluruh file dalam direktori training; panel mengambil status, bukan isi arsip | Contoh 67 raw / 118 MB adalah akumulasi arsip server, bukan ukuran satu analisis atau download otomatis |

Profiling historis 21 Mei 2026 menggunakan rekaman sintetis sekitar 26 KB.
Hasil tersebut tidak cukup untuk menyimpulkan bottleneck pada file 1–10 MB.
GZip dan preload model sudah tersedia, tetapi dampaknya pada rekaman production
belum diukur dalam review ini. CPU, RAM, tipe instance, CPU credits, ukuran
respons production, dan waktu rendering browser belum terverifikasi.

DuckDNS berperan dalam resolusi alamat, bukan pemrosesan COMTRADE. Ukur waktu DNS
secara terpisah dari koneksi, upload, komputasi, transfer, dan rendering sebelum
menentukan penyebab lambat.

## Urutan pekerjaan

### 1. Tetapkan baseline

- Uji rekaman representatif sekitar 1, 5, dan 10 MB dengan variasi jumlah channel,
  sample rate, durasi, dan format DAT; ukuran file saja tidak menentukan beban.
- Pisahkan cold load dan warm load, satu pengguna dan beberapa pengguna bersamaan.
- Catat DNS/connect/TLS, waktu upload, parse, penyimpanan, komputasi per endpoint,
  serialization/compression, ukuran respons terkompresi, serta waktu hingga grafik
  dan hasil analisis siap digunakan.
- Ukur CPU, RSS/peak RAM, disk I/O, swap, antrean, dan CPU credits bila instance
  memakai model burstable. Catat p50/p95 dan jumlah pengulangan pengujian.
- Gunakan file uji yang boleh dipakai; jangan masukkan waveform sensitif ke log.

### 2. Kurangi kode yang dimuat di awal

- Terapkan route lazy loading dan muat panel grafik ketika diperlukan.
- Inventarisasi trace Plotly dan kebutuhan export PDF sebelum memilih partial
  atau custom bundle; pertahankan kompatibilitas grafik yang digunakan.
- Verifikasi kompresi dan cache static assets pada deployment. Hashed assets
  dapat memakai cache panjang; HTML tetap harus mengambil build terbaru.
- Bandingkan ukuran transfer dan waktu siap pakai dengan baseline.

### 3. Kurangi data yang dikirim ke browser

- Tampilkan summary terlebih dahulu, lalu ambil channel dan rentang waktu yang
  sedang ditampilkan. Endpoint summary sudah ada, tetapi masih membaca payload
  lengkap di backend; metadata perlu disimpan agar summary benar-benar ringan.
- Buat preview waveform di server dengan metode yang mempertahankan ekstremum;
  mengambil setiap titik ke-N saja dapat melewatkan puncak gangguan.
- Saat zoom, ambil detail rentang tersebut. Pertahankan data resolusi penuh
  untuk perhitungan, export, dan investigasi.
- Representasikan digital status sebagai initial state dan transisi dengan
  timestamp/index yang tepat; uji kesetaraan terhadap sampel asli.
- Evaluasi binary waveform hanya jika pengukuran menunjukkan JSON tetap mahal.
  Jangan mengubah presisi numerik tanpa validasi toleransi.

### 4. Gunakan ulang parsing dan hasil komputasi

- Hindari membaca serta mengurai payload penuh untuk setiap panel.
- Cache payload/hasil dengan batas byte, TTL, dan eviction; hindari cache global
  tanpa batas yang justru menghabiskan RAM.
- Key cache mencakup analysis ID, revisi data, rasio CT/VT, parameter analisis,
  dan versi algoritma yang relevan; invalidasi saat data atau parameter berubah.
- Deduplicasi request identik yang sedang berlangsung dan hasil yang sudah ada.
- Pindahkan I/O sinkron dari event loop. Untuk pekerjaan CPU berat, ukur apakah
  executor yang ada cukup atau perlu process worker/job queue.
- Perhitungkan bahwa cache per proses tidak otomatis dibagi antarworker.

### 5. Tune hosting berdasarkan hasil pengukuran

- Tambah worker hanya setelah memeriksa CPU dan RAM; worker tambahan membawa
  salinan model/cache dan tidak otomatis mempercepat satu request.
- Pertimbangkan instance berbeda jika CPU, credits, atau RAM terbukti membatasi.
- Tambahkan background job dengan status/progress jika analisis lama atau
  concurrency membuat request interaktif terhambat; jangan menambah queue tanpa
  kebutuhan terukur.

## Keputusan tech stack

Pertahankan React + TypeScript + Vite dan FastAPI + Python untuk tahap ini.
Review menemukan peluang pada loading, transfer data, dan pekerjaan berulang;
belum ada bukti yang membenarkan rewrite backend.

| Lapisan | Arah yang disarankan | Kapan perlu diperluas |
|---|---|---|
| UI | React/Vite dengan halaman dan grafik dimuat sesuai kebutuhan | Ganti library waveform hanya jika profiling menunjukkan biaya rendering tetap dominan |
| Analisis | FastAPI + NumPy/SciPy + model Python yang sudah digunakan | Optimasi fungsi tertentu setelah profil CPU mengidentifikasi hot path |
| Penyimpanan | Pisahkan raw COMTRADE, metadata/feedback, dan hasil analisis | Object storage/database terpisah saat kebutuhan kapasitas atau multi-instance muncul |
| Cache | Cache terbatas dengan invalidasi eksplisit | Shared cache jika multiworker/multi-instance perlu berbagi hasil |
| Serving | Nginx di depan API dengan kompresi/cache static assets yang diverifikasi | CDN bila transfer static assets terbukti menjadi masalah |
| Job processing | Executor yang sesuai dengan karakter pekerjaan | Process worker/queue bila durasi dan concurrency membutuhkannya |

ONNX, Rust/Go, Redis, queue, dan upgrade AWS adalah opsi bersyarat, bukan
prasyarat. Pilih perubahan berdasarkan bottleneck yang terukur dan biaya
operasionalnya.

## Kriteria verifikasi

- Laporkan sebelum/sesudah untuk dataset dan kondisi jaringan yang sama;
  tetapkan target latency setelah baseline tersedia, tanpa menjanjikan angka
  percepatan yang belum diukur.
- Validasi timing inception/clearing, puncak, parameter elektrikal, klasifikasi,
  digital transitions, zoom, dan export PDF tetap konsisten.
- Uji invalidasi setelah perubahan rasio/parameter, pembatasan RAM cache,
  session expiration, dan request bersamaan.
- Preview/downsampling hanya mengubah penyajian; hasil analisis tetap memakai
  resolusi dan presisi yang telah divalidasi.

## Referensi

- [React lazy](https://react.dev/reference/react/lazy)
- [Plotly custom bundles](https://github.com/plotly/plotly.js/blob/main/CUSTOM_BUNDLE.md)
- [Starlette thread pool](https://www.starlette.io/threadpool/)
- [Panduan profiling lokal](profiling/README.md)
