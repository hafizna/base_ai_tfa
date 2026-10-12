# Penyelarasan konteks dan hipotesis penyebab

Implementasi 12 Oktober 2026. Bobot model penyebab tetap menggunakan versi sebelumnya.

## Perilaku yang berubah

Analisis satu COMTRADE dan rekonstruksi incident memakai pembaca konteks dan signature yang sama (`core/cause_context.py`). Fakta waveform, operasi proteksi, hasil pemulihan, dan hipotesis penyebab memiliki lingkup bukti berbeda.

- Reclose gagal berarti **pemulihan gagal pada percobaan yang terekam**. Rule tersebut tidak lagi menghentikan classifier dan mengeluarkan “gangguan permanen” sebagai pengganti penyebab.
- Pengulangan episode atau beberapa fasa dalam rekaman tidak otomatis membuktikan konduktor/tower rusak. Perubahan fasa harus dibandingkan antar episode.
- Incident menautkan hasil reclose sebagai bukti konteks. Bonus/penalti penyebab dari asumsi “petir harus berhasil, peralatan harus gagal” dihapus. Skor rekaman tidak diubah oleh penautan ini.
- PMT membuka tiga kutub/AR lockout tidak lagi menaikkan skor Hewan/Konduktor/Peralatan secara arbitrer atau menurunkan PETIR di endpoint analisis web.
- Receive sesudah trip tidak dipakai sebagai permission pada trip sebelumnya. Pembaca fitur training konteks mengikuti batas waktu yang sama.
- Impedansi dan voltage sag ringkasan memakai fasa terganggu terpilih. Minimum impedansi memakai V/I fundamental per siklus, bukan pembagian sesaat dekat zero crossing. Nilai ini **impedansi semu fasa tanpa kompensasi residual**, bukan resistansi kontak fault atau loop tanah relay yang telah dikalibrasi.
- Arus fault 50 Hz tidak diperlakukan sebagai arus sambaran untuk memastikan SF/BFO pada narasi web/laporan.

Respons analisis menambahkan `event_context`, `cause_assessment`, dan `episode_signatures`. Snapshot incident menyimpan provenance yang sama. UI/laporan membedakan skor rule konteks dari skor model penyebab, menandai penyebab sebagai hipotesis, dan menampilkan signature tiap episode. Model tidak tersedia berarti ranking kosong, bukan tujuh probabilitas seragam buatan.

Untuk kompatibilitas, `fault_type` masih dapat berisi `permanent` ketika reclose gagal atau `transient` ketika pemulihan bertahan. Consumer baru harus membaca `event_context.restoration_outcome` dan lingkup `recorded_interval_only`; field lama bukan konfirmasi sifat fisik permanen/transien. `protection_interpretation.event_class` kini menggunakan kelas urutan seperti `RECLOSE_REFAULT_SOTF`, `RECLOSE_FAILED`, `RECLOSE_SUCCESSFUL`, atau `FAULT_EVENT`.

## Signature dan batas model

Setiap episode menyimpan timing/fasa, peak/RMS arus dan tegangan, serta fit fundamental + DC per siklus lengkap. `fit_residual_ratio` adalah residual fit, **bukan THD** dan bukan pengenal pohon. CT, offset DC yang meluruh, switching, dan arcing dari berbagai sumber dapat menghasilkan distorsi serupa.

Episode kedua menggunakan referensi sehat sebelum episode pertama jika tersedia; dead time tidak dijadikan referensi sehat. Jika onset/clearing tidak tersedia, magnitudo episode tidak ditebak. Sampling efektif/Nyquist berasal dari timestamp, termasuk raw yang header sampling-nya nol. Perbandingan menyediakan kesamaan fasa dan rasio RMS kanal antar episode, tanpa label penyebab otomatis.

Bobot model penyebab dan tujuh kelas tidak berubah. Model aktif masih memakai 17 fitur lama: waveform episode pertama bersama fitur rekaman seperti fault count, reclose, trip type, dan zone. Signature tambahan belum menjadi input classifier; ini dinyatakan eksplisit melalui `used_by_deployed_model=false` dan `additional_episodes_evaluated_by_model=false`. Karena itu skor cause pada capture yang terpotong tidak dijanjikan identik dengan raw lengkap: cakupan waveform/referensi dan input konteks dapat berbeda. Konsistensi yang diverifikasi sekarang adalah pembacaan urutan dan pemisahan lingkup bukti, bukan kemampuan classifier baru.

## Pemeriksaan Rawalo

Raw berfingerprint `4bdc1f9ff37a12bdadeadf86910e43f2423d1f5acde3543e9acff1dbd10b9831` diuji sebagai rekaman lengkap serta tiga capture dengan batas 1,2 dan 1,55 detik. Capture buatan hanya untuk regresi, bukan contoh training independen.

| Bukti | Hasil |
|---|---|
| Urutan lengkap dan tiga capture | 2 episode, `RECLOSE_REFAULT_SOTF`, pemulihan gagal |
| Episode pertama, fasa B | 495,0–563,3 ms; peak 19.283,36 A; RMS 11.222,6 A |
| Episode kedua, fasa B | 1.585,0–1.662,3 ms; peak 22.783,99 A; RMS 11.838,0 A |
| Dead time | 1.051,6 ms, dipisahkan dari durasi fault |
| Receive pada trip pertama | Tidak aktif; receive datang setelah trip |
| Ringkasan impedansi | Fasa B: sekitar 4,18 Ω saat inception; minimum per siklus 3,88 Ω |
| Model penyebab lama | PETIR 66,6%, BENDA_ASING 23,9%; hipotesis belum terkonfirmasi |
| Rule konteks | Skor heuristik 90%, terpisah dari skor cause |
| SHA-256 model sebelum/sesudah | `a6c3e8ed7b4141c445a5f7886d8f5822ae8a3ba1f6db96e2ded0e1b0a74c7893` |

Nama folder kasus tidak digunakan sebagai ground truth otomatis. Hasil di atas tidak membuktikan POHON, HEWAN, ataupun PETIR.

Enam rekaman beranotasi konteks telah diekstrak ulang beserta signature. Evaluasi kembali menolak promosi: failed reclose/SOTF hanya satu kelompok kejadian independen dan POTT juga satu kelompok. Tidak ada bobot baru yang dipromosikan.

## Langkah pengembangan berikutnya

1. Tambahkan anotasi per episode dengan sumber bukti: electrical fault, trip path, mekanisme pemulihan, dan penyebab lapangan secara terpisah. Jangan menjadikan reclose gagal sebagai label cause.
2. Kumpulkan beberapa kelompok kejadian independen untuk setiap outcome dan tujuh penyebab, termasuk kontak vegetasi/hewan yang sama-sama bisa berulang. Kelompokkan duplikat/capture satu incident sebelum pemisahan train/validasi.
3. Bandingkan kandidat model memakai waveform per episode dan perubahan antar episode terhadap baseline 17 fitur. Audit khusus ketergantungan pada reclose/trip type, missing baseline, CT/scaling, serta perbedaan raw lengkap dan capture terpotong. Kalibrasi dan kemampuan menahan kesimpulan harus diuji pada incident yang belum pernah dilihat.
4. Promosikan kandidat hanya setelah evaluasi independen menunjukkan perbaikan. Signature baru tidak cukup untuk menyatakan model telah belajar konteks atau dapat mengenali pohon.

Artefak validasi lokal ada di `tmp/cause-alignment/` dan evaluasi dataset di `data/training-runs/context-alignment.evaluation.json`; keduanya tidak dimasukkan ke Git sebagai data training baru.

## Validasi implementasi

- Backend: 447 lulus; dua tes integrasi yang bergantung pada raw COMTRADE eksternal yang tidak tersedia dikeluarkan dari run ini. Setelah perubahan narasi final, 44 tes terkait laporan/incident/signature lulus lagi.
- Frontend: 49 lulus, production build berhasil. Run penuh memakai `--maxWorkers=2` setelah run dengan worker default mengalami timeout startup pada mesin ini.
- Headless Chromium/Playwright terhadap API lokal nyata: 10 grafik tampil, tidak ada error JavaScript, cause score 67% terpisah dari rule konteks 90%, dan PDF berhasil diunduh dengan dua locus PNG. Tidak memakai replay respons API.
- PDF tiga halaman dari pipeline langsung telah dirender dan diperiksa secara visual. PDF dari unduhan browser disimpan terpisah sebagai `tmp/cause-alignment/browser-report.pdf`. Pada sesi browser ini kedua PNG locus berisi sumbu kosong; pengujian membuktikan jalur ekspor dan isi konteks/cause, bukan akurasi trajektori locus. PDF langsung memakai diagram analog native dan tidak menyertakan PNG locus tersebut.

Nilai voltage sag pada narasi fitur classifier dan tabel parameter dapat berbeda karena jendela/perhitungan berbeda: pada Rawalo fitur model 43,8%, sedangkan RMS ringkasan fasa B selama episode pertama 49,3%. Ini tidak diubah diam-diam di vector model; penyeragaman fitur numerik memerlukan evaluasi kandidat model yang dijelaskan di atas.
