# Labeling Guidelines

Panduan singkat untuk kurasi label penyebab gangguan agar konsisten antar engineer, terutama setelah penambahan kelas `PERALATAN / PROTEKSI`.

Update terakhir: April 2026

---

## Prinsip Utama

Labeli berdasarkan **root cause fisik/operasional yang paling mungkin**, bukan hanya akibat akhirnya.

Urutan prioritas evidence:
1. status digital / SOE / relay target / event recorder
2. waveform arus-tegangan
3. laporan operasi / patroli / inspeksi lapangan
4. nama folder / nama file

Jika evidence masih lemah atau saling bertentangan, **jangan paksa label final**. Gunakan catatan di `notes`.

---

## Definisi Kelas

| Label | Gunakan Saat |
|---|---|
| `PETIR` | Ada indikasi kuat sambaran petir / cuaca buruk / karakter transient sangat konsisten dengan lightning |
| `LAYANG-LAYANG` | Ada bukti gangguan akibat layang-layang / tali layangan |
| `POHON / VEGETASI` | Ada bukti sentuhan pohon, ranting, atau ROW vegetasi |
| `HEWAN / BINATANG` | Ada bukti hewan sebagai penyebab utama |
| `BENDA ASING` | Ada benda asing non-hayati selain layang-layang |
| `KONDUKTOR / TOWER` | Ada indikasi kerusakan mekanik line: konduktor, joint, clamp, armor rod, tower, hardware line |
| `PERALATAN / PROTEKSI` | Ada indikasi akar gangguan berasal dari peralatan proteksi, telekomunikasi, instrument transformer, PMT, atau auxiliary system |

---

## Kapan Pakai `PERALATAN / PROTEKSI`

Masukkan ke `PERALATAN / PROTEKSI` bila akar masalah paling mungkin berasal dari:
- pilot wire
- teleprotection / teleproteksi
- PLCC / communication channel
- relay malfunction / logic issue / maloperate
- CT / VT / CVT abnormal
- PMT / CB mechanism failure
- DC supply / battery / auxiliary failure
- channel status / wiring / interface failure

Contoh frasa folder atau catatan yang cocok:
- `gangguan pilot wire`
- `teleprotection fail`
- `PLCC problem`
- `PMT gagal close`
- `relay abnormal`
- `CVT problem`

---

## Kapan Jangan Pakai `PERALATAN / PROTEKSI`

Jangan gunakan label ini jika penyebab utamanya masih lebih cocok sebagai:
- `PETIR`
- `LAYANG-LAYANG`
- `POHON / VEGETASI`
- `HEWAN / BINATANG`
- `BENDA ASING`
- `KONDUKTOR / TOWER`

Contoh:
- AR gagal karena konduktor putus: pakai `KONDUKTOR / TOWER`
- trip saat badai petir dan tidak ada bukti kuat equipment failure: pakai `PETIR` atau tahan dulu
- relay trip setelah benda asing tersangkut konduktor: pakai `BENDA ASING`, bukan `PERALATAN`

---

## Aturan Praktis Untuk Menghindari Kebingungan

1. Label berdasarkan **penyebab**, bukan **lokasi peralatan yang trip**.
2. Jika relay/proteksi hanya menjadi pihak yang mendeteksi gangguan eksternal, jangan otomatis beri label `PERALATAN`.
3. Jika ada bukti kuat equipment failure tetapi penyebab eksternal belum terbukti, labeli `PERALATAN / PROTEKSI`.
4. Jika masih ragu antara external cause vs equipment cause, biarkan `correct` kosong atau beri catatan `suspected_peralatan` di `notes`.

---

## Template Pengisian Notes

Gunakan format singkat berikut agar review berikutnya lebih mudah:

```text
Root cause: PERALATAN / PROTEKSI
Evidence: teleprotection fail + PMT tidak reclose + tidak ada bukti gangguan eksternal
Why not external: tidak ada indikasi petir/layang/pohon/hewan pada laporan lapangan
Confidence: medium
```

Jika belum yakin:

```text
Root cause: suspected_peralatan
Evidence: pilot wire abnormal, tetapi belum ada konfirmasi final
Confidence: low
```

---

## Rekomendasi Kurasi Tahap Awal

Untuk mencegah taksonomi terlalu cepat pecah, sementara ini **jangan pecah dulu** `PERALATAN / PROTEKSI` menjadi sub-label seperti:
- `PILOT_WIRE`
- `PLCC`
- `PMT`
- `CT_VT`
- `RELAY_MALOPERATE`

Kumpulkan dulu semuanya di umbrella label `PERALATAN / PROTEKSI`. Setelah jumlah datanya cukup, baru pertimbangkan pemecahan sub-kelas.

