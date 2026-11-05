# Prediksi Harga Komoditas Pangan DKI Jakarta (XGBoost vs LightGBM)

Implementasi rancangan pada proposal skripsi: memprediksi harga harian 20 atribut harga dari
10 kelompok komoditas pasar modern DKI Jakarta (PIHPS Bank Indonesia, 2 Jan 2023 – 31 Jul 2026)
dan membandingkan **XGBoost** vs **LightGBM** dengan **MAE, RMSE, R²** pada horizon
**1 hari, 1 minggu, dan 1 bulan** (= 1, 5, 22 hari kerja).

## Isi folder

| File | Fungsi |
|---|---|
| `01_training_evaluasi.ipynb` | Notebook utama (sudah dijalankan): EDA, korelasi, uji ADF, fitur, seleksi window, training, evaluasi, simpan model |
| `common.py` | Kode inti bersama: pra-pemrosesan, fitur, model, evaluasi, prediksi |
| `train.py` | Menjalankan seluruh pipeline dari terminal (hasil sama dengan notebook) |
| `app.py` | Website Streamlit: Dashboard, Data dan Visualisasi Harga, Evaluasi Model |
| `data/dataset_harga_preprocessed.xlsx` | Dataset |
| `artifacts/` | Model tersimpan (`models.joblib`) dan tabel hasil (CSV, `meta.json`) |

## Cara menjalankan

```bash
pip install -r requirements.txt

# (opsional) latih ulang; artifacts/ sudah berisi hasil siap pakai
python train.py                       # ±3 menit di CPU biasa
# atau buka 01_training_evaluasi.ipynb di Jupyter / VS Code / Colab

streamlit run app.py                  # buka website
```

Di Google Colab: unggah seluruh folder, jalankan `!pip install -r requirements.txt`, lalu buka notebook.

## Alur singkat

1. **Pra-pemrosesan**: tanggal → datetime, urut kronologis, numerik, *forward-fill* + interpolasi linear (tidak ada nilai hilang pada data ini).
2. **EDA**: grafik per kelompok, korelasi antaratribut, uji ADF (4 dari 20 atribut stasioner pada level; semuanya stasioner setelah *differencing*).
3. **Fitur** dari riwayat tiap atribut: lag, persentase perubahan, rata-rata & simpangan baku bergulir, jarak ke min/maks jendela, hari dalam minggu.
4. **Target** = perubahan harga `h` langkah ke depan (`y[t+h] − y[t]`), lalu ditambahkan ke harga hari ini.
5. **Pembagian data kronologis tanpa kebocoran**: data uji = sampel yang tanggal targetnya ≥ `TEST_START` (`2026-05-01`, 3 bulan terakhir). Window (7/14/30 hari kerja) dan hyperparameter dipilih pada **data validasi** (3 bulan sebelumnya), bukan pada data uji.
6. **Evaluasi** per atribut → dirata-ratakan per kelompok komoditas, dibandingkan dengan **baseline naive** (harga hari ini = harga ke depan).
7. **Model akhir** dilatih ulang pada seluruh data dan disimpan untuk website.

## Hasil pada data uji (Mei–Jul 2026)

Rata-rata seluruh atribut dan horizon:

| Model | MAE (Rp) | RMSE (Rp) |
|---|---|---|
| Naive (baseline) | 1.136 | 1.668 |
| LightGBM | 1.124 | 1.598 |
| XGBoost | 1.103 | 1.583 |

Per horizon (MAE rata-rata, Rp): 1 hari → naive 194,1 / LightGBM 194,2 / XGBoost 195,6;
1 minggu → naive 962,6 / LightGBM 1.011,2 / XGBoost 997,1;
1 bulan → naive 2.252,0 / LightGBM 2.166,9 / XGBoost 2.116,3.

**Cara membaca hasil ini (jujur):** harga pangan sangat persisten, sehingga baseline naive sangat kuat.
XGBoost dan LightGBM praktis setara baseline pada horizon 1 hari dan 1 minggu, dan hanya unggul tipis
pada horizon 1 bulan (XGBoost ±6% lebih rendah MAE-nya dibanding naive), itu pun hanya pada 20–25% atribut.
Selisih antara XGBoost dan LightGBM kecil dan tidak konsisten antarkelompok (lihat tabel *Model terbaik*
di notebook/website). Kesimpulan "model X lebih baik" sebaiknya disampaikan dengan kehati-hatian ini.
Konfigurasi yang lebih ekspresif (`dasar`) terbukti **lebih buruk** daripada baseline pada validasi,
karena mempelajari derau.

## Catatan & batasan

- **Periode uji.** Proposal menyebut uji pada Agustus–Oktober 2026, tetapi dataset berakhir 31 Juli 2026,
  sehingga yang dipakai adalah 3 bulan terakhir dataset. Setelah data aktual Agustus–Oktober ditambahkan
  ke xlsx (dengan format kolom yang sama), ubah `TEST_START = "2026-08-01"` di `common.py` dan jalankan ulang `train.py`.
- **R²** dihitung pada periode pendek: bisa sangat negatif (terutama horizon 1 bulan) dan tidak terdefinisi (NaN)
  bila harga aktual konstan sepanjang periode uji (mis. gula pasir). Baca bersama MAE/RMSE.
- **Hanya riwayat harga**: tidak ada cuaca, musim panen, hari raya, atau kebijakan.
- Prediksi website berupa **satu titik per horizon** (1, 5, 22 hari kerja ke depan), bukan kurva harian,
  sesuai rancangan *direct multi-horizon*. Tanggal prediksi dihitung dengan hari kerja Senin–Jumat
  (libur nasional tidak diperhitungkan).
- File `models.joblib` bergantung pada versi XGBoost/LightGBM; bila versi berbeda dan gagal dimuat, jalankan `python train.py`.
- Visualisasi di notebook memakai Matplotlib, di website memakai Plotly (interaktif).
- Tampilan website mengikuti rancangan antarmuka proposal (Lampiran 8-10). Status "Naik" berarti harga terbaru
  lebih tinggi daripada harga 1 minggu (default) sebelumnya; "Tidak Naik" berarti sama atau turun.
  Model terbaik keseluruhan ditetapkan dari MAE rata-rata terendah; alasannya ditulis otomatis, termasuk
  metrik yang justru lebih baik pada model lain.
