# Prediksi Harga Pangan DKI Jakarta

Aplikasi Streamlit untuk memprediksi harga harian 20 atribut pangan dari 10 kelompok komoditas pasar modern DKI Jakarta menggunakan XGBoost dan LightGBM. Data berasal dari PIHPS Bank Indonesia dan mencakup 2 Januari 2023 hingga 6 November 2026.

## Fitur Aplikasi

- **Dashboard** — menampilkan ringkasan harga dan perubahan harga pangan.
- **Prediksi** — menampilkan estimasi harga untuk atribut dan tanggal yang dipilih.
- **Evaluasi Model** — menampilkan MAE, RMSE, dan R² dari model terpilih.

## Struktur Proyek

```text
.
├── app.py                            # Aplikasi Streamlit
├── common.py                         # Pengolahan data, fitur, dan prediksi
├── train.py                          # Pelatihan dan evaluasi model
├── predict.py                        # Prediksi dan ekspor CSV dari terminal
├── cek_konsistensi.py                # Pemeriksaan model dan hasil evaluasi
├── paper.ipynb                       # Notebook analisis dan visualisasi
├── requirements.txt                  # Dependensi Python
├── data/
│   └── dataset_harga_preprocessed.xlsx
├── models/                           # Model akhir dan metadata per atribut
├── best_params.json                  # Parameter model terpilih
├── validation_report.json            # Ringkasan pemeriksaan data dan fitur
├── eval_results.csv                  # Metrik setiap konfigurasi
├── eval_horizon.csv                   # Metrik per horizon prediksi
└── test_predictions.csv              # Harga aktual dan prediksi pada data uji
```

## Cara Menjalankan

Buka terminal dari folder utama proyek, lalu instal dependensi.

```bash
python -m pip install -r requirements.txt
```

Jika folder `models/` belum berisi model, jalankan pelatihan terlebih dahulu.

```bash
python train.py
```

Jalankan aplikasi.

```bash
streamlit run app.py
```

Untuk membuat prediksi lima hari kerja melalui terminal, jalankan perintah berikut. Hasilnya tersimpan di `hasil_prediksi.csv`.

```bash
python predict.py --days 5
```

## Metode

Data dibagi secara kronologis dengan rasio 70:30. Periode uji berlangsung dari 11 September 2025 hingga 6 November 2026. Model mempelajari perubahan relatif harga harian menggunakan fitur dari riwayat harga dan kalender.

Untuk setiap atribut, konfigurasi dipilih berdasarkan MAE validasi silang terendah. Pencarian hiperparameter menggunakan Optuna dengan validasi silang deret waktu. Ukuran jendela masukan yang diuji adalah 3, 5, 10, dan 22 hari kerja. Model terpilih dilatih ulang pada seluruh data sebelum digunakan di aplikasi.

## Hasil Evaluasi

LightGBM terpilih untuk 11 atribut dan XGBoost untuk 9 atribut. Prediksi beberapa hari ke depan dihitung secara rekursif.

| Horizon prediksi | MAE rata-rata (Rp) | RMSE rata-rata (Rp) |
|---|---:|---:|
| 1 hari kerja | 276,21 | 936,89 |
| 5 hari kerja | 1.369,05 | 2.398,83 |
| 22 hari kerja | 3.367,49 | 4.825,72 |

Pada horizon satu hari kerja, metode naif yang memakai harga terakhir menghasilkan MAE 219,18, lebih rendah daripada MAE model terpilih sebesar 276,21.

## Batasan

- Model menggunakan riwayat harga dan fitur kalender.
- Kalender prediksi menghitung hari Senin–Jumat.
- R² tidak terdefinisi jika harga aktual konstan selama periode uji.
