# 1. LIBRARY YANG DIPAKAI
import glob
import json
import os

import pandas as pd

# ============================================================
# 2. TEMPAT MENYIMPAN TEMUAN MASALAH
# ============================================================
# Aku buat list kosong bernama masalah untuk menampung semua temuan.
# Kalau ada folder salah tempat, file hilang, atau hasil evaluasi tidak cocok,
# pesannya akan dimasukkan ke sini dan ditampilkan di akhir pemeriksaan.
masalah = []

# 1) Struktur folder & format model
# ============================================================
# 3. CEK STRUKTUR FOLDER DAN FORMAT INFORMASI MODEL
# ============================================================
# Pertama, aku cek apakah ada folder models di dalam folder models.
# Kondisi ini bisa terjadi kalau hasil training dalam ZIP diekstrak
# satu tingkat terlalu dalam, sehingga program mungkin sulit menemukan model.
if os.path.isdir(os.path.join("models", "models")):
    masalah.append("Ada folder models/models/ -> zip diekstrak ke tempat yang salah. Pindahkan isinya ke models/.")
# Aku cari semua file informasi model dengan akhiran _info.json
# di folder models. sorted() membuat urutannya konsisten saat diperiksa.
# Perlu dibedakan: file ini adalah metadata model, bukan selalu file
# model terlatihnya sendiri.
info_json = sorted(glob.glob(os.path.join("models", "*_info.json")))
# Selanjutnya aku cari file informasi dengan format lama _info.joblib.
# Pengecekan ini dibuat agar metadata dari versi lama tidak tercampur
# dengan metadata format baru yang sedang digunakan proyek.
info_lama = glob.glob(os.path.join("models", "*_info.joblib"))
# Jika tidak ditemukan satu pun metadata model format baru,
# aku catat sebagai masalah karena data pembandingnya tidak tersedia.
if not info_json:
    masalah.append("Tidak ada model format baru (*_info.json) di folder models/.")
# Jika masih ada metadata format lama, jumlahnya ikut dilaporkan.
# Script hanya memberi peringatan; tidak otomatis menghapus file tersebut.
if info_lama:
    masalah.append(f"Masih ada {len(info_lama)} model format lama (.joblib); hapus agar tidak tercampur.")

# 2) File hasil lain
# ============================================================
# 4. CEK KELENGKAPAN FILE OUTPUT TRAINING DAN EVALUASI
# ============================================================
# Bagian ini memastikan empat file pendukung tersedia di folder proyek:
# - eval_results.csv: ringkasan evaluasi dan model untuk tiap atribut.
# - eval_horizon.csv: hasil evaluasi menurut jangkauan prediksi.
# - test_predictions.csv: hasil prediksi yang disimpan untuk data pengujian.
# - validation_report.json: laporan validasi dalam bentuk JSON.
# Di sini aku hanya mengecek keberadaan file, belum memeriksa seluruh
# isi atau konsistensi keempat file tersebut.
for f in ["eval_results.csv", "eval_horizon.csv", "test_predictions.csv", "validation_report.json"]:
    # Kalau file tidak ada, nama file yang hilang ditambahkan ke daftar masalah.
    if not os.path.exists(f):
        masalah.append(f"File '{f}' tidak ada di folder proyek.")

# 3) eval_results.csv harus cocok dengan model di folder models/ (model, window, MAE uji)
# ============================================================
# 5. COCOKKAN HASIL EVALUASI DENGAN INFORMASI MODEL TERSIMPAN
# ============================================================
# Pemeriksaan lebih lanjut hanya berjalan jika eval_results.csv tersedia
# DAN setidaknya ada satu file metadata model berformat _info.json.
# Tujuannya agar informasi hasil evaluasi tidak tertukar dengan model
# dari proses training atau versi eksperimen yang berbeda.
if os.path.exists("eval_results.csv") and info_json:
    # Aku baca eval_results.csv sebagai DataFrame (tabel pandas) bernama e.
    # Setiap baris berisi informasi evaluasi model untuk suatu atribut.
    e = pd.read_csv("eval_results.csv")
    # Kolom Dipilih menandai baris model yang menjadi pilihan untuk suatu atribut.
    # Jika kolom ini tidak ada, file dianggap memakai format lama sehingga
    # proses pencocokan tidak diteruskan.
    if "Dipilih" not in e.columns:
        masalah.append("eval_results.csv versi lama (tidak ada kolom 'Dipilih'). Ambil yang baru dari Colab.")
    else:
        # Aku ambil hanya baris dengan nilai Dipilih yang dianggap True.
        # Setelah itu kolom Atribut dijadikan indeks supaya setiap baris
        # mudah dicocokkan dengan metadata model milik atribut tersebut.
        sel = e[e["Dipilih"]].set_index("Atribut")
        # Variabel cocok menghitung berapa atribut yang berhasil lolos
        # pemeriksaan nama model, window, dan MAE.
        cocok = 0
        # Aku periksa satu per satu atribut yang dipilih.
        # attr berisi nama atribut pangan, sedangkan r adalah data evaluasinya.
        for attr, r in sel.iterrows():
            # Nama atribut diubah dari spasi menjadi garis bawah agar sama
            # dengan pola nama file, lalu dibuat lokasi file _info.json-nya.
            p = os.path.join("models", attr.replace(" ", "_") + "_info.json")
            # Kalau metadata untuk atribut ini tidak ditemukan,
            # aku catat masalah dan lanjut mengecek atribut berikutnya.
            if not os.path.exists(p):
                masalah.append(f"Model untuk '{attr}' tidak ada.")
                continue
            # Metadata JSON dibaca untuk mengambil nama model, ukuran window,
            # serta nilai MAE yang tersimpan ketika model disiapkan.
            info = json.load(open(p, encoding="utf-8"))
            # Tiga bagian yang dibandingkan adalah:
            # 1) model_name dengan kolom Model: apakah algoritmanya sama;
            # 2) window dengan Input Window (Hari): apakah banyaknya
            #    pengamatan historis yang dipakai model sama;
            # 3) mae dengan MAE: apakah nilai kesalahan uji sesuai.
            # Selisih MAE > 0,000001 dianggap tidak cocok agar perbedaan
            # angka yang sangat kecil akibat presisi tetap ditoleransi.
            # Kalau salah satu tidak cocok, kemungkinan berkas berasal
            # dari proses training yang berbeda atau belum tersinkron.
            if (info["model_name"] != r["Model"] or int(info["window"]) != int(r["Input Window (Hari)"])
                    or abs(info["mae"] - r["MAE"]) > 1e-6):
                masalah.append(f"'{attr}': eval_results.csv BEDA dengan model (kemungkinan dari run yang berbeda).")
            else:
                # Jika ketiga informasi tadi sesuai, jumlah atribut cocok
                # ditambah satu.
                cocok += 1
        # Setelah semua atribut diperiksa, aku tampilkan jumlah yang cocok
        # dibandingkan jumlah seluruh atribut yang dipilih.
        print(f"Atribut cocok antara eval_results.csv dan models/: {cocok} dari {len(sel)}")

# ============================================================
# 6. TAMPILKAN HASIL AKHIR PEMERIKSAAN
# ============================================================
# Kalau list masalah berisi pesan, hasil akhir menampilkan PERHATIAN
# beserta rincian masing-masing masalah agar bisa diperbaiki dahulu.
if masalah:
    print("\nPERHATIAN:")
    for m in masalah:
        print(" -", m)
else:
    # Kalau tidak ada masalah yang terdeteksi oleh pemeriksaan di atas,
    # script menampilkan AMAN. Artinya pengecekan dasar ini lolos,
    # BUKAN jaminan seluruh isi CSV, semua file model, atau akurasi
    # prediksi sudah benar. Website dijalankan terpisah menggunakan Streamlit.
    print("\nAMAN: file hasil training konsisten dan siap dipakai (streamlit run app.py).")