"""
predict.py - Prediksi harga ke depan memakai MODEL HASIL train.py dan DATA TERBARU di Excel.

Alur harian:
  1. Tambahkan harga hari ini ke baris paling bawah dataset_harga_preprocessed.xlsx (lalu simpan & tutup Excel).
  2. Jalankan salah satu:
       python predict.py                              # semua komoditas, 5 hari kerja ke depan
       python predict.py --days 22                    # 1 bulan (22 hari kerja)
       python predict.py --date 2026-11-06            # sampai tanggal target
       python predict.py --attr "Cabai Merah Besar" --days 10
       python predict.py --days 5 --out hasil.csv     # simpan ke CSV (default: hasil_prediksi.csv)
  Model TIDAK dilatih ulang. Latih ulang (python train.py) cukup berkala, mis. tiap beberapa minggu.
"""
# Library argparse dipakai supaya kita bisa mengatur prediksi lewat perintah terminal.
# Contohnya, kita bisa memilih jumlah hari, tanggal tujuan, nama komoditas, dan nama file hasil.
# Jadi, setiap ingin mencoba prediksi yang berbeda, kita tidak perlu mengubah isi kode.
import argparse

# Pandas membantu membaca dan mengolah data dalam bentuk tabel.
# Di sini pandas juga dipakai untuk merapikan hasil prediksi dan menyimpannya menjadi CSV.
import pandas as pd

# Fungsi-fungsi ini diambil dari common.py agar proses prediksi memakai aturan yang sama
# dengan program lainnya, khususnya perhitungan fitur yang sebelumnya dipakai saat training.
# ATTR_TO_GROUP: daftar atribut harga yang dikenal sistem beserta kelompok komoditasnya.
# check_data_quality: memeriksa apakah data terbaru memiliki hal yang perlu diperhatikan.
# load_and_preprocess_data: membaca Excel dan menyiapkan data harga untuk digunakan.
# load_model: mengambil model yang sebelumnya sudah dilatih dan disimpan.
# predict_future_recursive: menghasilkan prediksi beberapa hari secara bertahap.
# steps_to_date: menghitung jarak hari kerja dari data terakhir ke tanggal tujuan.
from common import (
    ATTR_TO_GROUP, check_data_quality, load_and_preprocess_data, load_model, predict_future_recursive, steps_to_date,
)


# Fungsi main() menjadi pusat alur program ketika predict.py dijalankan dari terminal.
# Urutannya adalah membaca pilihan pengguna, memeriksa data, memuat model,
# menghitung prediksi, lalu menampilkan ringkasan dan menyimpan hasilnya.
def main():
    # ArgumentParser mengatur pilihan yang boleh ditulis setelah "python predict.py".
    # Keterangan ini juga membantu pengguna saat menjalankan "python predict.py --help".
    parser = argparse.ArgumentParser(description="Prediksi harga pangan dari model yang sudah dilatih")
    # --days menentukan berapa hari kerja yang ingin diprediksi.
    # Jika tidak diisi, program otomatis menggunakan 5 hari kerja.
    parser.add_argument("--days", type=int, default=5, help="jumlah hari kerja ke depan (default 5)")
    # --date digunakan ketika kita ingin menentukan tanggal akhir prediksi.
    # Pilihan tanggal ini akan menggantikan nilai --days melalui perhitungan hari kerja.
    parser.add_argument("--date", type=str, default=None, help="tanggal target (YYYY-MM-DD); menggantikan --days")
    # --attr digunakan jika kita hanya ingin melihat satu jenis harga pangan.
    # Tanpa --attr, program akan mencoba memprediksi seluruh atribut yang tersedia.
    parser.add_argument("--attr", type=str, default=None, help="nama atribut; kosongkan untuk semua")
    # --out menentukan nama file CSV yang akan menyimpan detail prediksi harian.
    # Jika tidak disebutkan, nama hasilnya adalah hasil_prediksi.csv.
    parser.add_argument("--out", type=str, default="hasil_prediksi.csv", help="file CSV keluaran")
    # Semua pilihan dari terminal dibaca dan dimasukkan ke variabel args.
    args = parser.parse_args()

    # Sebelum memprediksi, program memeriksa kualitas dataset lewat common.py.
    # Jika ada tanggal ganda, data terlambat diperbarui, atau nilai kosong,
    # pesannya ditampilkan sebagai peringatan, bukan langsung menghentikan program.
    for msg in check_data_quality():
        print(f"[PERINGATAN] {msg}")

    # Membaca dataset Excel yang sudah diproses agar tanggal dan harga siap digunakan.
    # Variabel df menyimpan tabel harga historis untuk semua atribut pangan.
    df = load_and_preprocess_data()
    # Menentukan daftar atribut yang akan diprediksi.
    # Jika pengguna memilih --attr, daftar ini hanya berisi atribut tersebut.
    # Jika tidak, yang dipakai adalah atribut dikenal yang memang ada di dataset.
    attrs = [args.attr] if args.attr else [a for a in ATTR_TO_GROUP if a in df.columns]
    # Mengecek apakah nama atribut pilihan pengguna benar-benar ada di Excel.
    # Kalau tidak ada, program berhenti dan menampilkan nama atribut yang tersedia
    # supaya pengguna bisa memperbaiki penulisannya.
    if args.attr and args.attr not in df.columns:
        raise SystemExit(f"Atribut '{args.attr}' tidak ada. Pilihan: {', '.join(df.columns)}")

    # Bagian ini dijalankan hanya jika pengguna memasukkan --date.
    # Tanggal target diubah menjadi jumlah hari kerja dari tanggal data terakhir,
    # sehingga proses prediksinya tetap menggunakan jumlah langkah (horizon).
    if args.date:
        args.days = steps_to_date(df.index.max(), args.date)
        # Target harus berada setelah data terakhir dan mencakup minimal satu hari kerja.
        # Jika tidak, prediksi tidak memiliki langkah ke depan yang perlu dihitung.
        if args.days < 1:
            raise SystemExit("Tanggal target harus hari kerja setelah data terakhir.")

    # Menampilkan informasi awal di terminal agar jelas data berhenti pada tanggal berapa
    # dan berapa hari kerja yang akan diprediksi oleh program.
    print(f"Data terakhir: {df.index.max():%d %b %Y} | Horizon: {args.days} hari kerja\n")

    # long_rows berisi detail hasil prediksi setiap tanggal untuk disimpan ke CSV.
    # summary berisi satu ringkasan per atribut untuk ditampilkan di terminal.
    long_rows, summary = [], []
    # Program memproses setiap atribut satu per satu supaya hasilnya terpisah
    # berdasarkan jenis komoditas dan model yang dipakai.
    for attr in attrs:
        # Mengambil model dan informasi pendukungnya dari folder model yang tersimpan.
        # Informasi ini antara lain berisi nama model dan ukuran window yang digunakan.
        # Di sini model TIDAK dilatih ulang, hanya dimuat untuk melakukan prediksi.
        try:
            model, info = load_model(attr)
        # Jika file model belum ada atau gagal dibaca, muncul pesan [LEWAT].
        # Program lanjut ke atribut berikutnya supaya kegagalan satu atribut
        # tidak langsung menghentikan prediksi untuk komoditas lain.
        except (FileNotFoundError, RuntimeError) as err:
            print(f"[LEWAT] {attr}: {err}")
            continue
        # Melakukan prediksi secara rekursif menggunakan harga historis atribut tersebut.
        # Rekursif berarti hasil prediksi hari pertama dimasukkan sebagai riwayat
        # untuk membantu memprediksi hari kedua, lalu dilanjutkan hingga horizon selesai.
        # Nilai window diambil dari metadata model, bukan ditentukan sembarangan.
        future = predict_future_recursive(model, df[attr], args.days, info["window"])
        # Mengambil harga terakhir yang benar-benar tercatat di dataset.
        # Harga ini digunakan sebagai pembanding untuk melihat besarnya perubahan prediksi.
        last_price = df[attr].iloc[-1]
        # Menyimpan setiap tanggal hasil prediksi dalam format tabel panjang.
        # Satu baris memuat tanggal, atribut, nama model, dan harga prediksinya.
        # Format ini memudahkan hasil dibuka kembali di Excel atau dianalisis.
        for tgl, harga in future.items():
            long_rows.append({"Tanggal": tgl.date(), "Atribut": attr, "Model": info["model_name"], "Prediksi": harga})
        # Menyiapkan ringkasan akhir untuk satu atribut, berisi nama model, window,
        # harga terakhir, prediksi pada tanggal akhir, dan persentase perubahannya.
        # Persentase = (harga prediksi akhir - harga terakhir) / harga terakhir x 100.
        # Angka positif berarti prediksi naik dan angka negatif berarti turun.
        summary.append({
            "Atribut": attr,
            "Model": f"{info['model_name']} (w={info['window']})",
            "Harga Terakhir": last_price,
            f"Prediksi {future.index[-1]:%d %b}": future.iloc[-1],
            "Perubahan (%)": (future.iloc[-1] - last_price) / last_price * 100,
        })

    # Jika tidak ada satu pun prediksi yang berhasil dibuat, program dihentikan.
    # Pesan ini membantu mengetahui bahwa model hasil training perlu disiapkan dulu.
    if not long_rows:
        raise SystemExit("Tidak ada model yang bisa dipakai. Jalankan 'python train.py' dulu.")

    # Semua detail prediksi harian disimpan ke file CSV sesuai nama pada --out.
    # index=False digunakan agar nomor baris DataFrame tidak ikut menjadi kolom tambahan.
    pd.DataFrame(long_rows).to_csv(args.out, index=False)
    # Mengatur lebar tampilan tabel di terminal supaya ringkasannya lebih terbaca.
    pd.set_option("display.width", 200)
    # Menampilkan tabel summary untuk semua atribut yang berhasil diprediksi.
    # Angka ditampilkan dengan dua angka desimal agar mudah dibaca.
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    # Memberi tahu lokasi/nama file CSV yang berisi detail prediksi setiap hari.
    print(f"\nDetail per hari tersimpan di: {args.out}")


# Kondisi ini memastikan main() hanya otomatis berjalan saat predict.py dijalankan
# sebagai program utama, bukan ketika file ini diimpor oleh program Python lain.
# Dengan begitu, fungsi di file ini tidak berjalan tanpa sengaja saat dipanggil.
if __name__ == "__main__":
    main()
