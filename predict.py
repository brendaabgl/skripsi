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
import argparse

import pandas as pd

from common import (
    ATTR_TO_GROUP, check_data_quality, load_and_preprocess_data, load_model, predict_future_recursive, steps_to_date,
)


def main():
    parser = argparse.ArgumentParser(description="Prediksi harga pangan dari model yang sudah dilatih")
    parser.add_argument("--days", type=int, default=5, help="jumlah hari kerja ke depan (default 5)")
    parser.add_argument("--date", type=str, default=None, help="tanggal target (YYYY-MM-DD); menggantikan --days")
    parser.add_argument("--attr", type=str, default=None, help="nama atribut; kosongkan untuk semua")
    parser.add_argument("--out", type=str, default="hasil_prediksi.csv", help="file CSV keluaran")
    args = parser.parse_args()

    for msg in check_data_quality():
        print(f"[PERINGATAN] {msg}")

    df = load_and_preprocess_data()
    attrs = [args.attr] if args.attr else [a for a in ATTR_TO_GROUP if a in df.columns]
    if args.attr and args.attr not in df.columns:
        raise SystemExit(f"Atribut '{args.attr}' tidak ada. Pilihan: {', '.join(df.columns)}")

    if args.date:
        args.days = steps_to_date(df.index.max(), args.date)
        if args.days < 1:
            raise SystemExit("Tanggal target harus hari kerja setelah data terakhir.")

    print(f"Data terakhir: {df.index.max():%d %b %Y} | Horizon: {args.days} hari kerja\n")

    long_rows, summary = [], []
    for attr in attrs:
        try:
            model, info = load_model(attr)
        except (FileNotFoundError, RuntimeError) as err:
            print(f"[LEWAT] {attr}: {err}")
            continue
        future = predict_future_recursive(model, df[attr], args.days, info["window"])
        last_price = df[attr].iloc[-1]
        for tgl, harga in future.items():
            long_rows.append({"Tanggal": tgl.date(), "Atribut": attr, "Model": info["model_name"], "Prediksi": harga})
        summary.append({
            "Atribut": attr,
            "Model": f"{info['model_name']} (w={info['window']})",
            "Harga Terakhir": last_price,
            f"Prediksi {future.index[-1]:%d %b}": future.iloc[-1],
            "Perubahan (%)": (future.iloc[-1] - last_price) / last_price * 100,
        })

    if not long_rows:
        raise SystemExit("Tidak ada model yang bisa dipakai. Jalankan 'python train.py' dulu.")

    pd.DataFrame(long_rows).to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"\nDetail per hari tersimpan di: {args.out}")


if __name__ == "__main__":
    main()
