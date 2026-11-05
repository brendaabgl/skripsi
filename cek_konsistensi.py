"""
cek_konsistensi.py - Pastikan semua file hasil training berasal dari run yang SAMA dan tersusun benar.
Jalankan di folder proyek:  python cek_konsistensi.py
"""
import glob
import json
import os

import pandas as pd

masalah = []

# 1) Struktur folder & format model
if os.path.isdir(os.path.join("models", "models")):
    masalah.append("Ada folder models/models/ -> zip diekstrak ke tempat yang salah. Pindahkan isinya ke models/.")
info_json = sorted(glob.glob(os.path.join("models", "*_info.json")))
info_lama = glob.glob(os.path.join("models", "*_info.joblib"))
if not info_json:
    masalah.append("Tidak ada model format baru (*_info.json) di folder models/.")
if info_lama:
    masalah.append(f"Masih ada {len(info_lama)} model format lama (.joblib); hapus agar tidak tercampur.")

# 2) File hasil lain
for f in ["eval_results.csv", "eval_horizon.csv", "test_predictions.csv", "validation_report.json"]:
    if not os.path.exists(f):
        masalah.append(f"File '{f}' tidak ada di folder proyek.")

# 3) eval_results.csv harus cocok dengan model di folder models/ (model, window, MAE uji)
if os.path.exists("eval_results.csv") and info_json:
    e = pd.read_csv("eval_results.csv")
    if "Dipilih" not in e.columns:
        masalah.append("eval_results.csv versi lama (tidak ada kolom 'Dipilih'). Ambil yang baru dari Colab.")
    else:
        sel = e[e["Dipilih"]].set_index("Atribut")
        cocok = 0
        for attr, r in sel.iterrows():
            p = os.path.join("models", attr.replace(" ", "_") + "_info.json")
            if not os.path.exists(p):
                masalah.append(f"Model untuk '{attr}' tidak ada.")
                continue
            info = json.load(open(p, encoding="utf-8"))
            if (info["model_name"] != r["Model"] or int(info["window"]) != int(r["Input Window (Hari)"])
                    or abs(info["mae"] - r["MAE"]) > 1e-6):
                masalah.append(f"'{attr}': eval_results.csv BEDA dengan model (kemungkinan dari run yang berbeda).")
            else:
                cocok += 1
        print(f"Atribut cocok antara eval_results.csv dan models/: {cocok} dari {len(sel)}")

if masalah:
    print("\nPERHATIAN:")
    for m in masalah:
        print(" -", m)
else:
    print("\nAMAN: file hasil training konsisten dan siap dipakai (streamlit run app.py).")