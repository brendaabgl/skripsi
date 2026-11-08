"""
train.py - Pelatihan XGBoost & LightGBM untuk prediksi harga komoditas pangan DKI Jakarta.

Perubahan dari versi sebelumnya (menjawab masukan dosen):
1. Hyperparameter TIDAK lagi di-hardcode (n_estimators=100, learning_rate=0.05),
   tetapi dicari otomatis dengan Optuna (TPE) + TimeSeriesSplit cross-validation
   yang HANYA memakai data latih (70% awal data), sehingga data uji tidak bocor.
2. Pemilihan model & input window terbaik memakai skor CV (data latih), bukan MAE data uji.
3. Metrik evaluasi: MAE, RMSE, dan R2.
4. R2 yang tidak terdefinisi (data uji konstan -> variansi 0) dicatat sebagai NaN,
   bukan 0, supaya tidak menyesatkan.
5. Prediksi periode uji disimpan (test_predictions.csv) untuk grafik aktual vs prediksi.
6. Evaluasi prediksi REKURSIF multi-hari (1, 5, 22 hari kerja) -> eval_horizon.csv,
   plus pemeriksaan otomatis (fitur prediksi = fitur training, tanpa kebocoran) -> validation_report.json.

Cara pakai:
    python train.py                 # default: 40 percobaan Optuna per kombinasi
    python train.py --models xgboost  # tanpa LightGBM (jika DLL diblokir Windows)
    python train.py --trials 15     # lebih cepat (untuk uji coba)
    python train.py --trials 80     # lebih teliti (lebih lama)
"""
# PENJELASAN: Library argparse dipakai agar kita bisa mengatur proses training lewat perintah terminal, misalnya jumlah percobaan dan model yang ingin dilatih.
import argparse
# PENJELASAN: JSON digunakan untuk menyimpan informasi model dan laporan validasi supaya hasil training bisa dibaca lagi oleh program lain.
import json
# PENJELASAN: OS membantu membuat dan memeriksa folder penyimpanan hasil training.
import os
# PENJELASAN: Warnings dipakai untuk mengatur pesan peringatan dari library selama training berlangsung.
import warnings

# PENJELASAN: Joblib tersedia sebagai alat bantu pemrosesan; pada script ini pemrosesan paralel memakai Parallel dan delayed dari joblib.
import joblib
# PENJELASAN: NumPy digunakan untuk perhitungan angka, seperti variansi, galat, dan konversi hasil prediksi menjadi array.
import numpy as np
# PENJELASAN: Pandas dipakai untuk mengolah tabel harga, memisahkan data berdasarkan tanggal, dan menyimpan hasil dalam CSV.
import pandas as pd
# PENJELASAN: Optuna mencari kombinasi hyperparameter yang menghasilkan kesalahan prediksi paling kecil, jadi pengaturannya tidak ditentukan satu per satu secara manual.
import optuna
# PENJELASAN: Parallel dan delayed memungkinkan beberapa atribut harga diproses bersamaan. Ini mempercepat training tanpa mengubah urutan waktu di dalam masing-masing data.
from joblib import Parallel, delayed
# PENJELASAN: Tiga fungsi metrik ini digunakan untuk mengukur seberapa dekat prediksi model dengan harga aktual.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# PENJELASAN: TimeSeriesSplit membagi data latih menjadi beberapa tahap berurutan berdasarkan waktu, sehingga validasi tidak memakai data masa depan untuk melatih masa lalu.
from sklearn.model_selection import TimeSeriesSplit
# PENJELASAN: XGBRegressor adalah algoritma XGBoost yang dilatih untuk memprediksi angka berupa persentase perubahan harga.
from xgboost import XGBRegressor

# PENJELASAN: Fungsi dari common.py dipakai bersama agar cara membuat fitur, mengecek kesesuaian fitur, membaca dataset, memprediksi, dan menyimpan model tetap konsisten.
from common import (
    ATTR_TO_GROUP, build_features, check_feature_parity, load_and_preprocess_data,
    predict_future_recursive, save_model_files,
)

try:  # LightGBM opsional: di Windows dengan kebijakan keamanan ketat DLL-nya bisa diblokir
    # PENJELASAN: LightGBM dicoba dimuat di sini. Kalau gagal karena library belum terpasang atau masalah DLL di Windows, program menyimpan informasinya untuk ditangani nanti.
    from lightgbm import LGBMRegressor
except (ImportError, OSError) as _e:
    LGBMRegressor = None
    _LGBM_ERROR = str(_e)

# PENJELASAN: Pesan peringatan umum disembunyikan agar tampilan terminal tidak terlalu penuh. Ini tidak berarti semua masalah program otomatis selesai.
warnings.filterwarnings("ignore")
# PENJELASAN: Log Optuna dibatasi agar hanya peringatan penting yang terlihat, bukan detail setiap langkah percobaan.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------------------------------------------------------
# KONFIGURASI (semua angka eksperimen dikumpulkan di sini, bukan tersebar di kode)
# ----------------------------------------------------------------------------
# PENJELASAN: 70% data paling awal dipakai untuk latih dan 30% sisanya untuk uji, sesuai urutan tanggal.
TRAIN_RATIO = 0.70              # pembagian kronologis 70:30 (70% awal = latih, 30% akhir = uji)
# PENJELASAN: Panjang riwayat yang dibandingkan adalah 3, 5, 10, dan 22 hari kerja.
WINDOW_SIZES = [3, 5, 10, 22]   # 3 hari, 1 minggu, 2 minggu, 1 bulan (hari kerja)
# PENJELASAN: Validasi silang dibuat menjadi tiga lipatan data deret waktu.
CV_SPLITS = 3                   # jumlah lipatan TimeSeriesSplit
# PENJELASAN: Jumlah percobaan Optuna standar adalah 40 untuk setiap kombinasi yang diuji.
DEFAULT_TRIALS = 40             # percobaan Optuna per (atribut, model, window)
# PENJELASAN: Seed menjaga proses yang memakai angka acak lebih mudah diulang dengan pengaturan yang sama.
SEED = 42
# PENJELASAN: Model terpilih dievaluasi lagi untuk perkiraan 1, 5, dan 22 hari kerja ke depan.
EVAL_HORIZONS = [1, 5, 22]       # evaluasi prediksi rekursif: 1 hari, 1 minggu, 1 bulan kerja
# PENJELASAN: Daftar algoritma yang ingin dibandingkan selama training.
ALL_MODELS = ["XGBoost", "LightGBM"]

# Loss yang dipakai kedua algoritma. "mse" (kuadrat) membuat model berani memprediksi perubahan harga. Loss "mae"
# pada harga yang sering datar menghasilkan prediksi perubahan nol di hampir semua hari (prediksi jadi garis datar).
# PENJELASAN: Loss kuadrat dipakai untuk proses pembelajaran model. Perlu dibedakan: loss training ini bukan metrik utama untuk memilih model, karena pemilihan menggunakan CV MAE.
LOSS = "mse"

# Titik awal pencarian = pengaturan lama (supaya hasil tuning minimal sebaik versi lama di CV)
# PENJELASAN: Parameter acuan lama tetap dicoba sebagai titik awal pencarian Optuna, lalu dibandingkan dengan percobaan lainnya.
BASELINE_PARAMS = {"n_estimators": 100, "learning_rate": 0.05}

# ----------------------------------------------------------------------------
# METRIK EVALUASI
# ----------------------------------------------------------------------------
# PENJELASAN: Fungsi ini menghitung R² dengan pengecekan khusus. Kalau nilai aktual semuanya sama, R² dianggap tidak terdefinisi dan disimpan sebagai NaN, bukan dipaksakan menjadi angka 0.
def safe_r2(y_true, y_pred):
    """
    R2 tidak terdefinisi bila data aktual konstan (variansi = 0): penyebutnya nol.
    sklearn mengembalikan 0.0 pada kasus ini, yang menyesatkan -> kita ganti NaN.
    """
    # PENJELASAN: Variansi nol berarti harga aktual yang dibandingkan tidak berubah sama sekali. Dalam kondisi ini perhitungan R² biasa tidak memberikan informasi yang dapat ditafsirkan secara tepat.
    if np.var(np.asarray(y_true, dtype=float)) == 0:
        return float("nan")
    return float(r2_score(y_true, y_pred))


# PENJELASAN: Fungsi ini merangkum tiga nilai evaluasi: MAE (rata-rata selisih absolut), RMSE (lebih menekankan kesalahan besar), dan R² (kemampuan menjelaskan variasi nilai aktual).
def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": safe_r2(y_true, y_pred),
    }


# ----------------------------------------------------------------------------
# DATA (pemuatan data & fitur ada di common.py)
# ----------------------------------------------------------------------------
# PENJELASAN: Batas training dan testing ditentukan dari seluruh tanggal asli, sehingga tanggal pemisah tetap sama walaupun window fitur yang dipakai berbeda.
def split_dates(index, train_ratio=TRAIN_RATIO):
    """Tanggal batas pembagian kronologis 70:30, dihitung dari seluruh deret (sama untuk semua window & atribut)."""
    # PENJELASAN: Jumlah data latih dihitung dari 70% seluruh observasi; hasilnya dibulatkan ke bawah menjadi bilangan bulat.
    k = int(len(index) * train_ratio)
    return index[k - 1], index[k]    # (hari terakhir latih, hari pertama uji)


# PENJELASAN: Baris fitur dipisahkan menurut tanggal. Kolom target dan prev_price tidak ikut dimasukkan sebagai input model karena keduanya hanya digunakan untuk pembelajaran target dan mengembalikan prediksi ke rupiah.
def split_train_test(df_feat, train_end, test_start):
    """Pembagian kronologis 70:30: baris dengan tanggal <= train_end = latih, >= test_start = uji."""
    # PENJELASAN: Bagian train memuat tanggal sampai batas training, sedangkan test dimulai sesudah batas itu. Pemisahan seperti ini membantu menjaga evaluasi sesuai alur waktu.
    train, test = df_feat.loc[df_feat.index <= train_end], df_feat.loc[df_feat.index >= test_start]
    drop = ["target", "prev_price"]
    return (
        train.drop(columns=drop), train["target"], train["prev_price"],
        test.drop(columns=drop), test["target"], test["prev_price"],
    )


# ----------------------------------------------------------------------------
# MODEL & PENCARIAN HYPERPARAMETER (OPTUNA + TimeSeriesSplit)
# ----------------------------------------------------------------------------
# PENJELASAN: Di sini program membuat objek model berdasarkan nama algoritma dan parameter hasil tuning. Proses training sebenarnya baru terjadi ketika metode fit() dipanggil.
def make_model(name, params):
    """Model dengan loss kuadrat (lihat LOSS). Hyperparameter lainnya dicari Optuna."""
    # PENJELASAN: Parameter disalin lebih dahulu agar isi parameter asli tidak berubah ketika ada penyesuaian.
    params = dict(params)
    # PENJELASAN: Jika ada parameter bernama loss, parameter tersebut dibuang dari dict karena jenis objective sudah ditentukan secara khusus untuk masing-masing algoritma.
    params.pop("loss", None)
    # PENJELASAN: Jika algoritmanya XGBoost, program membuat XGBRegressor dengan objective kuadrat; selain itu, dibuat LGBMRegressor dengan objective l2.
    if name == "XGBoost":
        return XGBRegressor(objective="reg:squarederror", random_state=SEED, n_jobs=1, verbosity=0, **params)
    return LGBMRegressor(objective="l2", random_state=SEED, n_jobs=1, verbose=-1, subsample_freq=1, **params)


# PENJELASAN: Fungsi ini menentukan rentang nilai parameter yang boleh dicoba Optuna. Setiap trial memilih kombinasi berbeda di dalam rentang tersebut.
def suggest_params(trial, name):
    """Ruang pencarian hyperparameter tiap algoritma (model dangkal & teregularisasi: datanya hanya ~650 baris)."""
    # PENJELASAN: Parameter umum yang dituning mencakup jumlah pohon, learning rate, proporsi data/baris dan fitur, regularisasi, serta kedalaman pohon.
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=25),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
    }
    if name == "XGBoost":
        # PENJELASAN: XGBoost memakai min_child_weight untuk mengendalikan pembentukan cabang berdasarkan bobot minimum data pada node.
        params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 20)
    else:
        # PENJELASAN: LightGBM memakai num_leaves untuk membatasi kompleksitas jumlah daun pohon.
        params["num_leaves"] = trial.suggest_int("num_leaves", 4, 32)
        params["min_child_samples"] = trial.suggest_int("min_child_samples", 10, 60)
    return params


# PENJELASAN: Fungsi ini menghasilkan prediksi out-of-fold: model dilatih pada bagian data yang lebih awal, lalu memprediksi lipatan validasi setelahnya. Ini dilakukan berulang sesuai jumlah lipatan.
def _oof_predictions(name, params, X, y, tscv):
    """Prediksi out-of-fold (persentase perubahan) dari TimeSeriesSplit; hanya baris validasi yang terisi."""
    idx, preds = [], []
    # PENJELASAN: Setiap perulangan memberikan indeks untuk subset training dan subset validasi yang berbeda, tetapi tetap mengikuti urutan waktu.
    for tr_idx, va_idx in tscv.split(X):
        model = make_model(name, params)
        # PENJELASAN: Baris fit() inilah salah satu tempat terjadinya pelatihan model pada data bagian training di setiap lipatan cross-validation.
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        idx.append(va_idx)
        preds.append(np.asarray(model.predict(X.iloc[va_idx]), dtype=float))
    return np.concatenate(idx), np.concatenate(preds)


# PENJELASAN: Fungsi tuning memilih hyperparameter berdasarkan CV MAE dalam satuan rupiah. Yang dipakai hanya data training, sehingga skor testing tidak menentukan pilihan model.
def tune_model(name, X_train, y_train, prev_train, n_trials):
    """
    Cari hyperparameter terbaik (loss tetap kuadrat). Skor = MAE (Rupiah) 1-hari out-of-fold pada TimeSeriesSplit,
    HANYA memakai data latih. Mengembalikan (params, cv_mae).
    """
    # PENJELASAN: Objek TimeSeriesSplit disiapkan untuk membagi data training ke dalam tiga lipatan validasi berurutan.
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    prev = prev_train.to_numpy(dtype=float)
    y = y_train.to_numpy(dtype=float)

    # PENJELASAN: Objective adalah nilai yang ingin diperkecil oleh Optuna. Setiap trial membuat model, memvalidasinya, lalu mengembalikan angka kesalahannya.
    def objective(trial):
        params = suggest_params(trial, name)
        idx, pred = _oof_predictions(name, params, X_train, y_train, tscv)
        # PENJELASAN: Prediksi model berupa perubahan relatif. Selisih perubahan tersebut dikalikan harga sebelumnya agar galat dihitung dalam rupiah, lalu dirata-ratakan sebagai CV MAE.
        return float(np.mean(np.abs(pred - y[idx]) * prev[idx]))

    # PENJELASAN: Study Optuna dibuat untuk mencari nilai objective yang paling kecil menggunakan strategi TPE.
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    # PENJELASAN: Pengaturan dasar dimasukkan sebagai trial awal untuk dibandingkan dengan opsi lainnya.
    study.enqueue_trial(BASELINE_PARAMS)  # pengaturan lama dicoba pertama sebagai pembanding
    # PENJELASAN: Optuna menjalankan objective berulang sesuai jumlah trial yang dipilih pengguna.
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)


# ----------------------------------------------------------------------------
# EVALUASI PREDIKSI REKURSIF MULTI-HARI (sama dengan cara forecasting di aplikasi)
# ----------------------------------------------------------------------------
# PENJELASAN: Bagian ini menguji kemampuan model memprediksi lebih dari satu hari. Dari beberapa titik awal pada periode uji, prediksi dibuat secara rekursif dan dibandingkan dengan harga aktual.
def evaluate_horizons(model, series, window_size, first_test_date):
    """
    Evaluasi jujur untuk prediksi ke depan: dari setiap hari asal (origin) di periode uji, model
    meramal H hari ke depan SECARA REKURSIF (hanya memakai harga sampai hari asal), lalu dibandingkan
    dengan harga aktual. `model` harus dilatih hanya pada data latih.
    """
    max_h = max(EVAL_HORIZONS)
    n = len(series)
    first_origin = max(series.index.get_loc(first_test_date) - 1, 0)   # hari terakhir sebelum periode uji
    # PENJELASAN: Siapkan penampung hasil aktual dan prediksi pada setiap horizon yang ingin diuji.
    pairs = {h: [] for h in EVAL_HORIZONS}          # h -> daftar (aktual, prediksi)
    # PENJELASAN: Perulangan berpindah dari satu tanggal asal prediksi ke tanggal berikutnya, sehingga performa tidak dinilai hanya dari satu titik waktu.
    for p in range(first_origin, n - 1):
        steps = min(max_h, n - 1 - p)
        # PENJELASAN: Fungsi dari common.py digunakan agar cara meramal beberapa hari ke depan sama dengan mekanisme prediksi yang dipakai aplikasi.
        fut = predict_future_recursive(
            model, series.iloc[: p + 1], steps, window_size, future_dates=list(series.index[p + 1: p + 1 + steps])
        )
        # PENJELASAN: Hasil hari ke-1, ke-5, dan ke-22 dikumpulkan hanya jika horizon tersebut masih memiliki data aktual untuk dibandingkan.
        for h in EVAL_HORIZONS:
            if h <= steps:
                pairs[h].append((series.iloc[p + h], fut.iloc[h - 1]))
    out = {}
    for h, rows in pairs.items():
        if not rows:
            continue
        a, f = (np.array(x, dtype=float) for x in zip(*rows))
        out[h] = {
            "N": len(a), "MAE": float(np.mean(np.abs(a - f))), "RMSE": float(np.sqrt(np.mean((a - f) ** 2))),
        }
    return out


# ----------------------------------------------------------------------------
# PIPELINE PER ATRIBUT (dijalankan paralel antar atribut)
# ----------------------------------------------------------------------------
# PENJELASAN: Ini proses utama untuk satu atribut pangan. Program mencoba berbagai window dan algoritma, memilih kandidat terbaik, memvalidasi hasilnya, lalu melatih model final.
def process_attribute(attr, series, n_trials, model_names):
    # PENJELASAN: Nama atribut dicocokkan dengan kelompok komoditas, misalnya cabai merah atau beras, agar hasil laporan lebih mudah dikelompokkan.
    group_name = ATTR_TO_GROUP.get(attr, "Lainnya")
    rows, candidates = [], []
    # PENJELASAN: Satu batas tanggal training-testing ditentukan untuk atribut ini sebelum seluruh kombinasi model dievaluasi.
    train_end, test_start = split_dates(series.index)

    # PENJELASAN: Masing-masing panjang window diuji satu per satu agar model dapat dibandingkan berdasarkan berapa banyak riwayat harga yang dipakai sebagai fitur.
    for w in WINDOW_SIZES:
        # PENJELASAN: Riwayat harga diubah menjadi fitur numerik dan target perubahan harga. Perhitungan fitur ini berasal dari common.py.
        df_feat = build_features(series, window_size=w)
        X_train, y_train, p_train, X_test, y_test, p_test = split_train_test(df_feat, train_end, test_start)

        # harga aktual hari uji
        # PENJELASAN: Target testing awalnya berupa persentase perubahan; rumus ini mengembalikannya menjadi harga aktual dalam rupiah.
        y_test_price = pd.Series(p_test.to_numpy() * (1.0 + y_test.to_numpy()), index=y_test.index)

        # PENJELASAN: Di setiap window, XGBoost dan/atau LightGBM diproses sesuai pilihan dari terminal.
        for name in model_names:
            # PENJELASAN: Hyperparameter terbaik dan skor CV MAE dicari khusus untuk algoritma serta window yang sedang diuji.
            best_params, cv_mae = tune_model(name, X_train, y_train, p_train, n_trials)

            # model untuk EVALUASI: dilatih hanya pada data latih, prediksi = persentase perubahan
            model = make_model(name, best_params)
            # PENJELASAN: Model kandidat dilatih hanya dengan data training supaya evaluasi terhadap data testing tetap terpisah.
            model.fit(X_train, y_train)
            # PENJELASAN: Model kandidat memprediksi persentase perubahan harga pada setiap baris data testing.
            pred_ret = np.asarray(model.predict(X_test), dtype=float)
            # PENJELASAN: Kembalikan hasil perubahan relatif ke nilai harga dalam rupiah.
            pred = p_test.to_numpy(dtype=float) * (1.0 + pred_ret)   # kembali ke Rupiah

            # PENJELASAN: Nilai MAE, RMSE, dan R² dihitung dari harga aktual dan harga prediksi pada periode testing.
            metrics = evaluate(y_test_price, pred)

            # PENJELASAN: Satu baris hasil evaluasi disiapkan untuk disimpan ke eval_results.csv, lengkap dengan nama algoritma, window, metrik, dan status Dipilih.
            rows.append({
                "Kelompok": group_name,
                "Atribut": attr,
                "Model": name,
                "Input Window (Hari)": w,
                "CV MAE": cv_mae,
                **metrics,
                "Dipilih": False,
            })
            # PENJELASAN: Detail kandidat disimpan sementara supaya program bisa menentukan kombinasi terbaik setelah semua percobaan selesai.
            candidates.append({
                "row_idx": len(rows) - 1,
                "name": name,
                "window": w,
                "params": best_params,
                "cv_mae": cv_mae,
                "metrics": metrics,
                "y_test": y_test_price,
                "pred": pred,
                "model": model,
                "n_train": len(X_train),
                "train_last_date": X_train.index.max(),
            })

    # Pilih kombinasi terbaik berdasarkan CV MAE (data latih). Seri -> window lebih kecil, lalu urutan model.
    # PENJELASAN: Kandidat terbaik dipilih dari CV MAE paling kecil, bukan dari nilai testing. Jika seri, program mengutamakan window lebih kecil lalu urutan nama model yang dipilih.
    best = min(candidates, key=lambda c: (round(c["cv_mae"], 10), c["window"], model_names.index(c["name"])))
    # PENJELASAN: Kombinasi yang menang diberi tanda Dipilih=True sehingga website dapat mengenalinya sebagai hasil model terpilih.
    rows[best["row_idx"]]["Dipilih"] = True

    # --- Pemeriksaan kebenaran (otomatis) ---
    # PENJELASAN: Pemeriksaan ini mencocokkan hasil fitur training dan fitur prediksi pada tanggal yang sama.
    check_feature_parity(series, best["window"])                       # fitur prediksi == fitur training
    # PENJELASAN: Tanggal pertama pengujian dicek agar berada setelah akhir data latih.
    leak_ok = bool(best["y_test"].index.min() > best["train_last_date"])  # data uji sesudah data latih
    if not leak_ok:
        raise ValueError(f"{attr}: data uji tidak berada setelah data latih (kebocoran).")

    # Evaluasi prediksi rekursif multi-hari (model dilatih hanya pada data latih)
    # PENJELASAN: Model terpilih diuji kembali untuk prediksi rekursif pada horizon 1, 5, dan 22 hari kerja.
    hz = evaluate_horizons(best["model"], series, best["window"], best["y_test"].index.min())
    horizon_rows = [{
        "Kelompok": group_name, "Atribut": attr, "Model": best["name"], "Input Window (Hari)": best["window"],
        "Horizon (Hari Kerja)": h, **m,
    } for h, m in hz.items()]

    # Model FINAL untuk forecasting masa depan: parameter terbaik, dilatih dengan seluruh data
    # PENJELASAN: Setelah evaluasi selesai, fitur dibangun lagi dari seluruh data untuk menyiapkan model final yang akan digunakan meramal masa depan.
    df_feat = build_features(series, window_size=best["window"])
    final_model = make_model(best["name"], best["params"])
    # PENJELASAN: Model final dilatih menggunakan seluruh data yang tersedia dengan kombinasi parameter terbaik. Model final inilah yang disimpan dan nanti digunakan oleh predict.py atau website.
    final_model.fit(df_feat.drop(columns=["target", "prev_price"]), df_feat["target"])

    # PENJELASAN: Metadata model memuat algoritma terpilih, window, parameter, dan metrik evaluasi. Informasi ini disimpan agar program lain tahu cara memakai model yang benar.
    info = {
        "model_name": best["name"],
        "window": best["window"],
        "params": best["params"],
        "cv_mae": best["cv_mae"],
        "mae": best["metrics"]["MAE"],
        "rmse": best["metrics"]["RMSE"],
        "r2": best["metrics"]["R2"],
    }
    # PENJELASAN: Tabel ini menyimpan pasangan harga aktual dan prediksi pada data testing untuk keperluan grafik serta pemeriksaan hasil.
    test_pred = pd.DataFrame({
        "Tanggal": best["y_test"].index,
        "Atribut": attr,
        "Model": best["name"],
        "Aktual": best["y_test"].values,
        "Prediksi": best["pred"],
    })
    # PENJELASAN: Ringkasan pemeriksaan konsistensi fitur dan pemisahan tanggal disiapkan sebagai isi validation_report.json.
    check = {
        "Atribut": attr, "Fitur prediksi = fitur training": True, "Data uji setelah data latih": leak_ok,
        "Baris latih": best["n_train"], "Baris uji": int(len(best["y_test"])),
        "Latih s.d.": str(best["train_last_date"].date()), "Uji mulai": str(best["y_test"].index.min().date()),
    }
    # PENJELASAN: Semua hasil untuk satu atribut dikembalikan ke pipeline utama agar dapat digabung dengan hasil atribut lainnya.
    return attr, rows, final_model, info, test_pred, horizon_rows, check


# PENJELASAN: Fungsi ini mengatur keseluruhan training: membaca dataset, memproses seluruh atribut, menyimpan model, lalu membuat file hasil evaluasi.
def run_training_pipeline(n_trials=DEFAULT_TRIALS, model_names=None, n_jobs=4):
    model_names = list(model_names or ALL_MODELS)
    # PENJELASAN: Folder models dibuat bila belum tersedia sebagai tempat menyimpan model yang sudah dilatih.
    os.makedirs("models", exist_ok=True)
    # PENJELASAN: Dataset dibaca dan dipraproses lewat fungsi bersama agar data yang digunakan training tersusun berdasarkan tanggal.
    df_data = load_and_preprocess_data()

    print("=== MEMULAI TRAINING + TUNING OPTUNA ===")
    print(f"Atribut: {df_data.shape[1]} | Model: {model_names} | Window: {WINDOW_SIZES} | Percobaan Optuna: {n_trials} | CV: {CV_SPLITS} lipatan")

    # backend 'threading': tanpa proses terpisah, jadi aman dari error BrokenProcessPool / WinError 6 di Windows
    # PENJELASAN: Setiap atribut dijalankan secara paralel menggunakan threading. Jumlah pekerjaan bersamaan ditentukan oleh opsi --jobs.
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=5)(
        delayed(process_attribute)(attr, df_data[attr], n_trials, model_names) for attr in df_data.columns
    )

    all_rows, all_test_pred, best_params_dump, all_horizon, all_checks = [], [], {}, [], []
    # PENJELASAN: Hasil dari setiap atribut dikumpulkan satu per satu untuk disimpan dan digabung menjadi laporan keseluruhan.
    for attr, rows, final_model, info, test_pred, horizon_rows, check in results:
        safe_filename = attr.replace(" ", "_")
        # PENJELASAN: Simpan model final bersama metadata agar siap dimuat tanpa training ulang oleh aplikasi.
        save_model_files(final_model, info, attr)   # format bawaan library (JSON/TXT), aman lintas versi
        all_rows.extend(rows)
        all_test_pred.append(test_pred)
        all_horizon.extend(horizon_rows)
        all_checks.append(check)
        best_params_dump[attr] = {k: info[k] for k in ("model_name", "window", "params", "cv_mae")}

    # PENJELASAN: eval_results.csv berisi hasil perbandingan seluruh kombinasi algoritma dan window beserta nilai evaluasinya.
    pd.DataFrame(all_rows).to_csv("eval_results.csv", index=False)
    # PENJELASAN: test_predictions.csv berisi harga aktual dan prediksi periode testing untuk tiap atribut model terpilih.
    pd.concat(all_test_pred).to_csv("test_predictions.csv", index=False)
    # PENJELASAN: eval_horizon.csv menyimpan nilai kesalahan prediksi rekursif pada berbagai horizon hari kerja.
    pd.DataFrame(all_horizon).to_csv("eval_horizon.csv", index=False)
    # PENJELASAN: validation_report.json menyimpan hasil pengecekan kesamaan fitur dan urutan data training-testing.
    with open("validation_report.json", "w", encoding="utf-8") as f:
        json.dump(all_checks, f, indent=2, ensure_ascii=False)
    # PENJELASAN: best_params.json menyimpan konfigurasi terbaik setiap atribut agar hasil tuning mudah ditinjau lagi.
    with open("best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_dump, f, indent=2, ensure_ascii=False)

    print("=== TRAINING SELESAI! Jalankan: streamlit run app.py ===")
    print("Pemeriksaan otomatis: fitur prediksi = fitur training dan tanpa kebocoran data -> LULUS untuk", len(all_checks), "atribut")
    print("File: eval_results.csv, eval_horizon.csv, test_predictions.csv, validation_report.json, best_params.json, folder models/")


# PENJELASAN: Bagian ini hanya berjalan ketika train.py dipanggil langsung dari terminal, bukan ketika modul di-import oleh file Python lain.
if __name__ == "__main__":
    # PENJELASAN: Parser membaca pilihan yang diberikan pengguna dari terminal untuk mengatur proses training.
    parser = argparse.ArgumentParser()
    # PENJELASAN: Opsi --trials mengatur jumlah percobaan tuning. Lebih banyak trial berarti ruang pencarian dicoba lebih luas, tetapi proses biasanya lebih lama.
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="jumlah percobaan Optuna")
    # PENJELASAN: Opsi --models menentukan apakah dua algoritma dilatih sekaligus atau hanya satu algoritma.
    parser.add_argument("--models", choices=["both", "xgboost", "lightgbm"], default="both",
                        help="algoritma yang dilatih (pakai xgboost jika LightGBM diblokir Windows)")
    # PENJELASAN: Opsi --jobs menentukan jumlah atribut yang dijalankan paralel; nilai 1 dapat dipakai bila komputer kesulitan memproses banyak pekerjaan sekaligus.
    parser.add_argument("--jobs", type=int, default=4, help="jumlah atribut yang diproses bersamaan (1 = paling aman)")
    args = parser.parse_args()

    # PENJELASAN: Pilihan teks dari terminal diubah menjadi daftar nama algoritma yang akan diproses.
    chosen = {"both": ALL_MODELS, "xgboost": ["XGBoost"], "lightgbm": ["LightGBM"]}[args.models]
    # PENJELASAN: Program menghentikan training jika LightGBM dipilih tetapi library-nya tidak berhasil dimuat, lalu menunjukkan alternatif perintah.
    if "LightGBM" in chosen and LGBMRegressor is None:
        raise SystemExit(
            "LightGBM tidak bisa dimuat di komputer ini:\n  " + _LGBM_ERROR +
            "\nJalankan: python train.py --models xgboost   (atau latih di Google Colab)"
        )
    # PENJELASAN: Perintah terakhir menjalankan pipeline training dengan pengaturan sesuai input terminal.
    run_training_pipeline(n_trials=args.trials, model_names=chosen, n_jobs=args.jobs)