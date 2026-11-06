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
import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import optuna
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from common import (
    ATTR_TO_GROUP, build_features, check_feature_parity, load_and_preprocess_data,
    predict_future_recursive, save_model_files,
)

try:  # LightGBM opsional: di Windows dengan kebijakan keamanan ketat DLL-nya bisa diblokir
    from lightgbm import LGBMRegressor
except (ImportError, OSError) as _e:
    LGBMRegressor = None
    _LGBM_ERROR = str(_e)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------------------------------------------------------
# KONFIGURASI (semua angka eksperimen dikumpulkan di sini, bukan tersebar di kode)
# ----------------------------------------------------------------------------
TRAIN_RATIO = 0.70              # pembagian kronologis 70:30 (70% awal = latih, 30% akhir = uji)
WINDOW_SIZES = [3, 5, 10, 22]   # 3 hari, 1 minggu, 2 minggu, 1 bulan (hari kerja)
CV_SPLITS = 3                   # jumlah lipatan TimeSeriesSplit
DEFAULT_TRIALS = 40             # percobaan Optuna per (atribut, model, window)
SEED = 42
EVAL_HORIZONS = [1, 5, 22]       # evaluasi prediksi rekursif: 1 hari, 1 minggu, 1 bulan kerja
ALL_MODELS = ["XGBoost", "LightGBM"]

# Loss yang dipakai kedua algoritma. "mse" (kuadrat) membuat model berani memprediksi perubahan harga. Loss "mae"
# pada harga yang sering datar menghasilkan prediksi perubahan nol di hampir semua hari (prediksi jadi garis datar).
LOSS = "mse"

# Titik awal pencarian = pengaturan lama (supaya hasil tuning minimal sebaik versi lama di CV)
BASELINE_PARAMS = {"n_estimators": 100, "learning_rate": 0.05}

# ----------------------------------------------------------------------------
# METRIK EVALUASI
# ----------------------------------------------------------------------------
def safe_r2(y_true, y_pred):
    """
    R2 tidak terdefinisi bila data aktual konstan (variansi = 0): penyebutnya nol.
    sklearn mengembalikan 0.0 pada kasus ini, yang menyesatkan -> kita ganti NaN.
    """
    if np.var(np.asarray(y_true, dtype=float)) == 0:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": safe_r2(y_true, y_pred),
    }


# ----------------------------------------------------------------------------
# DATA (pemuatan data & fitur ada di common.py)
# ----------------------------------------------------------------------------
def split_dates(index, train_ratio=TRAIN_RATIO):
    """Tanggal batas pembagian kronologis 70:30, dihitung dari seluruh deret (sama untuk semua window & atribut)."""
    k = int(len(index) * train_ratio)
    return index[k - 1], index[k]    # (hari terakhir latih, hari pertama uji)


def split_train_test(df_feat, train_end, test_start):
    """Pembagian kronologis 70:30: baris dengan tanggal <= train_end = latih, >= test_start = uji."""
    train, test = df_feat.loc[df_feat.index <= train_end], df_feat.loc[df_feat.index >= test_start]
    drop = ["target", "prev_price"]
    return (
        train.drop(columns=drop), train["target"], train["prev_price"],
        test.drop(columns=drop), test["target"], test["prev_price"],
    )


# ----------------------------------------------------------------------------
# MODEL & PENCARIAN HYPERPARAMETER (OPTUNA + TimeSeriesSplit)
# ----------------------------------------------------------------------------
def make_model(name, params):
    """Model dengan loss kuadrat (lihat LOSS). Hyperparameter lainnya dicari Optuna."""
    params = dict(params)
    params.pop("loss", None)
    if name == "XGBoost":
        return XGBRegressor(objective="reg:squarederror", random_state=SEED, n_jobs=1, verbosity=0, **params)
    return LGBMRegressor(objective="l2", random_state=SEED, n_jobs=1, verbose=-1, subsample_freq=1, **params)


def suggest_params(trial, name):
    """Ruang pencarian hyperparameter tiap algoritma (model dangkal & teregularisasi: datanya hanya ~650 baris)."""
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
        params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 20)
    else:
        params["num_leaves"] = trial.suggest_int("num_leaves", 4, 32)
        params["min_child_samples"] = trial.suggest_int("min_child_samples", 10, 60)
    return params


def _oof_predictions(name, params, X, y, tscv):
    """Prediksi out-of-fold (persentase perubahan) dari TimeSeriesSplit; hanya baris validasi yang terisi."""
    idx, preds = [], []
    for tr_idx, va_idx in tscv.split(X):
        model = make_model(name, params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        idx.append(va_idx)
        preds.append(np.asarray(model.predict(X.iloc[va_idx]), dtype=float))
    return np.concatenate(idx), np.concatenate(preds)


def tune_model(name, X_train, y_train, prev_train, n_trials):
    """
    Cari hyperparameter terbaik (loss tetap kuadrat). Skor = MAE (Rupiah) 1-hari out-of-fold pada TimeSeriesSplit,
    HANYA memakai data latih. Mengembalikan (params, cv_mae).
    """
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    prev = prev_train.to_numpy(dtype=float)
    y = y_train.to_numpy(dtype=float)

    def objective(trial):
        params = suggest_params(trial, name)
        idx, pred = _oof_predictions(name, params, X_train, y_train, tscv)
        return float(np.mean(np.abs(pred - y[idx]) * prev[idx]))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(BASELINE_PARAMS)  # pengaturan lama dicoba pertama sebagai pembanding
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)


# ----------------------------------------------------------------------------
# EVALUASI PREDIKSI REKURSIF MULTI-HARI (sama dengan cara forecasting di aplikasi)
# ----------------------------------------------------------------------------
def evaluate_horizons(model, series, window_size, first_test_date):
    """
    Evaluasi jujur untuk prediksi ke depan: dari setiap hari asal (origin) di periode uji, model
    meramal H hari ke depan SECARA REKURSIF (hanya memakai harga sampai hari asal), lalu dibandingkan
    dengan harga aktual. `model` harus dilatih hanya pada data latih.
    """
    max_h = max(EVAL_HORIZONS)
    n = len(series)
    first_origin = max(series.index.get_loc(first_test_date) - 1, 0)   # hari terakhir sebelum periode uji
    pairs = {h: [] for h in EVAL_HORIZONS}          # h -> daftar (aktual, prediksi)
    for p in range(first_origin, n - 1):
        steps = min(max_h, n - 1 - p)
        fut = predict_future_recursive(
            model, series.iloc[: p + 1], steps, window_size, future_dates=list(series.index[p + 1: p + 1 + steps])
        )
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
def process_attribute(attr, series, n_trials, model_names):
    group_name = ATTR_TO_GROUP.get(attr, "Lainnya")
    rows, candidates = [], []
    train_end, test_start = split_dates(series.index)

    for w in WINDOW_SIZES:
        df_feat = build_features(series, window_size=w)
        X_train, y_train, p_train, X_test, y_test, p_test = split_train_test(df_feat, train_end, test_start)

        # harga aktual hari uji
        y_test_price = pd.Series(p_test.to_numpy() * (1.0 + y_test.to_numpy()), index=y_test.index)

        for name in model_names:
            best_params, cv_mae = tune_model(name, X_train, y_train, p_train, n_trials)

            # model untuk EVALUASI: dilatih hanya pada data latih, prediksi = persentase perubahan
            model = make_model(name, best_params)
            model.fit(X_train, y_train)
            pred_ret = np.asarray(model.predict(X_test), dtype=float)
            pred = p_test.to_numpy(dtype=float) * (1.0 + pred_ret)   # kembali ke Rupiah

            metrics = evaluate(y_test_price, pred)

            rows.append({
                "Kelompok": group_name,
                "Atribut": attr,
                "Model": name,
                "Input Window (Hari)": w,
                "CV MAE": cv_mae,
                **metrics,
                "Dipilih": False,
            })
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
    best = min(candidates, key=lambda c: (round(c["cv_mae"], 10), c["window"], model_names.index(c["name"])))
    rows[best["row_idx"]]["Dipilih"] = True

    # --- Pemeriksaan kebenaran (otomatis) ---
    check_feature_parity(series, best["window"])                       # fitur prediksi == fitur training
    leak_ok = bool(best["y_test"].index.min() > best["train_last_date"])  # data uji sesudah data latih
    if not leak_ok:
        raise ValueError(f"{attr}: data uji tidak berada setelah data latih (kebocoran).")

    # Evaluasi prediksi rekursif multi-hari (model dilatih hanya pada data latih)
    hz = evaluate_horizons(best["model"], series, best["window"], best["y_test"].index.min())
    horizon_rows = [{
        "Kelompok": group_name, "Atribut": attr, "Model": best["name"], "Input Window (Hari)": best["window"],
        "Horizon (Hari Kerja)": h, **m,
    } for h, m in hz.items()]

    # Model FINAL untuk forecasting masa depan: parameter terbaik, dilatih dengan seluruh data
    df_feat = build_features(series, window_size=best["window"])
    final_model = make_model(best["name"], best["params"])
    final_model.fit(df_feat.drop(columns=["target", "prev_price"]), df_feat["target"])

    info = {
        "model_name": best["name"],
        "window": best["window"],
        "params": best["params"],
        "cv_mae": best["cv_mae"],
        "mae": best["metrics"]["MAE"],
        "rmse": best["metrics"]["RMSE"],
        "r2": best["metrics"]["R2"],
    }
    test_pred = pd.DataFrame({
        "Tanggal": best["y_test"].index,
        "Atribut": attr,
        "Model": best["name"],
        "Aktual": best["y_test"].values,
        "Prediksi": best["pred"],
    })
    check = {
        "Atribut": attr, "Fitur prediksi = fitur training": True, "Data uji setelah data latih": leak_ok,
        "Baris latih": best["n_train"], "Baris uji": int(len(best["y_test"])),
        "Latih s.d.": str(best["train_last_date"].date()), "Uji mulai": str(best["y_test"].index.min().date()),
    }
    return attr, rows, final_model, info, test_pred, horizon_rows, check


def run_training_pipeline(n_trials=DEFAULT_TRIALS, model_names=None, n_jobs=4):
    model_names = list(model_names or ALL_MODELS)
    os.makedirs("models", exist_ok=True)
    df_data = load_and_preprocess_data()

    print("=== MEMULAI TRAINING + TUNING OPTUNA ===")
    print(f"Atribut: {df_data.shape[1]} | Model: {model_names} | Window: {WINDOW_SIZES} | Percobaan Optuna: {n_trials} | CV: {CV_SPLITS} lipatan")

    # backend 'threading': tanpa proses terpisah, jadi aman dari error BrokenProcessPool / WinError 6 di Windows
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=5)(
        delayed(process_attribute)(attr, df_data[attr], n_trials, model_names) for attr in df_data.columns
    )

    all_rows, all_test_pred, best_params_dump, all_horizon, all_checks = [], [], {}, [], []
    for attr, rows, final_model, info, test_pred, horizon_rows, check in results:
        safe_filename = attr.replace(" ", "_")
        save_model_files(final_model, info, attr)   # format bawaan library (JSON/TXT), aman lintas versi
        all_rows.extend(rows)
        all_test_pred.append(test_pred)
        all_horizon.extend(horizon_rows)
        all_checks.append(check)
        best_params_dump[attr] = {k: info[k] for k in ("model_name", "window", "params", "cv_mae")}

    pd.DataFrame(all_rows).to_csv("eval_results.csv", index=False)
    pd.concat(all_test_pred).to_csv("test_predictions.csv", index=False)
    pd.DataFrame(all_horizon).to_csv("eval_horizon.csv", index=False)
    with open("validation_report.json", "w", encoding="utf-8") as f:
        json.dump(all_checks, f, indent=2, ensure_ascii=False)
    with open("best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_dump, f, indent=2, ensure_ascii=False)

    print("=== TRAINING SELESAI! Jalankan: streamlit run app.py ===")
    print("Pemeriksaan otomatis: fitur prediksi = fitur training dan tanpa kebocoran data -> LULUS untuk", len(all_checks), "atribut")
    print("File: eval_results.csv, eval_horizon.csv, test_predictions.csv, validation_report.json, best_params.json, folder models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="jumlah percobaan Optuna")
    parser.add_argument("--models", choices=["both", "xgboost", "lightgbm"], default="both",
                        help="algoritma yang dilatih (pakai xgboost jika LightGBM diblokir Windows)")
    parser.add_argument("--jobs", type=int, default=4, help="jumlah atribut yang diproses bersamaan (1 = paling aman)")
    args = parser.parse_args()

    chosen = {"both": ALL_MODELS, "xgboost": ["XGBoost"], "lightgbm": ["LightGBM"]}[args.models]
    if "LightGBM" in chosen and LGBMRegressor is None:
        raise SystemExit(
            "LightGBM tidak bisa dimuat di komputer ini:\n  " + _LGBM_ERROR +
            "\nJalankan: python train.py --models xgboost   (atau latih di Google Colab)"
        )
    run_training_pipeline(n_trials=args.trials, model_names=chosen, n_jobs=args.jobs)