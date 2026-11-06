"""
common.py - Kode yang dipakai bersama oleh train.py, predict.py, dan app.py.

Aturan penting: fungsi build_features() (training) dan predict_future_recursive() (prediksi)
harus membuat fitur yang SAMA PERSIS, makanya keduanya ditaruh di satu file ini.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

DATASET_NAME = "dataset_harga_preprocessed.xlsx"
MODELS_DIR = "models"

# Pemetaan Atribut ke Kelompok Komoditas
ATTR_TO_GROUP = {
    "Beras Kualitas Bawah I": "Beras",
    "Beras Kualitas Bawah II": "Beras",
    "Beras Kualitas Medium I": "Beras",
    "Beras Kualitas Medium II": "Beras",
    "Beras Kualitas Super I": "Beras",
    "Beras Kualitas Super II": "Beras",
    "Daging Ayam Ras Segar": "Daging Ayam",
    "Daging Sapi Kualitas 1": "Daging Sapi",
    "Daging Sapi Kualitas 2": "Daging Sapi",
    "Telur Ayam Ras Segar": "Telur Ayam",
    "Bawang Merah Ukuran Sedang": "Bawang Merah",
    "Bawang Putih Ukuran Sedang": "Bawang Putih",
    "Cabai Merah Besar": "Cabai Merah",
    "Cabai Merah Keriting": "Cabai Merah",
    "Cabai Rawit Hijau": "Cabai Rawit",
    "Cabai Rawit Merah": "Cabai Rawit",
    "Minyak Goreng Kemasan Bermerk 1": "Minyak Goreng",
    "Minyak Goreng Kemasan Bermerk 2": "Minyak Goreng",
    "Gula Pasir Kualitas Premium": "Gula Pasir",
    "Gula Pasir Lokal": "Gula Pasir",
}


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
def find_dataset_path():
    """Cari file Excel di folder 'data' dulu, lalu di folder utama proyek."""
    for p in (os.path.join("data", DATASET_NAME), DATASET_NAME):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"File '{DATASET_NAME}' tidak ditemukan (dicari di folder 'data' dan folder utama)."
    )


def _read_raw(path=None):
    path = path or find_dataset_path()
    df = pd.read_excel(path)
    if "Tanggal" in df.columns:
        df["Tanggal"] = pd.to_datetime(df["Tanggal"])
        df = df.set_index("Tanggal")
    elif df.index.name != "Tanggal":
        df.index = pd.to_datetime(df.index)
        df.index.name = "Tanggal"
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_index()


def load_and_preprocess_data(path=None):
    """Baca Excel -> tabel harga rapi (urut tanggal, sel kosong diisi ffill/bfill/interpolasi)."""
    df = _read_raw(path)
    return df.ffill().bfill().interpolate(method="linear")


def check_data_quality(path=None, max_stale_days=4):
    """
    Cek cepat setelah data diupdate manual. Mengembalikan daftar pesan peringatan
    (list kosong = aman). Dipakai oleh app.py dan predict.py.
    """
    msgs = []
    raw = _read_raw(path)
    if raw.index.duplicated().any():
        dup = raw.index[raw.index.duplicated()].strftime("%Y-%m-%d").tolist()
        msgs.append(f"Ada tanggal ganda di dataset: {', '.join(dup[:5])}. Hapus salah satunya.")
    last_date = raw.index.max()
    stale = (pd.Timestamp.today().normalize() - last_date).days
    if stale > max_stale_days:
        msgs.append(
            f"Data terakhir {last_date:%d %b %Y} ({stale} hari lalu). Tambahkan harga terbaru ke Excel "
            f"supaya prediksi dimulai dari tanggal yang benar."
        )
    last_rows = raw.tail(1)
    empty = last_rows.columns[last_rows.isna().any()].tolist()
    if empty:
        msgs.append(
            f"Baris terakhir ({last_date:%d %b %Y}) masih kosong untuk: {', '.join(empty[:4])}"
            f"{'...' if len(empty) > 4 else ''}. Sel kosong akan diisi otomatis dengan harga sebelumnya."
        )
    new_cols = [c for c in raw.columns if c not in ATTR_TO_GROUP]
    if new_cols:
        msgs.append(f"Kolom tidak dikenali (tidak ada model): {', '.join(new_cols)}")
    return msgs


# ----------------------------------------------------------------------------
# FITUR (dipakai training)
# ----------------------------------------------------------------------------
# Semua fitur dibuat RELATIF (persentase), bukan level harga mentah. Alasannya: harga pangan bergeser naik dari
# waktu ke waktu, sehingga model pohon yang belajar dari "level harga" melihat level baru di data uji dan menebak
# sembarangan. Target = persentase perubahan harga hari itu.
LONG_WINDOW = 22          # acuan jangka panjang (1 bulan kerja) untuk fitur "jarak ke rata-rata"
CHANGE_EPS = 1e-4         # perubahan < 0,01% dianggap "tidak berubah" (menyerap derau float prediksi rekursif)
MAX_RUN = 22              # batas atas fitur "berapa hari harga tidak berubah"


def _run_length_flat(prices):
    """Berapa hari berturut-turut (sampai hari terakhir `prices`) harga tidak berubah. Dipakai saat PREDIKSI."""
    n = 0
    for i in range(len(prices) - 1, 0, -1):
        if abs(prices[i] / prices[i - 1] - 1.0) < CHANGE_EPS:
            n += 1
            if n >= MAX_RUN:
                break
        else:
            break
    return n


def build_features(df_series, window_size):
    """
    Fitur relatif untuk memprediksi persentase perubahan harga hari t, hanya memakai harga sampai t-1:
      ret_1..ret_w     persentase perubahan harian pada w hari terakhir
      dev_mean_w       jarak harga kemarin ke rata-rata w hari terakhir
      dev_mean_long    jarak harga kemarin ke rata-rata 22 hari terakhir
      rel_std_w        simpangan baku w hari terakhir / rata-rata (volatilitas relatif)
      flat_days        berapa hari harga terakhir tidak berubah (harga pangan sering datar lalu melompat)
      dayofweek, month kalender
    Kolom 'target' = persentase perubahan harga hari t; 'prev_price' = harga t-1 (untuk kembali ke Rupiah).
    Keduanya BUKAN fitur.
    """
    w = window_size
    s = df_series.astype(float)
    ret = s.pct_change()
    df_feat = pd.DataFrame(index=s.index)
    df_feat["target"] = ret
    df_feat["prev_price"] = s.shift(1)

    for l in range(1, w + 1):
        df_feat[f"ret_{l}"] = ret.shift(l)

    shifted = s.shift(1)   # harga hari target tidak ikut dihitung
    df_feat[f"dev_mean_{w}"] = shifted / shifted.rolling(w).mean() - 1.0
    df_feat["dev_mean_long"] = shifted / shifted.rolling(LONG_WINDOW).mean() - 1.0
    # std dengan np.std(ddof=1) (bukan rolling().std() bawaan pandas yang berderau pada jendela konstan)
    roll_std = shifted.rolling(w).apply(lambda x: np.std(x, ddof=1), raw=True)
    df_feat[f"rel_std_{w}"] = roll_std / shifted.rolling(w).mean()

    changed = (ret.abs() >= CHANGE_EPS).fillna(True)
    flat = changed.groupby(changed.cumsum()).cumcount().clip(upper=MAX_RUN)   # hari datar sampai & termasuk t
    df_feat["flat_days"] = flat.shift(1)

    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month
    return df_feat.dropna()


# ----------------------------------------------------------------------------
# MODEL & PREDIKSI (dipakai app.py dan predict.py)
# ----------------------------------------------------------------------------
def _safe(attr):
    return attr.replace(" ", "_")


def save_model_files(model, info, attr):
    """
    Simpan model dalam format BAWAAN library (aman lintas versi & lintas komputer), bukan pickle/joblib:
      - XGBoost  -> models/<atribut>.json
      - LightGBM -> models/<atribut>.txt
      - info (nama model, window, metrik) -> models/<atribut>_info.json
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    base = os.path.join(MODELS_DIR, _safe(attr))
    if info["model_name"] == "XGBoost":
        model.save_model(base + ".json")
    else:
        model.booster_.save_model(base + ".txt")
    with open(base + "_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=float)


def load_model(attr):
    """
    Muat model hasil train.py untuk satu atribut. Mengembalikan (model, info).
    Model TIDAK dilatih ulang: hanya data terbarunya yang dipakai sebagai masukan prediksi.
    (Format lama .joblib masih dibaca bila ada, tetapi bisa gagal jika versi library berbeda.)
    """
    base = os.path.join(MODELS_DIR, _safe(attr))
    try:
        if os.path.exists(base + "_info.json"):
            with open(base + "_info.json", encoding="utf-8") as f:
                info = json.load(f)
            if info["model_name"] == "XGBoost":
                from xgboost import XGBRegressor
                model = XGBRegressor()
                model.load_model(base + ".json")
            else:
                import lightgbm as lgb
                model = lgb.Booster(model_file=base + ".txt")
            return model, info
        if os.path.exists(base + "_info.joblib") and os.path.exists(base + ".joblib"):   # format lama
            info = joblib.load(base + "_info.joblib")
            return joblib.load(base + ".joblib"), info
    except Exception as e:  # DLL LightGBM diblokir Windows, versi library beda, atau file rusak
        raise RuntimeError(
            f"Model untuk '{attr}' gagal dimuat: {e}\n"
            f"Jika model LightGBM diblokir Windows, latih ulang dengan: python train.py --models xgboost. "
            f"Jika file rusak, ekstrak ulang folder models/."
        ) from e
    raise FileNotFoundError(f"Model '{attr}' belum ada di folder '{MODELS_DIR}'. Jalankan 'python train.py' dulu.")


def make_feature_row(history, date, window_size):
    """
    Susun satu baris fitur untuk memprediksi harga pada `date`, dari riwayat harga `history`
    (urut lama -> baru, hari terakhir = hari sebelum `date`). Isinya HARUS sama dengan build_features().
    """
    w = window_size
    h = np.asarray(history, dtype=float)
    feat = {f"ret_{l}": h[-l] / h[-l - 1] - 1.0 for l in range(1, w + 1)}
    win = h[-w:]
    feat[f"dev_mean_{w}"] = h[-1] / win.mean() - 1.0
    feat["dev_mean_long"] = h[-1] / h[-LONG_WINDOW:].mean() - 1.0
    feat[f"rel_std_{w}"] = np.std(win, ddof=1) / win.mean()
    feat["flat_days"] = _run_length_flat(h)
    feat["dayofweek"] = date.dayofweek
    feat["month"] = date.month
    return feat


def predict_future_recursive(model, df_series, steps_ahead, window_size, future_dates=None):
    """
    Prediksi multi-step rekursif: setiap prediksi dipakai sebagai riwayat untuk langkah berikutnya.
    Default tanggalnya = hari kerja setelah data terakhir; `future_dates` dipakai saat evaluasi
    agar tanggal (dan fitur kalender) sama persis dengan data aktual.
    """
    history = list(df_series.values)
    if future_dates is None:
        future_dates = pd.date_range(
            start=df_series.index[-1] + pd.Timedelta(days=1), periods=steps_ahead * 2, freq="B"
        )[:steps_ahead]
    preds = []
    for f_date in future_dates[:steps_ahead]:
        feat = make_feature_row(history, f_date, window_size)
        predicted_return = float(model.predict(pd.DataFrame([feat]))[0])   # persentase perubahan (pecahan)
        p = float(history[-1] * (1.0 + predicted_return))
        preds.append(p)
        history.append(p)
    return pd.Series(preds, index=list(future_dates[:steps_ahead]))


def historical_range(series, steps, lo=0.10, hi=0.90, min_samples=60):
    """
    Rentang harga yang WAJAR (bukan ramalan titik) untuk 1..steps hari kerja ke depan, dari sebaran historis
    perubahan harga k hari kerja pada atribut itu: harga terakhir * (1 + kuantil 10% / 90% perubahan k hari).
    Hasilnya DataFrame (index 1..steps) berkolom 'lo' dan 'hi'; NaN bila riwayat terlalu pendek.
    Pada uji di data latih -> data uji, rentang 10-90% ini memuat harga aktual pada kira-kira 90% kasus (konservatif).
    """
    s = pd.Series(series, dtype=float).dropna()
    last = float(s.iloc[-1])
    rows = []
    for k in range(1, steps + 1):
        ch = (s.shift(-k) / s - 1.0).dropna()
        if len(ch) < min_samples:
            rows.append((np.nan, np.nan))
        else:
            q_lo, q_hi = np.quantile(ch.values, [lo, hi])
            rows.append((last * (1.0 + q_lo), last * (1.0 + q_hi)))
    return pd.DataFrame(rows, index=range(1, steps + 1), columns=["lo", "hi"])


def steps_to_date(last_date, target_date):
    """Jumlah hari kerja dari hari setelah `last_date` sampai `target_date` (akhir pekan tidak dihitung)."""
    start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).date()
    end = (pd.Timestamp(target_date) + pd.Timedelta(days=1)).date()
    return int(np.busday_count(start, end))


def check_feature_parity(series, window_size, n_checks=5, rtol=1e-9):
    """
    Uji kebenaran: fitur yang dibuat saat PREDIKSI (make_feature_row) harus identik dengan fitur
    saat TRAINING (build_features) pada tanggal yang sama. Bila beda, prediksi akan bergeser diam-diam.
    """
    df_feat = build_features(series, window_size)
    for date in df_feat.index[-n_checks:]:
        pos = series.index.get_loc(date)
        row = make_feature_row(series.iloc[:pos].values, date, window_size)
        expected = df_feat.loc[date].drop(["target", "prev_price"])
        got = np.array([row[k] for k in expected.index], dtype=float)
        if not np.allclose(got, expected.values.astype(float), rtol=rtol, atol=1e-9):
            raise ValueError(f"Fitur prediksi != fitur training pada {date:%Y-%m-%d} (window {window_size}).")
    return True