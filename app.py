import calendar
import json
import os
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from common import (
    ATTR_TO_GROUP,
    check_data_quality,
    find_dataset_path,
    load_and_preprocess_data,
    load_model,
    predict_future_recursive,
)

# -----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Harga Pangan DKI Jakarta",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MAX_STEPS = 750
NA_LABEL = "Tidak terdefinisi"
HISTORY_WINDOW = 180


# -----------------------------------------------------------------------------
# CSS - menyesuaikan tiga rancangan UI yang diberikan
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --navy: #14213d;
        --blue: #dce8f1;
        --blue-2: #edf4f9;
        --blue-3: #cbdce9;
        --muted: #64748b;
        --line: #d8e2eb;
        --white: #ffffff;
    }

    .stApp {
        background: #f8fafc;
        color: var(--navy);
    }
    .block-container {
        max-width: 1450px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    #MainMenu, footer, header {visibility: hidden;}

    h1, h2, h3, h4, p, label, div {
        font-family: "Arial", sans-serif;
    }
    h1, h2, h3 { color: var(--navy); }

    .brand-title {
        color: #111c34;
        font-size: 1.08rem;
        line-height: 1.15;
        font-weight: 800;
        padding-top: 0.2rem;
    }
    .brand-subtitle {
        color: #26364d;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .page-title {
        font-size: 1.85rem;
        line-height: 1.1;
        font-weight: 800;
        color: #101a31;
        margin: 0.2rem 0 0.15rem 0;
    }
    .page-subtitle {
        color: #43536b;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .top-rule {
        height: 3px;
        background: #1e293b;
        margin: 0.5rem 0 1rem 0;
        border-radius: 3px;
    }

    /* Navigasi horizontal */
    div[data-testid="stRadio"] > label {display: none;}
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 0.45rem;
        flex-wrap: wrap;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 24px;
        padding: 0.42rem 0.85rem;
        color: #26364d;
        font-weight: 650;
        margin: 0;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: #dce8f1;
        border-color: #d0deea;
        color: #14213d;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
        display: none;
    }

    /* Kartu dan panel */
    .kpi-card {
        background: linear-gradient(135deg, #e8f0f6 0%, #d5e2ec 100%);
        border: 1px solid #d9e4ed;
        border-radius: 18px;
        min-height: 112px;
        padding: 17px 18px;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 2px 7px rgba(31, 55, 80, 0.05);
    }
    .kpi-icon {
        width: 58px;
        height: 58px;
        min-width: 58px;
        border-radius: 50%;
        background: rgba(255,255,255,0.78);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.75rem;
    }
    .kpi-label {
        color: #3b4b62;
        font-size: 0.82rem;
        font-weight: 650;
        line-height: 1.2;
        margin-bottom: 5px;
    }
    .kpi-value {
        color: #101a31;
        font-size: 1.42rem;
        line-height: 1.15;
        font-weight: 850;
    }
    .kpi-sub {
        color: #52647a;
        font-size: 0.73rem;
        margin-top: 5px;
    }
    .panel {
        background: #ffffff;
        border: 1px solid #dce5ed;
        border-radius: 17px;
        padding: 14px 15px 12px 15px;
        box-shadow: 0 2px 8px rgba(31, 55, 80, 0.035);
        min-height: 100%;
    }
    .panel-heading {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #101a31;
        font-size: 1.08rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 2px;
    }
    .panel-icon {
        width: 36px;
        height: 36px;
        min-width: 36px;
        border-radius: 50%;
        background: #e7eff5;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .panel-caption {
        color: #68788d;
        font-size: 0.76rem;
        margin: 0 0 10px 46px;
    }
    .filter-panel {
        background: linear-gradient(120deg, #e8f0f6 0%, #d5e2ec 100%);
        border: 1px solid #d4e0e9;
        border-radius: 19px;
        padding: 15px 18px 10px 18px;
        margin: 0.6rem 0 1rem 0;
    }
    .filter-title {
        color: #14213d;
        font-size: 1.12rem;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .filter-subtitle {
        color: #63748a;
        font-size: 0.78rem;
        margin-bottom: 10px;
    }
    .bottom-band {
        background: linear-gradient(110deg, #e8f0f6 0%, #d5e2ec 100%);
        border: 1px solid #d6e2eb;
        border-radius: 19px;
        padding: 16px 18px;
        margin-top: 0.9rem;
    }
    .bottom-model {
        color: #14213d;
        font-size: 1.55rem;
        font-weight: 850;
        margin-top: 3px;
    }
    .bottom-caption {
        color: #607189;
        font-size: 0.78rem;
        margin-top: 3px;
    }
    .mini-metric {
        background: rgba(255,255,255,0.62);
        border: 1px solid #d7e2eb;
        border-radius: 16px;
        padding: 12px 10px;
        text-align: center;
        min-height: 76px;
    }
    .mini-label {
        color: #53647a;
        font-size: 0.78rem;
        margin-bottom: 4px;
    }
    .mini-value {
        color: #14213d;
        font-size: 1.15rem;
        font-weight: 850;
    }
    .status-up {color: #dc2626; font-weight: 750;}
    .status-down {color: #059669; font-weight: 750;}
    .status-flat {color: #64748b; font-weight: 750;}

    /* Kontrol Streamlit */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stDateInput"] label {
        color: #26364d;
        font-weight: 700;
    }
    div[data-testid="stSelectbox"] > div > div {
        border-radius: 12px;
        border-color: #d4e0e9;
        background: white;
    }
    div[data-testid="stButton"] button {
        border-radius: 12px;
        border: 1px solid #cbd9e5;
        background: #e5eef5;
        color: #14213d;
        font-weight: 700;
    }
    div[data-testid="stButton"] button:hover {
        border-color: #9fb6ca;
        color: #14213d;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #dbe4ec;
        border-radius: 10px;
        overflow: hidden;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid #dce5ed;
        border-radius: 17px;
        padding: 0.85rem 0.9rem;
        box-shadow: 0 2px 8px rgba(31, 55, 80, 0.035);
    }
    @media (max-width: 900px) {
        .kpi-card { min-height: 96px; padding: 12px; }
        .kpi-value { font-size: 1.12rem; }
        .bottom-band > div { grid-template-columns: 1fr 1fr !important; }
    }
    div[data-testid="stAlert"] {border-radius: 12px;}
    .small-note {
        color: #6b7b8f;
        font-size: 0.75rem;
        margin-top: 5px;
    }
    /* Popup kalender berada di portal Streamlit, bukan di tombol stPopover. */
    div[data-testid="stPopoverBody"]:has(.compact-calendar-marker),
    div[data-baseweb="popover"]:has(.compact-calendar-marker) > div {
        width: 310px !important;
        min-width: 310px !important;
        max-width: 310px !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stPopoverBody"]:has(.compact-calendar-marker) {
        padding: 10px !important;
        overflow-x: hidden !important;
    }
    /* Paksa kolom Streamlit muat dalam kalender 310px. */
    div[data-testid="stPopoverBody"]:has(.compact-calendar-marker) [data-testid="stHorizontalBlock"] {
        gap: 2px !important;
        flex-wrap: nowrap !important;
        min-width: 0 !important;
    }
    div[data-testid="stPopoverBody"]:has(.compact-calendar-marker) [data-testid="column"] {
        min-width: 0 !important;
        width: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stPopoverBody"]:has(.compact-calendar-marker) [data-testid="stVerticalBlock"] {
        gap: 3px !important;
    }
    div[data-testid="stPopoverBody"]:has(.compact-calendar-marker) [data-testid="stButton"] button {
        width: 100% !important;
        min-width: 0 !important;
        min-height: 30px !important;
        height: 30px !important;
        padding: 0 !important;
        border-radius: 6px !important;
        font-size: 12px !important;
    }
    div[data-testid="stPopoverBody"]:has(.compact-calendar-marker) [data-testid="stCaptionContainer"] p {
        font-size: 10px !important;
        margin: 0 !important;
    }
    .compact-calendar-marker {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# LOAD DATA DAN HASIL EVALUASI
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file_mtime):
    return load_and_preprocess_data()


@st.cache_data
def load_quality_messages(file_mtime):
    return check_data_quality()


def get_file_mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0


@st.cache_data
def load_evaluation_results(file_mtime):
    if os.path.exists("eval_results.csv"):
        return pd.read_csv("eval_results.csv")
    return None


@st.cache_data
def load_test_predictions(file_mtime):
    if os.path.exists("test_predictions.csv"):
        return pd.read_csv("test_predictions.csv", parse_dates=["Tanggal"])
    return None


@st.cache_data
def load_horizon_results(file_mtime):
    if os.path.exists("eval_horizon.csv"):
        return pd.read_csv("eval_horizon.csv")
    return None


@st.cache_data
def load_validation_report(file_mtime):
    if os.path.exists("validation_report.json"):
        with open("validation_report.json", encoding="utf-8") as f:
            return json.load(f)
    return None


DATA_PATH = find_dataset_path()
DATA_MTIME = os.path.getmtime(DATA_PATH)
df_data = load_data(DATA_MTIME).sort_index()
df_eval_results = load_evaluation_results(get_file_mtime("eval_results.csv"))
df_test_pred = load_test_predictions(get_file_mtime("test_predictions.csv"))
df_horizon = load_horizon_results(get_file_mtime("eval_horizon.csv"))
validation_report = load_validation_report(get_file_mtime("validation_report.json"))

if df_eval_results is None:
    st.error("File 'eval_results.csv' belum ditemukan. Jalankan 'python train.py' terlebih dahulu.")
    st.stop()

required_eval_cols = {"Kelompok", "Atribut", "Model", "MAE", "RMSE", "R2"}
missing_cols = required_eval_cols - set(df_eval_results.columns)
if missing_cols:
    st.error(f"Kolom evaluasi belum lengkap: {', '.join(sorted(missing_cols))}. Jalankan ulang training.")
    st.stop()

if "MAE Naive" not in df_eval_results.columns:
    df_eval_results["MAE Naive"] = np.nan

# Gunakan kelompok yang benar-benar memiliki atribut dalam dataset.
kelompok_options = list(dict.fromkeys(
    ATTR_TO_GROUP.get(attr, "Lainnya")
    for attr in df_data.columns
))
atribut_options = [attr for attr in ATTR_TO_GROUP if attr in df_data.columns]


# -----------------------------------------------------------------------------
# FUNGSI BANTU
# -----------------------------------------------------------------------------
def fmt_r2(value):
    if pd.isna(value):
        return NA_LABEL
    return f"{float(value):.3f}"


def fmt_rp(value, decimals=0):
    if pd.isna(value):
        return "—"
    return f"Rp {float(value):,.{decimals}f}"


def style_eval(frame):
    formats = {}
    for col in ["MAE", "RMSE", "MAE Naive", "RMSE Naive"]:
        if col in frame.columns:
            formats[col] = "Rp {:,.2f}"
    if "R2" in frame.columns:
        formats["R2"] = "{:.4f}"
    return frame.style.format(formats, na_rep=NA_LABEL)


def panel_heading(icon, title, subtitle=None):
    st.markdown(
        f"""
        <div class="panel-heading"><span class="panel-icon">{icon}</span>{title}</div>
        {f'<div class="panel-caption">{subtitle}</div>' if subtitle else ''}
        """,
        unsafe_allow_html=True,
    )


def kpi_card(icon, label, value, subtitle=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_metric(label, value):
    st.markdown(
        f"""
        <div class="mini-metric">
            <div class="mini-label">{label}</div>
            <div class="mini-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_chosen_results(results):
    if "Dipilih" in results.columns:
        chosen = results[results["Dipilih"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    else:
        sort_col = "CV MAE" if "CV MAE" in results.columns else "MAE"
        chosen = results.sort_values(sort_col).groupby("Atribut", as_index=False).head(1).copy()
    if chosen.empty:
        sort_col = "CV MAE" if "CV MAE" in results.columns else "MAE"
        chosen = results.sort_values(sort_col).groupby("Atribut", as_index=False).head(1).copy()
    return chosen


def get_best_model_summary(results):
    chosen = get_chosen_results(results)
    if "CV MAE" in chosen.columns:
        summary = (
            chosen.groupby("Model")
            .agg(
                Jumlah=("Atribut", "nunique"),
                CV_MAE=("CV MAE", "mean"),
                MAE=("MAE", "mean"),
                RMSE=("RMSE", "mean"),
                R2=("R2", "mean"),
            )
            .reset_index()
            .sort_values(["Jumlah", "CV_MAE"], ascending=[False, True])
        )
    else:
        summary = (
            chosen.groupby("Model")
            .agg(
                Jumlah=("Atribut", "nunique"),
                MAE=("MAE", "mean"),
                RMSE=("RMSE", "mean"),
                R2=("R2", "mean"),
            )
            .reset_index()
            .sort_values("Jumlah", ascending=False)
        )
    return chosen, summary.iloc[0]


def status_from_change(pct):
    if pd.isna(pct) or abs(pct) < 1e-10:
        return "Tidak Naik", "Harga relatif stabil", "status-flat", "—"
    if pct > 0:
        return "Naik", "Harga cenderung naik", "status-up", f"↑ +{pct:.1f}%"
    return "Turun", "Harga cenderung turun", "status-down", f"↓ {pct:.1f}%"


def tight_ylim(ax, *series_list):
    values = np.concatenate([
        np.asarray(series, dtype=float)
        for series in series_list
        if series is not None and len(series)
    ])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    low, high = float(values.min()), float(values.max())
    pad = max((high - low) * 0.15, abs(float(values.mean())) * 0.005, 1)
    ax.set_ylim(low - pad, high + pad)


def format_rp_axis(ax):
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:,.0f}")
    )


# -----------------------------------------------------------------------------
# KALENDER POPUP KUSTOM: navigasi tetap DI DALAM popover
# -----------------------------------------------------------------------------
def calendar_shift_month(view_key, amount, minimum, maximum):
    """Pindah tampilan bulan/tahun tanpa mengubah tanggal prediksi terpilih."""
    current = pd.Timestamp(st.session_state[view_key])
    shifted = (current + pd.DateOffset(months=amount)).replace(day=1)
    low = pd.Timestamp(minimum).replace(day=1)
    high = pd.Timestamp(maximum).replace(day=1)
    st.session_state[view_key] = max(low, min(shifted, high)).date()


def calendar_choose_day(selected_key, view_key, chosen):
    """Mengubah tanggal target hanya ketika pengguna menekan angka tanggal."""
    st.session_state[selected_key] = chosen
    st.session_state[view_key] = chosen.replace(day=1)


def get_model_and_info(attribute):
    try:
        model, info = load_model(attribute)
        return model, info, None
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        return None, None, str(error)

@st.cache_data(show_spinner=False)
def get_all_next_day_predictions(data_mtime, model_versions):
    """Prediksi 1 hari kerja berikutnya untuk semua jenis komoditas."""

    target_date = pd.bdate_range(
        start=df_data.index.max() + pd.Timedelta(days=1),
        periods=1
    )[0]

    rows = []
    errors = []

    for attr in atribut_options:
        series = df_data[attr].dropna().sort_index()

        if series.empty:
            errors.append(f"{attr}: tidak ada data harga")
            continue

        model, info, error = get_model_and_info(attr)

        if model is None:
            errors.append(f"{attr}: {error}")
            continue

        try:
            hasil = predict_future_recursive(
                model,
                series,
                steps_ahead=1,
                window_size=int(info["window"]),
                future_dates=pd.DatetimeIndex([target_date])
            )

            harga_terakhir = float(series.iloc[-1])
            harga_prediksi = float(hasil.iloc[-1])

            if not np.isfinite(harga_prediksi):
                raise ValueError("Prediksi tidak valid")

            selisih = harga_prediksi - harga_terakhir

            perubahan = (
                selisih / harga_terakhir * 100
                if harga_terakhir != 0
                else np.nan
            )

            if selisih > 0.005:
                status = "Naik"
            elif selisih < -0.005:
                status = "Turun"
            else:
                status = "Tetap"

            rows.append({
                "Kelompok": ATTR_TO_GROUP.get(attr, "Lainnya"),
                "Jenis Komoditas": attr,
                "Harga Terakhir": harga_terakhir,
                "Prediksi Besok": harga_prediksi,
                "Perubahan (%)": perubahan,
                "Status": status,
                "Model": info["model_name"]
            })

        except (ValueError, TypeError, KeyError,
                IndexError, RuntimeError) as error:
            errors.append(f"{attr}: {error}")

    return pd.DataFrame(rows), errors, target_date


def get_model_cache_signature():
    """Memperbarui cache ketika file model berubah."""
    import glob

    paths = []

    for ext in ("*_info.json", "*.json", "*.txt", "*.joblib"):
        paths.extend(glob.glob(os.path.join("models", ext)))

    return tuple(
        sorted(
            (path, os.path.getmtime(path))
            for path in set(paths)
        )
    )



# -----------------------------------------------------------------------------
# NAVIGASI HORIZONTAL
# -----------------------------------------------------------------------------
nav_brand, nav_menu, nav_refresh = st.columns([2.05, 4.55, 0.35])
with nav_brand:
    st.markdown(
        """
        <div class="brand-title">Prediksi Harga Pangan</div>
        <div class="brand-subtitle">DKI Jakarta</div>
        """,
        unsafe_allow_html=True,
    )
with nav_menu:
    page = st.radio(
        "Navigasi",
        ["Dashboard", "Data dan Visualisasi Harga", "Evaluasi Model"],
        horizontal=True,
        label_visibility="collapsed",
        key="page_navigation",
    )
with nav_refresh:
    if st.button("↻", help="Muat ulang data dan hasil evaluasi", key="refresh_app"):
        st.cache_data.clear()
        st.rerun()
st.markdown('<div class="top-rule"></div>', unsafe_allow_html=True)



# =============================================================================
# HALAMAN 1 - DASHBOARD
# =============================================================================
if page == "Dashboard":
    st.markdown('<div class="page-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Ringkasan informasi prediksi harga bahan pangan di DKI Jakarta</div>',
        unsafe_allow_html=True,
    )

    date_range = (
        f"Periode {df_data.index.min():%d %b %Y} – {df_data.index.max():%d %b %Y}"
        if not df_data.empty else "Periode data tidak tersedia"
    )
    chosen, best_model_row = get_best_model_summary(df_eval_results)
    n_groups = len(set(ATTR_TO_GROUP.get(attr, "Lainnya") for attr in df_data.columns))
    n_attributes = len(df_data.columns)

    k1, k2, k3 = st.columns(3, gap="medium")
    with k1:
        kpi_card("🌶️", "Jumlah Kelompok Bahan Pangan", f"{n_groups} Kelompok", "Komoditas yang dianalisis")
    with k2:
        kpi_card("📋", "Jumlah Data Harga", f"{n_attributes} Jenis Bahan", date_range)
    with k3:
        kpi_card(
            "📊",
            "Model Prediksi Terbaik",
            str(best_model_row["Model"]),
            "Paling sering terpilih berdasarkan validasi data latih",
        )

    st.write("")
    left, right = st.columns(2, gap="medium")

    # Mengambil seluruh atribut harga yang tersedia
    sample_attrs = list(df_data.columns)

    latest_rows = []
    status_rows = []
    for attr in sample_attrs:
        series = df_data[attr].dropna()
        if series.empty:
            continue
        latest_price = float(series.iloc[-1])
        previous_price = float(series.iloc[-2]) if len(series) >= 2 else latest_price
        change = latest_price - previous_price
        pct = (change / previous_price * 100) if previous_price else 0.0
        label, desc, css_class, change_label = status_from_change(pct)

        latest_rows.append({
            "Bahan Pangan": attr,
            "Harga Terbaru": fmt_rp(latest_price),
            "Perubahan Harga": change_label,
        })
        status_rows.append({
            "Bahan Pangan": attr,
            "Harga Terbaru": fmt_rp(latest_price),
            "Status Saat Ini": (
                f"↑ {label}" if label == "Naik"
                else (f"↓ {label}" if label == "Turun" else f"• {label}")
            ),
        })

    with left:
        with st.container(border=True):
            panel_heading("▥", "Harga Terbaru dan Perubahan", "Lihat harga terbaru dan perubahan dibanding periode sebelumnya")
            if latest_rows:
                st.dataframe(
                    pd.DataFrame(latest_rows),
                    hide_index=True,
                    use_container_width=True,
                    height=740,
                )
            else:
                st.info("Data harga terbaru belum tersedia.")

    with right:
        with st.container(border=True):
            panel_heading("◆", "Status Harga Saat Ini", "Informasi sederhana apakah harga sedang naik atau turun")
            if status_rows:
                st.dataframe(
                    pd.DataFrame(status_rows),
                    hide_index=True,
                    use_container_width=True,
                    height=740,
                )
            else:
                st.info("Data status harga belum tersedia.")

    st.write("")
    st.markdown(
        f"""
        <div class="bottom-band">
            <div style="display:grid;grid-template-columns:2.2fr 1fr 1fr 1fr;gap:12px;align-items:center;">
                <div>
                    <div class="panel-heading"><span class="panel-icon">⚙️</span>Model Prediksi Terbaik</div>
                    <div class="bottom-model">{best_model_row['Model']}</div>
                    <div class="bottom-caption">
                        Model paling sering terpilih melalui validasi pada data latih.
                        Ringkasan metrik dihitung dari atribut yang memilih model tersebut.
                    </div>
                </div>
                <div class="mini-metric"><div class="mini-label">MAE</div><div class="mini-value">{fmt_rp(best_model_row['MAE'])}</div></div>
                <div class="mini-metric"><div class="mini-label">RMSE</div><div class="mini-value">{fmt_rp(best_model_row['RMSE'])}</div></div>
                <div class="mini-metric"><div class="mini-label">R² Score</div><div class="mini-value">{fmt_r2(best_model_row['R2'])}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_r2_undefined = int(
        chosen.groupby("Atribut")["R2"].apply(lambda values: values.isna().all()).sum()
    )
    if n_r2_undefined:
        st.caption(
            f"Catatan: R² tidak terdefinisi pada {n_r2_undefined} atribut karena harga aktual periode uji konstan. "
            "Nilai tersebut tetap ditampilkan sebagai 'Tidak terdefinisi', bukan dipaksakan menjadi 0 atau 1."
        )


# =============================================================================
# HALAMAN 2 - DATA DAN VISUALISASI HARGA
# =============================================================================
elif page == "Data dan Visualisasi Harga":
    st.markdown('<div class="page-title">Data dan Visualisasi Harga</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            """
            <div class="filter-title">⌕ &nbsp;Pilih Data yang Ingin Dilihat</div>
            <div class="filter-subtitle">Pilih komoditas, jenis barang, dan tanggal prediksi</div>
            """,
            unsafe_allow_html=True,
        )
        f1, f2, f3 = st.columns([1, 1, 1], gap="medium")

        with f1:
            selected_group = st.selectbox(
                "🌱 Kelompok Komoditas",
                kelompok_options,
                key="data_group",
            )

        attrs_in_group = [
            attr for attr, group in ATTR_TO_GROUP.items()
            if group == selected_group and attr in df_data.columns
        ]
        if not attrs_in_group:
            st.warning("Tidak ada atribut harga untuk kelompok ini.")
            st.stop()

        with f2:
            selected_attr = st.selectbox(
                "🌶️ Jenis Barang",
                attrs_in_group,
                key="data_attribute",
            )

        # Ambil riwayat harga untuk atribut yang dipilih.
        series = df_data[selected_attr].dropna().sort_index()
        if series.empty:
            st.warning("Data historis untuk atribut ini belum tersedia.")
            st.stop()

        last_date = pd.Timestamp(series.index[-1]).normalize()

        # Hari kerja pertama setelah tanggal data terakhir.
        first_forecast_date = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=1,
        )[0]

        # Batas maksimal pemilihan tanggal mengikuti MAX_STEPS hari kerja.
        max_forecast_date = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=MAX_STEPS,
        )[-1]

        min_date = first_forecast_date.date()
        max_date = max_forecast_date.date()
        date_key = f"forecast_target_date_{selected_attr}_{last_date:%Y%m%d}"
        view_key = f"calendar_visible_month_{selected_attr}_{last_date:%Y%m%d}"

        if date_key not in st.session_state:
            st.session_state[date_key] = min_date
        if view_key not in st.session_state:
            st.session_state[view_key] = st.session_state[date_key].replace(day=1)

        # Perbaiki state jika rentang tanggal berubah karena dataset diperbarui.
        if not min_date <= st.session_state[date_key] <= max_date:
            st.session_state[date_key] = min_date
            st.session_state[view_key] = min_date.replace(day=1)

        selected_date = st.session_state[date_key]
        visible_month = pd.Timestamp(st.session_state[view_key])
        min_month = pd.Timestamp(min_date).replace(day=1)
        max_month = pd.Timestamp(max_date).replace(day=1)

        with f3:
            st.markdown(
                """
                <div style="
                    font-weight: 600;
                    font-size: 14px;
                    margin-bottom: 5px;
                ">
                    🗓️ Pilih Tanggal Prediksi
                </div>
                """,
                unsafe_allow_html=True
            )
            # st.popover menggantikan st.date_input: tidak akan muncul popup
            # tanggal bawaan yang tak bisa ditambahi tombol khusus.
            with st.popover(
                f"📅 {selected_date:%d/%m/%Y}",
                use_container_width=True,
            ):
                st.markdown('<span class="compact-calendar-marker"></span>', unsafe_allow_html=True)
                # Navigasi DI DALAM popup, di atas nama hari dan angka tanggal.
                prev_year, prev_month, month_title, next_month, next_year = st.columns(
                    [1, 1, 3, 1, 1], gap="small"
                )
                with prev_year:
                    st.button("«", key=f"py_{date_key}", help="Tahun sebelumnya",
                              disabled=visible_month - pd.DateOffset(years=1) < min_month,
                              on_click=calendar_shift_month,
                              args=(view_key, -12, min_date, max_date),
                              use_container_width=True)
                with prev_month:
                    st.button("‹", key=f"pm_{date_key}", help="Bulan sebelumnya",
                              disabled=visible_month <= min_month,
                              on_click=calendar_shift_month,
                              args=(view_key, -1, min_date, max_date),
                              use_container_width=True)
                with month_title:
                    st.markdown(
                        f"<div style='text-align:center; font-weight:700; "
                        f"padding-top:8px; white-space:nowrap;'>"
                        f"{calendar.month_name[visible_month.month]} "
                        f"{visible_month.year}</div>",
                        unsafe_allow_html=True,
                    )
                with next_month:
                    st.button("›", key=f"nm_{date_key}", help="Bulan berikutnya",
                              disabled=visible_month >= max_month,
                              on_click=calendar_shift_month,
                              args=(view_key, 1, min_date, max_date),
                              use_container_width=True)
                with next_year:
                    st.button("»", key=f"ny_{date_key}", help="Tahun berikutnya",
                              disabled=visible_month + pd.DateOffset(years=1) > max_month,
                              on_click=calendar_shift_month,
                              args=(view_key, 12, min_date, max_date),
                              use_container_width=True)

                # Minggu dimulai hari Senin; tampilkan sel tanggal per minggu.
                weekday_cols = st.columns(7, gap="small")
                for i, label in enumerate(["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]):
                    weekday_cols[i].caption(label)

                weeks = calendar.monthcalendar(visible_month.year, visible_month.month)
                for week_idx, week in enumerate(weeks):
                    days = st.columns(7, gap="small")
                    for weekday_idx, day_number in enumerate(week):
                        with days[weekday_idx]:
                            if day_number == 0:
                                st.write(" ")
                                continue
                            candidate = date(visible_month.year, visible_month.month, day_number)
                            st.button(
                                str(day_number),
                                key=f"calendar_{date_key}_{visible_month:%Y%m}_{day_number}",
                                disabled=not (min_date <= candidate <= max_date),
                                type="secondary",
                                use_container_width=True,
                                on_click=calendar_choose_day,
                                args=(date_key, view_key, candidate),
                            )
                st.caption(
                    f"Tanggal tersedia: {min_date:%d/%m/%Y} sampai "
                    f"{max_date:%d/%m/%Y}. Akhir pekan dialihkan "
                    "ke hari kerja berikutnya."
                )

    # Tanggal yang diminta pengguna.
    requested_date = pd.Timestamp(selected_date).normalize()

    # Model menggunakan kalender hari kerja Senin-Jumat.
    # Jika tanggal pilihan jatuh pada akhir pekan, gunakan hari kerja berikutnya.
    effective_target_date = pd.bdate_range(
        start=requested_date,
        periods=1,
    )[0]

    # Buat seluruh tanggal prediksi dari hari setelah data terakhir
    # sampai tanggal efektif yang dipilih pengguna.
    forecast_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        end=effective_target_date,
        freq="B",
    )
    steps_ahead = len(forecast_dates)

    if steps_ahead < 1:
        st.error("Pilih tanggal setelah tanggal data terakhir.")
        st.stop()

    if steps_ahead > MAX_STEPS:
        st.error(
            f"Tanggal prediksi melebihi batas {MAX_STEPS} hari kerja. "
            "Silakan pilih tanggal yang lebih dekat."
        )
        st.stop()

    requested_date_text = requested_date.strftime("%d/%m/%Y")
    effective_date_text = effective_target_date.strftime("%d/%m/%Y")
    if effective_target_date.normalize() != requested_date:
        horizon_label = effective_date_text
        date_note = (
            f"Tanggal pilihan {requested_date_text} jatuh pada akhir pekan. "
            f"Prediksi menggunakan hari kerja berikutnya, yaitu {effective_date_text}."
        )
    else:
        horizon_label = effective_date_text
        date_note = (
            f"Estimasi untuk {effective_date_text} "
            f"({steps_ahead} hari kerja dari data terakhir)."
        )

    model_obj, model_info, model_error = get_model_and_info(selected_attr)
    future_series = None
    prediction_error = None

    if model_obj is not None:
        try:
            future_series = predict_future_recursive(
                model_obj,
                series,
                steps_ahead,
                window_size=int(model_info["window"]),
                future_dates=forecast_dates,
            )
        except Exception as error:
            prediction_error = str(error)
    else:
        prediction_error = model_error

    st.write("")
    table_col, chart_col = st.columns([1, 1], gap="medium")

    recent = series.tail(5)
    recent_table = pd.DataFrame({
        "Tanggal": recent.index.strftime("%d/%m/%Y"),
        "Komoditas": selected_attr,
        "Harga (Rp)": recent.values,
    })

    with table_col:
        with st.container(border=True):
            panel_heading(
                "▤",
                "Data Harga Terbaru",
                f"Riwayat harga {selected_attr} beberapa hari terakhir",
            )
            st.dataframe(
                recent_table.style.format({"Harga (Rp)": "{:,.0f}"}),
                hide_index=True,
                use_container_width=True,
                height=220,
            )

    with chart_col:
        with st.container(border=True):
            panel_heading(
                "⌁",
                "Grafik Perubahan Harga",
                "Riwayat 180 hari kerja terakhir dan prediksi sampai tanggal pilihan",
            )
            fig, ax = plt.subplots(figsize=(8, 3.5))

            # Perpanjang rentang historis agar tren perubahan harga lebih mudah diamati.
            history_chart = series.tail(HISTORY_WINDOW)
            ax.plot(
                history_chart.index,
                history_chart.values,
                color="#2878c8",
                linewidth=2,
                label="Harga historis",
            )

            if future_series is not None and len(future_series):
                forecast_x = [history_chart.index[-1]] + list(future_series.index)
                forecast_y = [float(history_chart.iloc[-1])] + list(future_series.values)
                ax.plot(
                    forecast_x,
                    forecast_y,
                    color="#e39a42",
                    linestyle="--",
                    linewidth=2,
                    marker="o",
                    markersize=3,
                    label="Prediksi",
                )
                ax.axvline(
                    history_chart.index[-1],
                    color="#94a3b8",
                    linestyle=":",
                    linewidth=1,
                )

                # Pastikan skala sumbu Y mencakup data historis dan hasil prediksi.
                tight_ylim(ax, history_chart.values, future_series.values)

            ax.set_ylabel("Harga (Rp)")
            ax.set_xlabel("Tanggal")
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.legend(loc="best", fontsize=8)
            format_rp_axis(ax)
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.write("")
    if future_series is not None and len(future_series):
        last_price = float(series.iloc[-1])
        predicted_price = float(future_series.iloc[-1])
        price_change = predicted_price - last_price
        percent_change = (price_change / last_price * 100) if last_price else 0.0

        bottom1, bottom2, bottom3 = st.columns([1, 1, 1], gap="medium")
        with bottom1:
            kpi_card("🗓️", "Tanggal Prediksi", horizon_label, date_note)
        with bottom2:
            change_icon = "↑" if percent_change > 0 else ("↓" if percent_change < 0 else "•")
            kpi_card(
                "📊",
                "Perkiraan Perubahan",
                f"{change_icon} {percent_change:+.2f}%",
                f"{price_change:+,.0f} Rupiah dibanding harga terakhir",
            )
        with bottom3:
            kpi_card(
                "🪙",
                "Perkiraan Harga",
                fmt_rp(predicted_price),
                f"Estimasi pada {effective_date_text}",
            )
    else:
        st.warning(
            "Prediksi belum dapat ditampilkan. Pastikan model untuk atribut ini tersedia dan sudah dilatih. "
            f"{prediction_error or ''}"
        )
        b1, b2, b3 = st.columns(3)
        with b1:
            kpi_card("🗓️", "Tanggal Prediksi", horizon_label, date_note)
        with b2:
            kpi_card(
                "📊",
                "Perkiraan Perubahan",
                "Belum tersedia",
                "Model belum dapat menghitung prediksi",
            )
        with b3:
            kpi_card(
                "🪙",
                "Perkiraan Harga",
                "Belum tersedia",
                "Periksa model dan jalankan training jika diperlukan",
            )


# =============================================================================
# HALAMAN 3 - EVALUASI MODEL
# =============================================================================
elif page == "Evaluasi Model":
    st.markdown('<div class="page-title">Evaluasi Model</div>', unsafe_allow_html=True)

    chosen = get_chosen_results(df_eval_results)
    if "Kelompok" not in chosen.columns:
        chosen["Kelompok"] = chosen["Atribut"].map(ATTR_TO_GROUP).fillna("Lainnya")

    table_col, chart_col = st.columns([1.15, 1], gap="medium")

    with table_col:
        with st.container(border=True):
            panel_heading("▤", "Fitur Evaluasi Model", "Ringkasan hasil model terpilih pada data pengujian")
            show_cols = ["Kelompok", "Model", "MAE", "RMSE", "R2"]
            table_eval = chosen[[col for col in show_cols if col in chosen.columns]].copy()
            table_eval = table_eval.rename(columns={
                "Kelompok": "Kelompok Komoditas",
                "R2": "R² Score",
            })
            st.dataframe(
                table_eval.style.format(
                    {
                        "MAE": "Rp {:,.2f}",
                        "RMSE": "Rp {:,.2f}",
                        "R² Score": "{:.4f}",
                    },
                    na_rep=NA_LABEL,
                ),
                hide_index=True,
                use_container_width=True,
                height=350,
            )

    
    with chart_col:
        with st.container(border=True):

            panel_heading(
                "🗓️",
                "Prediksi Harga 1 Hari ke Depan",
                "Estimasi harga seluruh jenis komoditas pada hari kerja berikutnya"
            )

            predictions, prediction_errors, next_date = (
                get_all_next_day_predictions(
                    DATA_MTIME,
                    get_model_cache_signature()
                )
            )

            st.caption(
                f"Data terakhir: {df_data.index.max():%d/%m/%Y} | "
                f"Tanggal prediksi: {next_date:%d/%m/%Y}"
            )
            

            if predictions.empty:
                st.warning(
                    "Prediksi belum tersedia. "
                    "Pastikan model dan dataset sudah tersedia."
                )

            else:
                tabel = predictions.copy()

                kolom_tampil = [
                    "Jenis Komoditas",
                    "Harga Terakhir",
                    "Prediksi Besok"
                ]

                st.dataframe(
                    tabel[kolom_tampil].rename(
                        columns={"Prediksi Besok": "Harga Prediksi"}
                    ).style.format(
                        {
                            "Harga Terakhir": lambda x: f"Rp {x:,.0f}".replace(",", "."),
                            "Harga Prediksi": lambda x: f"Rp {x:,.0f}".replace(",", ".")
                        },
                        na_rep="—"
                    ),
                    hide_index=True,
                    use_container_width=True,
                    height=225
                )

                st.caption(
                    f"Menampilkan {len(tabel)} dari "
                    f"{len(predictions)} jenis komoditas."
                )


            if prediction_errors:
                st.warning(
                    f"{len(prediction_errors)} komoditas "
                    "belum berhasil diprediksi."
                )

                with st.expander("Detail Kendala"):
                    for error in prediction_errors:
                        st.write(error)



    