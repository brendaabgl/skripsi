import calendar
from html import escape
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
    historical_range,
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

MAX_STEPS = 250
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

    /* Dropdown: tema gelap Streamlit membuat teks abu-abu di atas latar putih. Paksa teks gelap yang jelas. */
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #c3d2df !important;
        border-radius: 12px !important;
    }
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        color: #14213d !important;
        -webkit-text-fill-color: #14213d !important;
        font-weight: 600;
    }
    div[data-baseweb="select"] input::placeholder {
        color: #6b7b8f !important;
        -webkit-text-fill-color: #6b7b8f !important;
    }
    div[data-baseweb="select"] svg {
        fill: #14213d !important;
        color: #14213d !important;
    }
    /* Daftar hasil pencarian menempel langsung di bawah kotak input (tanpa celah) */
    div[data-baseweb="popover"]:has(ul[role="listbox"]) {
        margin-top: -6px !important;
    }
    div[data-baseweb="popover"]:has(ul[role="listbox"]) > div,
    ul[role="listbox"] {
        background: #ffffff !important;
        border: 1px solid #c3d2df !important;
        border-radius: 0 0 12px 12px !important;
        box-shadow: 0 6px 14px rgba(31, 55, 80, 0.10) !important;
    }
    ul[role="listbox"] li,
    li[role="option"] {
        background: #ffffff !important;
    }
    ul[role="listbox"] li *,
    li[role="option"] * {
        color: #14213d !important;
        -webkit-text-fill-color: #14213d !important;
    }
    ul[role="listbox"] li:hover,
    li[role="option"][aria-selected="true"],
    li[role="option"]:hover {
        background: #e5eef5 !important;
    }
    /* Kotak tanggal ikut terang dan terbaca */
    div[data-testid="stDateInput"] > div > div {
        background: #ffffff !important;
        border: 1px solid #c3d2df !important;
        border-radius: 12px !important;
    }
    div[data-testid="stDateInput"] input {
        background: #ffffff !important;
        color: #14213d !important;
        -webkit-text-fill-color: #14213d !important;
        font-weight: 600;
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

    /* ===== Halaman awam (Harga Hari Ini): besar, jelas, responsif di HP ===== */
    div[data-testid="stRadio"] label p { color: #14213d !important; font-weight: 650; }
    .home-sum { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: 12px 0 18px; }
    .home-sum-card { border-radius: 18px; padding: 16px 18px; text-align: center; border: 2px solid #dce5ed; background: #fff; }
    .home-sum-card.up { border-color: #f3b4b4; background: #fff5f5; }
    .home-sum-card.down { border-color: #b7e1c4; background: #f1faf4; }
    .home-sum-card.flat { border-color: #d5dde5; background: #f6f8fa; }
    .home-sum-num { font-size: 2.5rem; font-weight: 900; line-height: 1.15; }
    .home-sum-card.up .home-sum-num { color: #c62828; }
    .home-sum-card.down .home-sum-num { color: #2e7d32; }
    .home-sum-card.flat .home-sum-num { color: #546e7a; }
    .home-sum-text { font-size: 1.1rem; font-weight: 800; color: #14213d; }
    .home-sum-sub { font-size: 0.92rem; color: #5b6b7e; }
    .mv-box { border-radius: 16px; padding: 12px 16px; background: #fff; border: 1px solid #dce5ed; border-left: 8px solid #94a3b8; }
    .mv-box.up { border-left-color: #c62828; }
    .mv-box.down { border-left-color: #2e7d32; }
    .mv-title { font-size: 1.05rem; font-weight: 850; color: #14213d; margin-bottom: 6px; }
    .mv-row { display: flex; justify-content: space-between; gap: 10px; font-size: 1.02rem; padding: 3px 0; color: #26364d; }
    .mv-box.up .mv-val { color: #c62828; font-weight: 800; }
    .mv-box.down .mv-val { color: #2e7d32; font-weight: 800; }
    .mv-empty { font-size: 1rem; color: #6b7b8f; }
    .home-section { font-size: 1.4rem; font-weight: 850; color: #14213d; margin: 26px 0 8px; }
    .home-subsection { font-size: 1.15rem; font-weight: 800; color: #14213d; margin: 14px 0 8px; }
    .pc-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 14px; margin-top: 6px; }
    .pc-card { background: #fff; border-radius: 16px; padding: 14px 16px; border: 1px solid #dce5ed; border-left: 8px solid #94a3b8; box-shadow: 0 2px 8px rgba(31, 55, 80, 0.05); }
    .pc-card.up { border-left-color: #c62828; }
    .pc-card.down { border-left-color: #2e7d32; }
    .pc-name { font-size: 1.05rem; font-weight: 750; color: #26364d; }
    .pc-price { font-size: 1.85rem; font-weight: 900; color: #14213d; margin: 4px 0; }
    .pc-change { font-size: 1rem; font-weight: 800; color: #546e7a; }
    .pc-card.up .pc-change { color: #c62828; }
    .pc-card.down .pc-change { color: #2e7d32; }
    .pc-note { font-size: 0.85rem; color: #6b7b8f; }
    .detail-box { background: #fff; border: 1px solid #dce5ed; border-radius: 18px; padding: 16px 20px; margin: 8px 0 12px; }
    .detail-name { font-size: 1.25rem; font-weight: 800; color: #26364d; }
    .detail-price { font-size: 2.6rem; font-weight: 900; color: #14213d; line-height: 1.2; }
    .detail-text { font-size: 1.1rem; color: #26364d; }
    .pf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-bottom: 12px; }
    .pf-card { background: #fff; border: 1px solid #dce5ed; border-radius: 16px; padding: 14px 16px; }
    .pf-when { font-size: 1.1rem; font-weight: 850; color: #14213d; }
    .pf-date { font-size: 0.9rem; color: #6b7b8f; margin-bottom: 6px; }
    .pf-price { font-size: 1.7rem; font-weight: 900; color: #14213d; }
    .pf-phrase { font-size: 1rem; font-weight: 800; color: #546e7a; }
    .pf-phrase.up { color: #c62828; }
    .pf-phrase.down { color: #2e7d32; }
    .pf-range { font-size: 0.92rem; color: #46566a; margin-top: 8px; }
    .audience-badge { display: inline-block; background: #e8eef9; color: #26427a; border-radius: 999px; padding: 4px 12px; font-size: 0.82rem; font-weight: 700; margin: 2px 0 10px; }

    /* Penyesuaian layout agar menyerupai referensi terbaru */
    .stApp { background: #ffffff; }
    .block-container { max-width: 1380px; padding-top: .55rem; }
    .page-title { font-size: 1.65rem; margin: .1rem 0 .65rem; }
    .top-rule { margin: .15rem 0 .45rem; height: 2px; }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.filter-title) {
        background: linear-gradient(115deg, #dce7ef, #cfdee9) !important;
        border: none !important;
        border-radius: 20px !important;
        padding: .8rem 1.1rem !important;
        box-shadow: none !important;
        margin-bottom: .45rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.filter-title) label {
        font-weight: 750 !important;
    }
    .panel-heading { font-size: 1.05rem; }
    .panel-icon { width: 30px; height: 30px; min-width: 30px; font-size: 1rem; }
    .panel-caption { margin-left: 40px; }
    .forecast-summary { padding-top: 14px; }
    .forecast-item { display: flex; align-items: flex-start; gap: 10px; padding: 10px 4px 7px; min-height: 85px; }
    .forecast-symbol { background: #e5eef5; border-radius: 50%; width: 30px; height: 30px;
        min-width: 30px; display: inline-flex; align-items: center; justify-content: center;
        color: #1d344d; font-size: 1rem; }
    .forecast-label { font-size: .75rem; color: #334155; font-weight: 750; }
    .forecast-value { font-size: 1.3rem; font-weight: 850; color: #111c34; line-height: 1.35; }
    .forecast-foot { font-size: .69rem; color: #64748b; }
    .forecast-price { padding-top: 2px; }
    @media (max-width: 750px) { .forecast-value { font-size: 1.06rem; } }
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
    for col in ["MAE", "RMSE"]:
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



PROJECTION_STEPS = 780          # 3 tahun x 260 hari kerja
BDAYS_PER_YEAR = 260


@st.cache_data(show_spinner=False)
def get_long_term_projection(attr, data_mtime, model_versions):
    """Proyeksi otomatis 3 tahun (hari kerja) dari data terakhir + rentang historis. Di-cache per atribut."""
    model, info = load_model(attr)
    series = df_data[attr].dropna()
    fut = predict_future_recursive(model, series, PROJECTION_STEPS, int(info["window"]))
    band = historical_range(series, PROJECTION_STEPS)
    band.index = fut.index
    return fut, band, info["model_name"]


# -----------------------------------------------------------------------------
# NAVIGASI HORIZONTAL
# -----------------------------------------------------------------------------
nav_brand, nav_menu, nav_refresh = st.columns([2.0, 5.4, 0.25])
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
        ["Dashboard", "Prediksi", "Evaluasi Model"],
        index=0,
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
# =============================================================================
# HALAMAN 1 - HARGA HARI INI (UNTUK MASYARAKAT UMUM)
# =============================================================================
if page == "Dashboard":
    # Halaman untuk masyarakat umum: bahasa sederhana, tulisan besar, tanpa istilah teknis.
    HARI_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    BULAN_ID = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    GROUP_EMOJI = {
        "Beras": "🍚", "Daging Ayam": "🍗", "Daging Sapi": "🥩", "Telur Ayam": "🥚",
        "Bawang Merah": "🧅", "Bawang Putih": "🧄", "Cabai Merah": "🌶️", "Cabai Rawit": "🌶️",
        "Minyak Goreng": "🛢️", "Gula Pasir": "🍬",
    }

    def rp_id(value):
        return "Rp " + f"{float(value):,.0f}".replace(",", ".")

    def pct_id(value, digits=1):
        return f"{abs(float(value)):.{digits}f}".replace(".", ",") + "%"

    def date_id(ts, with_day=True):
        ts = pd.Timestamp(ts)
        core = f"{ts.day} {BULAN_ID[ts.month - 1]} {ts.year}"
        return f"{HARI_ID[ts.dayofweek]}, {core}" if with_day else core

    # --- hitung ringkasan tiap bahan pangan (dibanding 5 hari kerja sebelumnya = "minggu lalu") ---
    stats = []
    for attr in [a for a in ATTR_TO_GROUP if a in df_data.columns]:
        s_attr = df_data[attr].dropna()
        if s_attr.empty:
            continue
        last = float(s_attr.iloc[-1])
        ref = float(s_attr.iloc[-6]) if len(s_attr) >= 6 else float(s_attr.iloc[0])
        diff = last - ref
        pct = diff / ref * 100 if ref else 0.0
        status = "up" if pct > 0.05 else ("down" if pct < -0.05 else "flat")
        stats.append({
            "attr": attr, "group": ATTR_TO_GROUP[attr], "last": last, "diff": diff, "pct": pct,
            "status": status, "avg_month": float(s_attr.tail(22).mean()),
        })

    st.markdown('<div class="page-title">Harga Pangan Hari Ini</div>', unsafe_allow_html=True)
    if df_data.empty or not stats:
        st.info("Data harga belum tersedia.")
        st.stop()
    last_data_date = pd.Timestamp(df_data.index.max()).normalize()
    st.markdown(
        f'<div class="page-subtitle">Pasar modern DKI Jakarta · Harga terakhir tercatat '
        f'<b>{date_id(last_data_date)}</b></div>',
        unsafe_allow_html=True,
    )
    if (pd.Timestamp.today().normalize() - last_data_date).days > 4:
        st.warning(
            f"Harga terbaru belum dimasukkan. Yang tampil adalah harga tanggal {date_id(last_data_date, False)}."
        )

    n_up = sum(1 for s in stats if s["status"] == "up")
    n_down = sum(1 for s in stats if s["status"] == "down")
    n_flat = sum(1 for s in stats if s["status"] == "flat")
    st.markdown(
        '<div class="home-sum">'
        f'<div class="home-sum-card up"><div class="home-sum-num">▲ {n_up}</div>'
        '<div class="home-sum-text">bahan pangan NAIK</div><div class="home-sum-sub">dibanding minggu lalu</div></div>'
        f'<div class="home-sum-card down"><div class="home-sum-num">▼ {n_down}</div>'
        '<div class="home-sum-text">bahan pangan TURUN</div><div class="home-sum-sub">dibanding minggu lalu</div></div>'
        f'<div class="home-sum-card flat"><div class="home-sum-num">● {n_flat}</div>'
        '<div class="home-sum-text">bahan pangan TETAP</div><div class="home-sum-sub">dibanding minggu lalu</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- yang paling berubah minggu ini ---
    def mover_box(title, items, css):
        rows = "".join(
            f'<div class="mv-row"><span class="mv-name">{GROUP_EMOJI.get(i["group"], "")} {i["attr"]}</span>'
            f'<span class="mv-val">{"▲" if i["status"] == "up" else "▼"} {pct_id(i["pct"])}</span></div>'
            for i in items
        ) or '<div class="mv-empty">Tidak ada</div>'
        return f'<div class="mv-box {css}"><div class="mv-title">{title}</div>{rows}</div>'

    movers_up = sorted([s for s in stats if s["status"] == "up"], key=lambda s: -s["pct"])[:3]
    movers_down = sorted([s for s in stats if s["status"] == "down"], key=lambda s: s["pct"])[:3]
    mv1, mv2 = st.columns(2, gap="medium")
    with mv1:
        st.markdown(mover_box("Naik paling tinggi minggu ini", movers_up, "up"), unsafe_allow_html=True)
    with mv2:
        st.markdown(mover_box("Turun paling banyak minggu ini", movers_down, "down"), unsafe_allow_html=True)

    # --- daftar harga ---
    st.markdown('<div class="home-section">Daftar Harga Bahan Pangan</div>', unsafe_allow_html=True)
    groups = list(dict.fromkeys(s["group"] for s in stats))
    chosen_group = st.radio(
        "Pilih kelompok bahan pangan", ["Semua"] + groups, horizontal=True,
        key="home_group", label_visibility="collapsed",
    )
    cards = ""
    for s in stats:
        if chosen_group != "Semua" and s["group"] != chosen_group:
            continue
        if s["status"] == "flat":
            change = "● Harga tetap"
        else:
            word = "Naik" if s["status"] == "up" else "Turun"
            arrow = "▲" if s["status"] == "up" else "▼"
            change = f"{arrow} {word} {rp_id(abs(s['diff']))} ({pct_id(s['pct'])})"
        cards += (
            f'<div class="pc-card {s["status"]}">'
            f'<div class="pc-name">{GROUP_EMOJI.get(s["group"], "")} {s["attr"]}</div>'
            f'<div class="pc-price">{rp_id(s["last"])}</div>'
            f'<div class="pc-change">{change}</div>'
            '<div class="pc-note">dibanding minggu lalu</div></div>'
        )
    st.markdown(f'<div class="pc-grid">{cards}</div>', unsafe_allow_html=True)

    # --- perkembangan dan perkiraan satu bahan ---
    st.markdown('<div class="home-section">Lihat Perkembangan dan Perkiraan Harga</div>', unsafe_allow_html=True)
    detail_attr = st.selectbox(
        "Pilih bahan pangan", [s["attr"] for s in stats], key="home_detail",
    )
    ds = next(s for s in stats if s["attr"] == detail_attr)
    series_d = df_data[detail_attr].dropna()
    diff_avg = (ds["last"] - ds["avg_month"]) / ds["avg_month"] * 100 if ds["avg_month"] else 0.0
    if diff_avg > 1:
        avg_text = f"<b>{pct_id(diff_avg)} lebih mahal</b> dari rata-rata sebulan terakhir"
    elif diff_avg < -1:
        avg_text = f"<b>{pct_id(diff_avg)} lebih murah</b> dari rata-rata sebulan terakhir"
    else:
        avg_text = "<b>hampir sama</b> dengan rata-rata sebulan terakhir"
    st.markdown(
        f'<div class="detail-box"><div class="detail-name">{GROUP_EMOJI.get(ds["group"], "")} {detail_attr}</div>'
        f'<div class="detail-price">{rp_id(ds["last"])}</div>'
        f'<div class="detail-text">Harga {date_id(last_data_date, False)} {avg_text}.</div></div>',
        unsafe_allow_html=True,
    )

    import matplotlib.dates as mdates

    # Pilihan ini mengubah horizon yang terlihat pada GRAFIK (bukan hanya kartu informasi).
    st.markdown('<div class="home-subsection">Grafik Harga dan Prediksi</div>', unsafe_allow_html=True)
    graph_period = st.radio(
        "Tampilkan prediksi pada grafik",
        options=[1, 5, 22],
        format_func=lambda n: {1: "1 Hari ke Depan", 5: "1 Minggu ke Depan", 22: "1 Bulan ke Depan"}[n],
        horizontal=True,
        key="dashboard_graph_horizon",
    )

    model_d, info_d, error_d = get_model_and_info(detail_attr)
    future_d = None
    if model_d is not None:
        try:
            future_d = predict_future_recursive(
                model_d, series_d, 22, window_size=int(info_d["window"])
            )
        except Exception as exc:
            error_d = str(exc)
    try:
        band_d = historical_range(series_d, 22)
    except Exception:
        band_d = None
    next_dates = pd.bdate_range(start=last_data_date + pd.Timedelta(days=1), periods=22)

    # Ukuran grafik dibuat lebih ringkas daripada versi sebelumnya.
    hist_d = series_d.tail(90)
    chart_spacer_l, chart_center, chart_spacer_r = st.columns([1, 3.4, 1])
    with chart_center:
        fig_d, ax_d = plt.subplots(figsize=(7.2, 2.7))
        ax_d.plot(hist_d.index, hist_d.values, color="#2563eb", linewidth=1.8, label="Harga asli")
        ax_d.axhline(ds["avg_month"], color="#64748b", linestyle="--", linewidth=1.1, label="Rata-rata 1 bulan")
        ax_d.scatter([hist_d.index[-1]], [hist_d.iloc[-1]], color="#f08c2b", s=30, zorder=5)
        if future_d is not None and len(future_d) >= graph_period:
            shown_future = future_d.iloc[:graph_period]
            ax_d.plot(
                [hist_d.index[-1]] + list(shown_future.index),
                [float(hist_d.iloc[-1])] + list(shown_future.values),
                color="#e58a31", linestyle="--", marker="o", markersize=2.5,
                linewidth=1.7, label="Prediksi",
            )
            tight_ylim(ax_d, hist_d.values, shown_future.values, [ds["avg_month"]])
        else:
            tight_ylim(ax_d, hist_d.values, [ds["avg_month"]])
        ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
        ax_d.tick_params(labelsize=8)
        ax_d.set_ylabel("Harga (Rp)", fontsize=9)
        ax_d.set_xlabel("Tanggal", fontsize=9)
        ax_d.grid(True, alpha=0.20, linestyle="--")
        ax_d.legend(loc="best", fontsize=8)
        format_rp_axis(ax_d)
        fig_d.autofmt_xdate(rotation=20)
        fig_d.tight_layout()
        st.pyplot(fig_d, use_container_width=True)
        plt.close(fig_d)
        st.caption(
            f"Garis biru: harga asli 90 observasi terakhir. "
            f"Garis oranye: prediksi sampai {graph_period} hari kerja ke depan."
        )
        if future_d is None:
            st.warning(f"Grafik prediksi belum tersedia. {error_d or 'Model belum dapat dijalankan.'}")

    st.markdown('<div class="home-subsection">Perkiraan ke depan</div>', unsafe_allow_html=True)
    fc_cards = ""
    for steps, label in ((1, "Hari berikutnya"), (5, "Seminggu lagi"), (22, "Sebulan lagi")):
        when = date_id(next_dates[steps - 1])
        if future_d is not None:
            price = float(future_d.iloc[steps - 1])
            pct = (price - ds["last"]) / ds["last"] * 100 if ds["last"] else 0.0
            if abs(pct) < 0.5:
                phrase, css = "Hampir sama dengan sekarang", "flat"
            elif pct > 0:
                phrase, css = f"Cenderung naik sekitar {pct_id(pct)}", "up"
            else:
                phrase, css = f"Cenderung turun sekitar {pct_id(pct)}", "down"
            price_html = f'<div class="pf-price">{rp_id(price)}</div><div class="pf-phrase {css}">{phrase}</div>'
        else:
            price_html = '<div class="pf-phrase flat">Perkiraan belum tersedia</div>'
        range_html = ""
        if band_d is not None and np.isfinite(band_d["lo"].iloc[steps - 1]):
            lo_v, hi_v = float(band_d["lo"].iloc[steps - 1]), float(band_d["hi"].iloc[steps - 1])
            if hi_v - lo_v < 1:
                range_html = '<div class="pf-range">Biasanya tidak berubah dalam periode ini</div>'
            else:
                range_html = f'<div class="pf-range">Biasanya berkisar <b>{rp_id(lo_v)}</b> sampai <b>{rp_id(hi_v)}</b></div>'
        fc_cards += (
            f'<div class="pf-card"><div class="pf-when">{label}</div><div class="pf-date">{when}</div>'
            f'{price_html}{range_html}</div>'
        )
    st.markdown(f'<div class="pf-grid">{fc_cards}</div>', unsafe_allow_html=True)
    st.info(
        "Perkiraan ini dibuat komputer dari pola harga sebelumnya, jadi hanya gambaran dan bukan kepastian. "
        "Harga pangan bisa naik atau turun mendadak karena cuaca, pasokan, atau hari raya. "
        "\"Biasanya berkisar\" menunjukkan rentang perubahan yang pernah terjadi pada bahan ini."
    )


# =============================================================================
# HALAMAN 2 - DATA DAN VISUALISASI HARGA
# =============================================================================
elif page == "Prediksi":
    st.markdown('<div class="page-title">Prediksi Harga Pangan</div>', unsafe_allow_html=True)

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

        # Batas maksimal pemilihan tanggal: 250 hari kerja ke depan.
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

    # Rentang harga wajar (sebaran historis perubahan harga), bukan ramalan titik
    price_band = None
    if future_series is not None and len(future_series):
        try:
            price_band = historical_range(series, len(future_series))
            price_band.index = future_series.index
            if price_band.isna().all().all():
                price_band = None
        except Exception:
            price_band = None
        st.caption(
                    "**⚠️ Prediksi adalah estimasi, bukan harga pasti!!!** kalender hanya memperhitungkan Senin–Jumat."
                )

    # ===============================================================
    # BARIS 1: DATA HARGA (KIRI) + RINGKASAN PREDIKSI (KANAN)
    # BARIS 2: GRAFIK HARGA ASLI (KIRI) + GRAFIK PREDIKSI (KANAN)
    # ===============================================================
    last_price = float(series.iloc[-1])
    has_prediction = future_series is not None and len(future_series) > 0
    if has_prediction:
        predicted_price = float(future_series.iloc[-1])
        price_change = predicted_price - last_price
        percent_change = price_change / last_price * 100 if last_price else 0.0
        movement_arrow = "↑" if percent_change > 0 else ("↓" if percent_change < 0 else "→")
        movement_color = "#16803c" if percent_change > 0 else ("#dc2626" if percent_change < 0 else "#475569")
        movement_text = f"{movement_arrow} {abs(percent_change):.1f}%".replace(".", ",")
        price_text = "Rp " + f"{predicted_price:,.0f}".replace(",", ".")
    else:
        movement_color = "#64748b"
        movement_text = "Belum tersedia"
        price_text = "Belum tersedia"

    recent = series.tail(5)
    recent_table = pd.DataFrame({
        "Tanggal": recent.index.strftime("%d/%m/%Y"),
        "Komoditas": selected_attr,
        "Harga (Rp)": recent.values,
    })

    top_left, top_right = st.columns([1, 1], gap="large")
    with top_left:
        panel_heading("▤", "Data Harga Terbaru", f"Harga {selected_attr.lower()} beberapa hari terakhir")
        st.dataframe(
            recent_table.style.format({"Harga (Rp)": lambda x: f"{x:,.0f}".replace(",", ".")}),
            hide_index=True, use_container_width=True, height=220,
        )

    with top_right:
        st.markdown('<div class="forecast-summary">', unsafe_allow_html=True)
        metric_left, metric_right = st.columns(2, gap="small")
        with metric_left:
            st.markdown(
                '<div class="forecast-item"><span class="forecast-symbol">▣</span>'
                '<div><div class="forecast-label">Tanggal Prediksi</div>'
                f'<div class="forecast-value">{steps_ahead} Hari Kerja ke Depan</div>'
                f'<div class="forecast-foot">Target: {effective_date_text}</div></div></div>',
                unsafe_allow_html=True,
            )
        with metric_right:
            st.markdown(
                '<div class="forecast-item"><span class="forecast-symbol">▥</span>'
                '<div><div class="forecast-label">Perkiraan Kenaikan</div>'
                f'<div class="forecast-value" style="color:{movement_color}">{movement_text}</div>'
                '<div class="forecast-foot">Dibanding harga terakhir</div></div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="forecast-item forecast-price"><span class="forecast-symbol">▤</span>'
            '<div><div class="forecast-label">Perkiraan Harga</div>'
            f'<div class="forecast-value">{price_text}</div>'
            '<div class="forecast-foot">Estimasi harga pada tanggal pilihan</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        if not has_prediction:
            st.warning(f"Model prediksi belum tersedia. {prediction_error or ''}")

    chart_left, chart_right = st.columns([1, 1], gap="large")
    history_chart = series.tail(HISTORY_WINDOW)  # Riwayat singkat untuk grafik prediksi
    full_history_chart = series  # Seluruh data yang tersedia, tanpa batas 180 baris
    with chart_left:
        panel_heading("⌁", "Grafik Harga Asli", f"Seluruh riwayat dataset: {full_history_chart.index.min():%d/%m/%Y}–{full_history_chart.index.max():%d/%m/%Y} ({len(full_history_chart):,} data)")
        fig_real, ax_real = plt.subplots(figsize=(7, 3.2))
        ax_real.plot(full_history_chart.index, full_history_chart.values, color="#256bd8", linewidth=1.6)
        ax_real.set_ylabel("Harga (Rp)")
        ax_real.set_xlabel("Tanggal")
        ax_real.grid(True, alpha=0.25, linestyle="--")
        tight_ylim(ax_real, full_history_chart.values)
        format_rp_axis(ax_real)
        fig_real.autofmt_xdate(rotation=0)
        fig_real.tight_layout()
        st.pyplot(fig_real, use_container_width=True)
        plt.close(fig_real)

    with chart_right:
        panel_heading("⌁", "Grafik PREDIKSI Perubahan Harga", "Memudahkan melihat tren harga")
        fig_pred, ax_pred = plt.subplots(figsize=(7, 3.2))
        ax_pred.plot(history_chart.index, history_chart.values, color="#256bd8", linewidth=1.6, label="Harga asli")
        if has_prediction:
            forecast_x = [history_chart.index[-1]] + list(future_series.index)
            forecast_y = [last_price] + list(future_series.values)
            ax_pred.plot(forecast_x, forecast_y, color="#e58a31", linewidth=2, linestyle="--", marker="o", markersize=2.8, label="Prediksi")
            if price_band is not None:
                ax_pred.fill_between(
                    list(price_band.index), price_band["lo"].values, price_band["hi"].values,
                    color="#e58a31", alpha=0.12, label="Rentang historis",
                )
            ax_pred.axvline(history_chart.index[-1], color="#94a3b8", linewidth=1, linestyle=":")
            tight_ylim(ax_pred, history_chart.values, future_series.values,
                       None if price_band is None else price_band["lo"].dropna().values,
                       None if price_band is None else price_band["hi"].dropna().values)
        else:
            tight_ylim(ax_pred, history_chart.values)
        ax_pred.set_ylabel("Harga (Rp)")
        ax_pred.set_xlabel("Tanggal")
        ax_pred.grid(True, alpha=0.25, linestyle="--")
        ax_pred.legend(loc="best", fontsize=8)
        format_rp_axis(ax_pred)
        fig_pred.autofmt_xdate(rotation=0)
        fig_pred.tight_layout()
        st.pyplot(fig_pred, use_container_width=True)
        plt.close(fig_pred)

    if has_prediction:
        st.caption(
            "Prediksi adalah estimasi, bukan harga pasti; kalender hanya memperhitungkan Senin–Jumat."
        )


# =============================================================================
# HALAMAN 3 - EVALUASI MODEL
# =============================================================================

elif page == "Evaluasi Model":
    st.markdown('<div class="page-title">Evaluasi Model</div>', unsafe_allow_html=True)

    chosen = get_chosen_results(df_eval_results)
    # Hanya tampilkan jenis komoditas yang memiliki kelompok terpetakan.
    # Data evaluasi aslinya tetap utuh; baris kategori "Lainnya" hanya disembunyikan.
    chosen["Kelompok"] = chosen["Atribut"].map(ATTR_TO_GROUP)
    chosen = chosen[chosen["Kelompok"].notna()].copy()

    # Pertahankan dua panel berdampingan; gunakan tabel HTML responsif
    # agar kolom R² tetap terlihat tanpa scroll horizontal.
    table_col, chart_col = st.columns([1.15, 1], gap="medium")

    with table_col:
        with st.container(border=True):
            panel_heading("▤", "Fitur Evaluasi Model", "Setiap baris menunjukkan hasil evaluasi untuk satu jenis komoditas")
            show_cols = ["Kelompok", "Atribut", "Model", "MAE", "RMSE", "R2"]
            table_eval = chosen[[col for col in show_cols if col in chosen.columns]].copy()
            table_eval = table_eval.rename(columns={
                "Kelompok": "Kelompok Komoditas",
                "Atribut": "Jenis Komoditas",
                "R2": "R² Score",
            })
            # Tabel HTML memakai table-layout: fixed: semua enam kolom
            # selalu muat di dalam panel kiri, termasuk kolom R².
            st.markdown("""
                <style>
                .eval-table-scroll {
                    max-height: 350px;
                    overflow-y: auto;
                    overflow-x: hidden;
                    border: 1px solid #dce5ed;
                    border-radius: 9px;
                    width: 100%;
                }
                .eval-table {
                    table-layout: fixed;
                    border-collapse: collapse;
                    width: 100%;
                    font-family: Arial, sans-serif;
                    font-size: clamp(8px, .72vw, 10.5px);
                    color: #26364d;
                }
                .eval-table th, .eval-table td {
                    border-bottom: 1px solid #e7edf2;
                    border-right: 1px solid #e7edf2;
                    padding: 7px 3px;
                    vertical-align: middle;
                    overflow-wrap: anywhere;
                }
                .eval-table th {
                    background: #f7f9fc;
                    color: #64748b;
                    font-weight: 600;
                    position: sticky;
                    top: 0;
                    z-index: 1;
                    text-align: left;
                }
                .eval-table td.metric { text-align: right; white-space: nowrap; }
                .eval-table td:last-child, .eval-table th:last-child { border-right: none; }
                .eval-table tbody tr:hover { background: #f0f6fa; }
                </style>
            """, unsafe_allow_html=True)

            table_rows = []
            for _, result in table_eval.iterrows():
                def metric_money(key):
                    value = result[key]
                    return (f"Rp {float(value):,.2f}" if pd.notna(value) else NA_LABEL)

                r2 = (f"{float(result['R² Score']):.4f}"
                      if pd.notna(result['R² Score']) else NA_LABEL)
                table_rows.append(
                    "<tr>"
                    f"<td>{escape(str(result['Kelompok Komoditas']))}</td>"
                    f"<td>{escape(str(result['Jenis Komoditas']))}</td>"
                    f"<td>{escape(str(result['Model']))}</td>"
                    f"<td class='metric'>{escape(metric_money('MAE'))}</td>"
                    f"<td class='metric'>{escape(metric_money('RMSE'))}</td>"
                    f"<td class='metric'>{escape(r2)}</td>"
                    "</tr>"
                )
            html_table = (
                '<div class="eval-table-scroll"><table class="eval-table">'
                '<colgroup><col style="width:15%"><col style="width:29%">'
                '<col style="width:12%"><col style="width:15%">'
                '<col style="width:16%"><col style="width:13%"></colgroup>'
                '<thead><tr><th>Kelompok</th><th>Jenis Komoditas</th>'
                '<th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>'
                '<tbody>' + ''.join(table_rows) + '</tbody></table></div>'
            )
            st.markdown(html_table, unsafe_allow_html=True)

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

            st.caption(
                "Prediksi merupakan estimasi berdasarkan data historis, "
                "bukan kepastian harga. Hari kerja dihitung "
                "Senin–Jumat, belum memperhitungkan libur nasional."
            )


    # Ringkasan berdasarkan model yang dipilih untuk setiap atribut.
    chosen, best_model_row = get_best_model_summary(df_eval_results)
    xgb_chosen = chosen[
        chosen["Model"].astype(str).str.strip().str.lower() == "xgboost"
    ]
    if not xgb_chosen.empty:
        xgb_mae = fmt_rp(xgb_chosen["MAE"].mean())
        xgb_rmse = fmt_rp(xgb_chosen["RMSE"].mean())
        xgb_r2 = fmt_r2(xgb_chosen["R2"].mean())
        xgb_count = xgb_chosen["Atribut"].nunique()
        xgb_section = f"""
            <div style="border-top:1px solid #cbd9e5;margin-top:12px;padding-top:10px;
                        display:grid;grid-template-columns:2.2fr 1fr 1fr 1fr;gap:12px;align-items:center;">
                <div>
                    <div style="font-size:.90rem;font-weight:800;color:#26364d;">XGBoost</div>
                    <div style="font-size:.70rem;color:#607189;">
                        Rata-rata pengujian pada {xgb_count} atribut yang memilih XGBoost
                    </div>
                </div>
                <div style="text-align:center;font-size:.78rem;color:#475569;">MAE<br>
                    <b style="color:#14213d;font-size:.90rem;">{xgb_mae}</b></div>
                <div style="text-align:center;font-size:.78rem;color:#475569;">RMSE<br>
                    <b style="color:#14213d;font-size:.90rem;">{xgb_rmse}</b></div>
                <div style="text-align:center;font-size:.78rem;color:#475569;">R² Score<br>
                    <b style="color:#14213d;font-size:.90rem;">{xgb_r2}</b></div>
            </div>
        """
        # Hilangkan indentasi dan newline agar HTML tidak dibaca sebagai blok kode Markdown.
        xgb_section = "".join(line.strip() for line in xgb_section.splitlines())
    else:
        xgb_section = ""  # Tidak mengarang nilai jika tidak ada atribut terpilih XGBoost.

    st.write("")
    st.markdown(
        f"""
        <div class="bottom-band">
            <div style="display:grid;grid-template-columns:2.2fr 1fr 1fr 1fr;gap:12px;align-items:center;">
                <div>
                    <div class="panel-heading"><span class="panel-icon">🏆</span>Model dengan Kinerja Prediksi Terbaik</div>
                    <div class="bottom-model">{best_model_row['Model']}</div>
                    <div class="bottom-caption">Nilai metrik merupakan rata-rata hasil pengujian
                        untuk atribut yang memilih model tersebut.
                    </div>
                </div>
                <div class="mini-metric"><div class="mini-label">MAE</div><div class="mini-value">{fmt_rp(best_model_row['MAE'])}</div></div>
                <div class="mini-metric"><div class="mini-label">RMSE</div><div class="mini-value">{fmt_rp(best_model_row['RMSE'])}</div></div>
                <div class="mini-metric"><div class="mini-label">R² Score</div><div class="mini-value">{fmt_r2(best_model_row['R2'])}</div></div>
            </div>{xgb_section}</div>
        """,
        unsafe_allow_html=True,
    )
