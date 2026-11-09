"""
common.py - Kode yang dipakai bersama oleh train.py, predict.py, dan app.py.

Aturan penting: fungsi build_features() (training) dan predict_future_recursive() (prediksi)
harus membuat fitur yang SAMA PERSIS, makanya keduanya ditaruh di satu file ini.
"""
# json dipakai untuk menyimpan dan membaca informasi pendamping model, misalnya jenis model,
# ukuran window, dan hasil evaluasinya. Informasi ini diperlukan saat model dimuat lagi.
import json
# os membantu mencari file, membuat folder, dan menyusun lokasi penyimpanan
# supaya program tidak perlu menulis alamat file secara manual berkali-kali.
import os

# joblib disiapkan untuk membaca file model dari format penyimpanan lama.
# Model dengan format baru disimpan dengan format asli XGBoost atau LightGBM.
import joblib
# NumPy digunakan untuk perhitungan angka, simpangan baku, kuantil, dan perbandingan fitur.
import numpy as np
# Pandas digunakan untuk membaca Excel serta mengolah data harga berdasarkan tanggal.
import pandas as pd

# Nama dataset dan folder model ditentukan sekali di sini agar semua fungsi
# memakai sumber data dan lokasi model yang konsisten.
DATASET_NAME = "dataset_harga_preprocessed.xlsx"
MODELS_DIR = "models"

# Pemetaan Atribut ke Kelompok Komoditas
# Setiap nama kolom harga dipasangkan dengan kelompok komoditasnya.
# Contohnya, beberapa kualitas beras tetap masuk kelompok Beras.
# Pemetaan ini membantu filter pada website dan pemeriksaan kolom dataset.
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
# Fungsi ini mencari dataset di dua lokasi: folder data/ terlebih dahulu, lalu folder utama.
# Kalau tidak ditemukan di keduanya, program berhenti dengan pesan yang menjelaskan nama file.
def find_dataset_path():
    """Cari file Excel di folder 'data' dulu, lalu di folder utama proyek."""
    # Pencarian dilakukan berurutan supaya file di folder data/ lebih diutamakan.
    for p in (os.path.join("data", DATASET_NAME), DATASET_NAME):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"File '{DATASET_NAME}' tidak ditemukan (dicari di folder 'data' dan folder utama)."
    )


# Bagian ini membaca isi Excel sebelum melakukan pengisian data yang kosong.
# Tanggal dijadikan indeks agar urutan waktunya dapat digunakan pada proses prediksi.
def _read_raw(path=None):
    path = path or find_dataset_path()
    df = pd.read_excel(path)
    # Kalau ada kolom Tanggal, ubah menjadi tipe tanggal lalu jadikan indeks tabel.
    if "Tanggal" in df.columns:
        df["Tanggal"] = pd.to_datetime(df["Tanggal"])
        df = df.set_index("Tanggal")
    # Jika tidak ada kolom Tanggal, kode mencoba memakai indeks yang sudah tersedia
    # sebagai tanggal. Ini bukan berarti isi tanggalnya pasti sudah benar.
    elif df.index.name != "Tanggal":
        df.index = pd.to_datetime(df.index)
        df.index.name = "Tanggal"
    # Semua kolom harga dikonversi menjadi angka. Isi yang tidak bisa dibaca sebagai
    # angka diubah menjadi NaN supaya bisa ditangani pada tahap berikutnya.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_index()


# Fungsi ini menyiapkan dataset yang akan digunakan oleh proses training dan prediksi.
# Data mentah dibaca lebih dulu, lalu nilai kosong dicoba diisi secara berurutan.
def load_and_preprocess_data(path=None):
    """Baca Excel -> tabel harga rapi (urut tanggal, sel kosong diisi ffill/bfill/interpolasi)."""
    df = _read_raw(path)
    # ffill mengambil nilai sebelumnya, bfill mengambil nilai sesudahnya jika masih kosong,
    # lalu interpolasi linear diterapkan pada sisa nilai yang memungkinkan.
    # Urutan metode ini penting: hasilnya tidak selalu sama dengan interpolasi dulu.
    return df.ffill().bfill().interpolate(method="linear")


# Pemeriksaan ini hanya menghasilkan pesan peringatan tentang kualitas data.
# Fungsi ini tidak melatih model, tidak memperbarui Excel, dan tidak otomatis
# memperbaiki tanggal ganda maupun nama kolom yang tidak dikenali.
def check_data_quality(path=None, max_stale_days=4):
    """
    Cek cepat setelah data diupdate manual. Mengembalikan daftar pesan peringatan
    (list kosong = aman). Dipakai oleh app.py dan predict.py.
    """
    # Semua temuan dikumpulkan dalam list supaya pemanggil dapat menampilkan
    # lebih dari satu peringatan sekaligus. List kosong berarti tidak ada masalah
    # yang terdeteksi oleh aturan pemeriksaan sederhana ini.
    msgs = []
    raw = _read_raw(path)
    # Tanggal yang muncul dua kali dapat menyebabkan satu hari punya beberapa baris.
    # Karena itu, kode mencatat beberapa contoh tanggal yang ganda.
    if raw.index.duplicated().any():
        dup = raw.index[raw.index.duplicated()].strftime("%Y-%m-%d").tolist()
        msgs.append(f"Ada tanggal ganda di dataset: {', '.join(dup[:5])}. Hapus salah satunya.")
    last_date = raw.index.max()
    # Selisih tanggal hari ini dan tanggal terakhir pada dataset dihitung dalam hari kalender.
    # Jika lebih dari batas yang ditetapkan, pengguna diminta memperbarui data.
    stale = (pd.Timestamp.today().normalize() - last_date).days
    if stale > max_stale_days:
        msgs.append(
            f"Data terakhir {last_date:%d %b %Y} ({stale} hari lalu). Tambahkan harga terbaru ke Excel "
            f"supaya prediksi dimulai dari tanggal yang benar."
        )
    # Baris harga paling akhir diperiksa karena akan menjadi titik awal prediksi berikutnya.
    # Nilai yang masih kosong diberi peringatan, meskipun proses preprocessing dapat mengisinya.
    last_rows = raw.tail(1)
    empty = last_rows.columns[last_rows.isna().any()].tolist()
    if empty:
        msgs.append(
            f"Baris terakhir ({last_date:%d %b %Y}) masih kosong untuk: {', '.join(empty[:4])}"
            f"{'...' if len(empty) > 4 else ''}. Sel kosong akan diisi otomatis dengan harga sebelumnya."
        )
    # Kolom yang belum didaftarkan di ATTR_TO_GROUP dianggap tidak dikenali.
    # Artinya, daftar komoditas yang didukung belum mencakup kolom tersebut.
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
# LONG_WINDOW adalah acuan rata-rata 22 pengamatan (kira-kira satu bulan kerja).
# CHANGE_EPS adalah batas perubahan sangat kecil yang dianggap tetap.
# MAX_RUN membatasi hitungan berapa pengamatan berturut-turut harga tidak berubah.
LONG_WINDOW = 22          # acuan jangka panjang (1 bulan kerja) untuk fitur "jarak ke rata-rata"
CHANGE_EPS = 1e-4         # perubahan < 0,01% dianggap "tidak berubah" (menyerap derau float prediksi rekursif)
MAX_RUN = 22              # batas atas fitur "berapa hari harga tidak berubah"


# Dipakai saat prediksi untuk menghitung berapa perpindahan harga terakhir
# yang nyaris tidak berubah. Penghitungan dimulai dari data terbaru ke belakang.
def _run_length_flat(prices):
    """Berapa hari berturut-turut (sampai hari terakhir `prices`) harga tidak berubah. Dipakai saat PREDIKSI."""
    n = 0
    # Bandingkan harga saat ini dengan harga sebelumnya. Kalau perubahan di bawah
    # CHANGE_EPS, hitungan bertambah; begitu berubah nyata, perulangan berhenti.
    for i in range(len(prices) - 1, 0, -1):
        if abs(prices[i] / prices[i - 1] - 1.0) < CHANGE_EPS:
            n += 1
            if n >= MAX_RUN:
                break
        else:
            break
    return n


# Fungsi utama pembentukan fitur untuk TRAINING. Data harga dalam rupiah diubah
# menjadi informasi pola perubahan harga agar model belajar perilaku perubahan,
# bukan hanya besarnya harga. Nilai pada hari target tidak dipakai sebagai inputnya.
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
    # window_size menyatakan jumlah pengamatan historis yang dilihat model.
    # Nilai ini perlu konsisten antara training dan saat membuat prediksi.
    w = window_size
    s = df_series.astype(float) #mengubah harga menjadi angka desimal
    # Hitung persentase perubahan harga terhadap pengamatan sebelumnya.
    # Contoh: dari 10.000 ke 10.500 menghasilkan 0,05 atau kenaikan 5%.
    ret = s.pct_change()
    df_feat = pd.DataFrame(index=s.index)
    # Target adalah perubahan harga yang hendak diprediksi pada tanggal tersebut.
    # prev_price adalah harga sebelumnya yang nantinya dapat dipakai untuk
    # mengubah prediksi persentase kembali menjadi nominal rupiah.
    df_feat["target"] = ret
    df_feat["prev_price"] = s.shift(1)

    # Buat fitur ret_1 sampai ret_w yang masing-masing mewakili perubahan
    # harga pada 1, 2, hingga w pengamatan sebelum tanggal target.
    for l in range(1, w + 1):
        df_feat[f"ret_{l}"] = ret.shift(l)

    # Data digeser satu langkah agar fitur hanya memakai data yang sudah diketahui
    # sebelum tanggal target. Ini penting untuk mengurangi risiko data leakage.
    shifted = s.shift(1)   # harga hari target tidak ikut dihitung
    # Bandingkan harga terakhir yang diketahui dengan rata-rata jendela pendek.
    # Hasil positif berarti harga berada di atas rata-ratanya, dan sebaliknya.
    df_feat[f"dev_mean_{w}"] = shifted / shifted.rolling(w).mean() - 1.0
    # Selain jendela pendek, bandingkan juga dengan rata-rata 22 pengamatan
    # agar model mendapatkan gambaran pola harga yang lebih panjang.
    df_feat["dev_mean_long"] = shifted / shifted.rolling(LONG_WINDOW).mean() - 1.0
    # std dengan np.std(ddof=1) (bukan rolling().std() bawaan pandas yang berderau pada jendela konstan)
    # Simpangan baku menggambarkan seberapa besar harga berfluktuasi.
    # Dibagi dengan rata-rata supaya ukurannya relatif dan tidak bergantung
    # langsung pada besar nominal masing-masing komoditas.
    roll_std = shifted.rolling(w).apply(lambda x: np.std(x, ddof=1), raw=True)
    df_feat[f"rel_std_{w}"] = roll_std / shifted.rolling(w).mean()

    # Tandai kapan harga mengalami perubahan yang cukup besar.
    # Dari tanda ini dihitung berapa lama harga cenderung tetap, lalu digeser
    # supaya tanggal target tidak ikut memberi informasi ke input model.
    changed = (ret.abs() >= CHANGE_EPS).fillna(True)
    flat = changed.groupby(changed.cumsum()).cumcount().clip(upper=MAX_RUN)   # hari datar sampai & termasuk t
    df_feat["flat_days"] = flat.shift(1)

    # Tambahkan hari dalam minggu dan bulan sebagai fitur kalender.
    # Nilai ini membantu model mengenali kemungkinan pola terkait waktu.
    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month
    # Baris yang belum memiliki riwayat cukup akan mengandung NaN.
    # Baris seperti itu tidak dipakai untuk training karena fiturnya belum lengkap.
    return df_feat.dropna()


# ----------------------------------------------------------------------------
# MODEL & PREDIKSI (dipakai app.py dan predict.py)
# ----------------------------------------------------------------------------
# Nama atribut dijadikan nama dasar file dengan mengganti spasi menjadi garis bawah.
# Contohnya, "Gula Pasir Lokal" menjadi "Gula_Pasir_Lokal".
def _safe(attr):
    return attr.replace(" ", "_")


# Fungsi ini dipanggil setelah model selesai dilatih oleh proses training.
# Model beserta metadata disimpan agar dapat dipakai di aplikasi tanpa perlu
# melakukan pelatihan ulang saat pengguna membuka halaman prediksi.
def save_model_files(model, info, attr):
    """
    Simpan model dalam format BAWAAN library (aman lintas versi & lintas komputer), bukan pickle/joblib:
      - XGBoost  -> models/<atribut>.json
      - LightGBM -> models/<atribut>.txt
      - info (nama model, window, metrik) -> models/<atribut>_info.json
    """
    # Pastikan folder models/ sudah ada. Kalau belum ada, folder dibuat otomatis.
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Susun lokasi file dari nama atribut yang dipilih pengguna.
    base = os.path.join(MODELS_DIR, _safe(attr))
    # Bentuk penyimpanan disesuaikan dengan library: XGBoost memakai .json,
    # sedangkan LightGBM memakai .txt dari booster yang sudah dilatih.
    if info["model_name"] == "XGBoost":
        model.save_model(base + ".json")
    else:
        model.booster_.save_model(base + ".txt")
    # Metadata model disimpan terpisah agar proses prediksi tahu model apa
    # yang harus dibuka dan berapa window historis yang diperlukan.
    with open(base + "_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=float)


# Fungsi ini penting untuk memastikan website hanya MEMUAT model tersimpan.
# Tidak ada proses .fit() di sini. File yang tersimpan di folder models/
# dipakai langsung untuk melakukan prediksi dengan data harga terbaru.
def load_model(attr):
    """
    Muat model hasil train.py untuk satu atribut. Mengembalikan (model, info).
    Model TIDAK dilatih ulang: hanya data terbarunya yang dipakai sebagai masukan prediksi.
    (Format lama .joblib masih dibaca bila ada, tetapi bisa gagal jika versi library berbeda.)
    """
    # Susun lokasi file dari nama atribut yang dipilih pengguna.
    base = os.path.join(MODELS_DIR, _safe(attr))
    try:
        # Utamakan metadata format baru. Metadata menentukan apakah file model
        # akan dibaca sebagai XGBoost atau LightGBM.
        if os.path.exists(base + "_info.json"):
            with open(base + "_info.json", encoding="utf-8") as f:
                info = json.load(f)
            if info["model_name"] == "XGBoost":
                from xgboost import XGBRegressor
                model = XGBRegressor()
                # Memuat bobot dan struktur model XGBoost yang telah disimpan.
                model.load_model(base + ".json")
            else:
                import lightgbm as lgb
                # Memuat model LightGBM langsung dari file teks hasil training.
                model = lgb.Booster(model_file=base + ".txt")
            return model, info
        # Dukungan format lama disediakan untuk kompatibilitas file yang pernah dibuat.
        # Namun, membaca format lama bisa lebih sensitif terhadap perbedaan versi library.
        if os.path.exists(base + "_info.joblib") and os.path.exists(base + ".joblib"):   # format lama
            info = joblib.load(base + "_info.joblib")
            return joblib.load(base + ".joblib"), info
    # Kalau model gagal dimuat, tampilkan pesan yang lebih mudah ditelusuri.
    # Kemungkinan penyebabnya antara lain file rusak atau masalah pada library.
    except Exception as e:  # DLL LightGBM diblokir Windows, versi library beda, atau file rusak
        raise RuntimeError(
            f"Model untuk '{attr}' gagal dimuat: {e}\n"
            f"Jika model LightGBM diblokir Windows, latih ulang dengan: python train.py --models xgboost. "
            f"Jika file rusak, ekstrak ulang folder models/."
        ) from e
    raise FileNotFoundError(f"Model '{attr}' belum ada di folder '{MODELS_DIR}'. Jalankan 'python train.py' dulu.")


# Fungsi ini membuat SATU baris fitur ketika ingin memprediksi tanggal baru.
# Rumus dan nama fiturnya mengikuti build_features() supaya model menerima
# jenis input yang sama dengan saat dilatih.
def make_feature_row(history, date, window_size):
    """
    Susun satu baris fitur untuk memprediksi harga pada `date`, dari riwayat harga `history`
    (urut lama -> baru, hari terakhir = hari sebelum `date`). Isinya HARUS sama dengan build_features().
    """
    # window_size menyatakan jumlah pengamatan historis yang dilihat model.
    # Nilai ini perlu konsisten antara training dan saat membuat prediksi.
    w = window_size
    # Ubah riwayat harga menjadi array angka untuk memudahkan pengambilan
    # beberapa data paling akhir sesuai panjang jendela.
    h = np.asarray(history, dtype=float)
    # Susun perubahan harga untuk lag 1 sampai lag w dari riwayat terbaru.
    # Ini menjadi masukan dasar yang menggambarkan arah perubahan sebelumnya.
    feat = {f"ret_{l}": h[-l] / h[-l - 1] - 1.0 for l in range(1, w + 1)}
    # Ambil w harga terbaru untuk menghitung jarak ke rata-rata dan volatilitas.
    win = h[-w:]
    feat[f"dev_mean_{w}"] = h[-1] / win.mean() - 1.0
    feat["dev_mean_long"] = h[-1] / h[-LONG_WINDOW:].mean() - 1.0
    feat[f"rel_std_{w}"] = np.std(win, ddof=1) / win.mean()
    # Tambahkan jumlah pengamatan terakhir yang cenderung tidak berubah,
    # serta fitur kalender dari tanggal yang akan diprediksi.
    feat["flat_days"] = _run_length_flat(h)
    feat["dayofweek"] = date.dayofweek
    feat["month"] = date.month
    return feat


# Prediksi rekursif berarti hasil prediksi satu hari menjadi bagian dari riwayat
# untuk memprediksi hari berikutnya. Cara ini memungkinkan prediksi beberapa
# hari ke depan meski harga aktual untuk hari-hari mendatang belum tersedia.
# Perlu diingat: kesalahan prediksi bisa ikut terbawa ke langkah selanjutnya.
def predict_future_recursive(model, df_series, steps_ahead, window_size, future_dates=None):
    """
    Prediksi multi-step rekursif: setiap prediksi dipakai sebagai riwayat untuk langkah berikutnya.
    Default tanggalnya = hari kerja setelah data terakhir; `future_dates` dipakai saat evaluasi
    agar tanggal (dan fitur kalender) sama persis dengan data aktual.
    """
    # Awali proses dengan rangkaian harga historis yang tersedia.
    history = list(df_series.values)
    # Jika tanggal tidak diberikan khusus, buat daftar hari kerja berikutnya
    # (Senin sampai Jumat) setelah tanggal data terakhir.
    if future_dates is None:
        future_dates = pd.date_range(
            start=df_series.index[-1] + pd.Timedelta(days=1), periods=steps_ahead * 2, freq="B"
        )[:steps_ahead]
    # Simpan setiap nominal harga hasil prediksi ke dalam list.
    preds = []
    # Untuk setiap tanggal, buat fitur berdasarkan riwayat saat itu,
    # lalu minta model menghasilkan prediksi perubahan harga.
    for f_date in future_dates[:steps_ahead]:
        feat = make_feature_row(history, f_date, window_size)
        # Model memberikan perubahan dalam bentuk pecahan: misalnya 0,02 = naik 2%.
        predicted_return = float(model.predict(pd.DataFrame([feat]))[0])   # persentase perubahan (pecahan)
        # Ubah persentase perubahan menjadi harga rupiah berdasarkan harga terakhir.
        # Misalnya Rp10.000 x (1 + 0,02) menghasilkan prediksi Rp10.200.
        p = float(history[-1] * (1.0 + predicted_return))
        preds.append(p)
        # Hasil hari ini ditambahkan ke riwayat sementara agar bisa dipakai
        # pada prediksi hari berikutnya (inti dari metode rekursif).
        history.append(p)
    # Kembalikan prediksi sebagai deret berindeks tanggal supaya mudah
    # ditampilkan pada tabel, grafik, maupun keluaran terminal.
    return pd.Series(preds, index=list(future_dates[:steps_ahead]))


# Fungsi ini membuat rentang berdasarkan pola perubahan harga yang pernah terjadi.
# Rentang tersebut bukan kepastian, bukan interval kepercayaan statistik otomatis,
# dan bukan hasil model prediksi utama. Ini hanya gambaran variasi historis.
def historical_range(series, steps, lo=0.10, hi=0.90, min_samples=60):
    """
    Rentang harga yang WAJAR (bukan ramalan titik) untuk 1..steps hari kerja ke depan, dari sebaran historis
    perubahan harga k hari kerja pada atribut itu: harga terakhir * (1 + kuantil 10% / 90% perubahan k hari).
    Hasilnya DataFrame (index 1..steps) berkolom 'lo' dan 'hi'; NaN bila riwayat terlalu pendek.
    Pada uji di data latih -> data uji, rentang 10-90% ini memuat harga aktual pada kira-kira 90% kasus (konservatif).
    """
    # Pastikan hanya data harga numerik yang tersedia yang ikut dihitung.
    s = pd.Series(series, dtype=float).dropna()
    last = float(s.iloc[-1])
    rows = []
    # Hitung perubahan historis untuk jarak 1 hari, 2 hari, dan seterusnya
    # sampai sebanyak langkah yang ingin ditampilkan.
    for k in range(1, steps + 1):
        ch = (s.shift(-k) / s - 1.0).dropna()
        # Jika contoh perubahan historis masih terlalu sedikit,
        # batas rentang tidak dipaksakan dan diisi NaN.
        if len(ch) < min_samples:
            rows.append((np.nan, np.nan))
        else:
            # Cari batas bawah dan atas dari kuantil perubahan historis,
            # lalu terapkan ke harga terakhir untuk membentuk kisaran rupiah.
            q_lo, q_hi = np.quantile(ch.values, [lo, hi])
            rows.append((last * (1.0 + q_lo), last * (1.0 + q_hi)))
    return pd.DataFrame(rows, index=range(1, steps + 1), columns=["lo", "hi"])


# Fungsi ini menghitung jumlah hari kerja menuju tanggal target.
# Sabtu dan Minggu tidak dihitung, sehingga jumlah langkah sesuai dengan
# kalender hari kerja yang digunakan pada prediksi. Hari libur nasional belum dikecualikan.
def steps_to_date(last_date, target_date):
    """Jumlah hari kerja dari hari setelah `last_date` sampai `target_date` (akhir pekan tidak dihitung)."""
    # Mulai perhitungan sesudah tanggal data terakhir dan sertakan tanggal
    # target melalui batas akhir yang ditambah satu hari.
    start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).date()
    end = (pd.Timestamp(target_date) + pd.Timedelta(days=1)).date()
    return int(np.busday_count(start, end))


# Fungsi pengecekan ini memastikan fitur saat training sama dengan fitur
# saat prediksi untuk beberapa tanggal terakhir. Jika tidak sama, hasil model
# bisa meleset karena menerima format atau nilai fitur yang berbeda.
def check_feature_parity(series, window_size, n_checks=5, rtol=1e-9):
    """
    Uji kebenaran: fitur yang dibuat saat PREDIKSI (make_feature_row) harus identik dengan fitur
    saat TRAINING (build_features) pada tanggal yang sama. Bila beda, prediksi akan bergeser diam-diam.
    """
    # Siapkan fitur versi training sebagai acuan perbandingan.
    df_feat = build_features(series, window_size)
    # Ambil tanggal-tanggal terakhir yang lengkap, lalu buat fitur versi prediksi
    # hanya dari data yang tersedia sebelum masing-masing tanggal tersebut.
    for date in df_feat.index[-n_checks:]:
        pos = series.index.get_loc(date)
        row = make_feature_row(series.iloc[:pos].values, date, window_size)
        # Kolom target dan harga sebelumnya dibuang dari perbandingan karena
        # keduanya bukan masukan model. Yang dibandingkan hanya fitur input.
        expected = df_feat.loc[date].drop(["target", "prev_price"])
        got = np.array([row[k] for k in expected.index], dtype=float)
        # Gunakan toleransi numerik kecil agar selisih pembulatan komputer tidak
        # dianggap sebagai perbedaan fitur yang bermakna.
        if not np.allclose(got, expected.values.astype(float), rtol=rtol, atol=1e-9):
            raise ValueError(f"Fitur prediksi != fitur training pada {date:%Y-%m-%d} (window {window_size}).")
    # Jika semua perbandingan lolos, kembalikan True sebagai tanda cocok.
    return True