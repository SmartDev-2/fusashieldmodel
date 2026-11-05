"""
=============================================================================
FUSASHIELD CLOUD CONTROLLER (100% DATA RIIL: IOT & CITRA LENGKAP)
Sistem Mitigasi Presisi Layu Fusarium Berbasis IoT dan Agen Hayati Otomatis
=============================================================================
Sistem HANYA bekerja jika KEDUA syarat terpenuhi secara lengkap:
  1. Data Telemetri IoT dari ESP32 LENGKAP & AKTIF (Suhu, Tanah, RH Udara).
  2. Citra Daun Cabai TERSEDIA & VALID di Supabase Storage.
TANPA DATA DEMO / NILAI SIMULASI.
=============================================================================
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import io
import re
import textwrap
import time
import json
import requests
from datetime import datetime, timezone
from PIL import Image, ImageOps
import numpy as np
import keras
from supabase import create_client, Client

# ==========================================================
# 1. KONFIGURASI KREDENSIAL (DARI .ENV)
# ==========================================================
def _muat_env(path=".env"):
    """Muat variabel environment dari file .env secara otomatis."""
    lokasi = os.path.join(os.path.dirname(os.path.abspath(__file__)), path) if "__file__" in globals() else path
    if os.path.exists(lokasi):
        try:
            with open(lokasi, "r", encoding="utf-8") as f:
                for l in f:
                    b = l.strip()
                    if b and not b.startswith("#") and "=" in b:
                        k, v = b.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception as e:
            print(f"[WARN] Gagal membaca {path}: {e}")

_muat_env()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "fusashield-image")
SUPABASE_FOLDER = os.environ.get("SUPABASE_FOLDER", "cabai_01")

FIREBASE_DB_URL = os.environ.get("FIREBASE_DB_URL", "")
FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY", "")
NODE_PATH = os.environ.get("NODE_PATH", "/monitoring/cabai_01")

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "fusashield_mobilenetv2.keras")
LABELS_PATH = os.path.join(BASE_DIR, "class_names.json")

EXT_GAMBAR = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Cooldown Protection: Penguncian pompa otomatis selama 12 jam (43.200 detik)
# setelah penyemprotan untuk mencegah kejenuhan air tanah (over-watering).
COOLDOWN_STANDAR_DETIK = 12 * 3600  # 12 jam (43.200 detik)
COOLDOWN_PER_TINGKAT = {
    "KRITIS":    COOLDOWN_STANDAR_DETIK,   # dosis >= 70 ml (12 jam / 43.200 detik)
    "WASPADA":   COOLDOWN_STANDAR_DETIK,   # 35 ml <= dosis < 70 ml (12 jam / 43.200 detik)
    "PREVENTIF": COOLDOWN_STANDAR_DETIK,   # 10 ml <= dosis < 35 ml (12 jam / 43.200 detik)
}
SKALA_COOLDOWN = 1.0

# Foto lebih tua dari 24 jam dianggap tidak representatif untuk kondisi daun saat ini
FOTO_MAKS_UMUR_JAM = 24.0

# ESP32 mengirim data tiap 10 detik. Jika uptime_ms tidak berubah > 45 detik,
# ESP32 dianggap offline / mati.
IOT_TIMEOUT_DETIK = 45

# ---------------- LLM (Groq) : Narasi rekomendasi agronomi XAI ----------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.environ.get("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_TIMEOUT_DETIK = 12
LLM_INTERVAL_MIN_DETIK = 600   # jarak minimal antar panggilan jika hanya kondisi yang berubah

# Nama kunci sensor di Firebase (alias dicoba berurutan).
KUNCI_SENSOR = {
    "temperature":   ["temperature", "suhu", "temp", "soil_temperature"],
    "soil_moisture": ["soil_moisture", "kelembapan", "moisture", "soil", "humidity_soil"],
    "humidity":      ["humidity", "kelembapan_udara", "air_humidity", "rh"],
}

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Patch kompatibilitas Keras initializers (input_axes/output_axes)
try:
    import keras.initializers
    for _name in dir(keras.initializers):
        _obj = getattr(keras.initializers, _name)
        if isinstance(_obj, type) and issubclass(_obj, keras.initializers.Initializer):
            _orig = _obj.__init__
            def _patch(orig_fn):
                def _wrapped(self, *args, **kwargs):
                    kwargs.pop("input_axes", None)
                    kwargs.pop("output_axes", None)
                    return orig_fn(self, *args, **kwargs)
                return _wrapped
            _obj.__init__ = _patch(_orig)
except Exception:
    pass

try:
    from keras.src.initializers import random_initializers
    for _name in dir(random_initializers):
        _obj = getattr(random_initializers, _name)
        if isinstance(_obj, type) and hasattr(_obj, "__init__"):
            _orig = _obj.__init__
            def _patch_rand(orig_fn):
                def _wrapped(self, *args, **kwargs):
                    kwargs.pop("input_axes", None)
                    kwargs.pop("output_axes", None)
                    return orig_fn(self, *args, **kwargs)
                return _wrapped
            _obj.__init__ = _patch_rand(_orig)
except Exception:
    pass

try:
    import keras.layers
    _orig_bn = keras.layers.BatchNormalization.__init__
    def _patched_bn(self, *args, **kwargs):
        kwargs.pop("renorm", None)
        kwargs.pop("renorm_clipping", None)
        kwargs.pop("renorm_momentum", None)
        return _orig_bn(self, *args, **kwargs)
    keras.layers.BatchNormalization.__init__ = _patched_bn
except Exception:
    pass

print("[INFO] Memuat model CNN MobileNetV2 (.keras)...")
model = keras.models.load_model(MODEL_PATH, compile=False)
with open(LABELS_PATH, "r") as f:
    class_names = json.load(f)
print(f"[INFO] Model MobileNetV2 siap digunakan! ({len(class_names)} kelas: {class_names})")

# Cache internal pipeline
_foto_terakhir_diproses = None
_cache_cnn = None              # hasil CNN foto terakhir: dict(label, conf, skor, path)
_waktu_pompa_terakhir = 0      # epoch detik saat pompa terakhir menyala
_ack_terakhir = None           # id ack terakhir dari ESP32
_iot_uptime_terakhir = None    # uptime_ms ESP32 terakhir yang terlihat
_iot_waktu_berubah = 0.0       # kapan uptime_ms terakhir berubah (epoch)
_iot_pernah_berubah = False    # bukti ESP32 aktif
_llm_cache = {"key": None, "teks": "", "sumber": "", "ts": 0.0}
_fuzzy_detail = {}


# ==========================================================
# 2. SELEKSI & AKUISISI CITRA DARI SUPABASE
# ==========================================================
def tingkat_dosis(dosis_ml):
    if dosis_ml >= 70.0:
        return "KRITIS"
    if dosis_ml >= 35.0:
        return "WASPADA"
    if dosis_ml >= 10.0:
        return "PREVENTIF"
    return "AMAN"


def waktu_foto_epoch(info):
    """Waktu unggah foto (epoch detik): dari created_at, atau dari angka timestamp di nama file."""
    ca = info.get("created_at") or info.get("updated_at")
    if ca:
        try:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass
    m = re.match(r"(\d{10,13})", info.get("name", ""))
    if m:
        v = int(m.group(1))
        return v / 1000.0 if v > 1e11 else float(v)
    return None


def pilih_file_terbaru(files):
    """Pilih file gambar terbaru dari hasil list() Supabase."""
    kandidat = [
        f for f in (files or [])
        if f.get("name") and f["name"].lower().endswith(EXT_GAMBAR)
    ]
    if not kandidat:
        return None

    def kunci(f):
        waktu = f.get("created_at") or f.get("updated_at") or ""
        m = re.match(r"(\d{10,13})", f["name"])
        prefix = int(m.group(1)) if m else 0
        return (str(waktu), prefix)

    return max(kandidat, key=kunci)


def cari_file_terbaru():
    """Return (path_lengkap, info_file) atau (None, None)."""
    opsi = {"limit": 100, "sortBy": {"column": "created_at", "order": "desc"}}
    try:
        files = supabase.storage.from_(BUCKET_NAME).list(SUPABASE_FOLDER, opsi)
    except Exception as e:
        print(f"[ERROR] Gagal list folder '{SUPABASE_FOLDER}': {e}")
        return None, None

    info = pilih_file_terbaru(files)
    if info is not None:
        return f"{SUPABASE_FOLDER}/{info['name']}", info

    return None, None


def download_gambar(path_lengkap):
    """Download citra dari Supabase dan lakukan koreksi rotasi EXIF."""
    data = None
    try:
        data = supabase.storage.from_(BUCKET_NAME).download(path_lengkap)
    except Exception as e:
        try:
            url = supabase.storage.from_(BUCKET_NAME).get_public_url(path_lengkap)
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.content
        except Exception as e2:
            print(f"[ERROR] Download gambar gagal: {e2}")
            return None

    try:
        img = Image.open(io.BytesIO(data))
        return ImageOps.exif_transpose(img)
    except Exception as e:
        print(f"[ERROR] File bukan gambar valid: {e}")
        return None


# ==========================================================
# 3. KOMUNIKASI FIREBASE
# ==========================================================
def ambil_data_iot_firebase():
    url = f"{FIREBASE_DB_URL}{NODE_PATH}.json?auth={FIREBASE_API_KEY}"
    try:
        res = requests.get(url, timeout=8)
        res.raise_for_status()
        return res.json() or {}
    except Exception as e:
        print(f"[ERROR] Gagal membaca Firebase: {e}")
        return {}


# ==========================================================
# 4. ANALISIS CNN & MESIN FUZZY LOGIC
# ==========================================================
def analisis_citra_pil(pil_img):
    img_resized = pil_img.convert("RGB").resize((224, 224))
    img_array = keras.utils.img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)

    probs = model.predict(img_batch, verbose=0)[0]
    idx_max = int(np.argmax(probs))
    label = class_names[idx_max]
    conf = float(probs[idx_max] * 100.0)

    idx_sehat = class_names.index("Healthy Leaf") if "Healthy Leaf" in class_names else 3
    idx_klorosis = class_names.index("Nutrition Deficiency") if "Nutrition Deficiency" in class_names else 4

    p_sehat = float(probs[idx_sehat])
    p_klorosis = float(probs[idx_klorosis])

    skor_gejala = ((1.0 - p_sehat) * 60.0) + (p_klorosis * 40.0)
    skor_gejala = float(np.clip(skor_gejala, 0.0, 100.0))

    return label, round(conf, 2), round(skor_gejala, 2)


def trapesium(x, a, b, c, d):
    """Fungsi keanggotaan trapesium [a, b, c, d] dengan prioritas plateau."""
    if b <= x <= c:
        return 1.0
    if x <= a or x >= d:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def hitung_fuzzy_dan_workflow(hst, rh_udara, soil_moisture, temperature, skor_cnn):
    mu_rh = {
        "Rendah": trapesium(rh_udara, 0.0, 0.0, 50.0, 60.0),
        "Sedang": trapesium(rh_udara, 50.0, 60.0, 75.0, 85.0),
        "Tinggi": trapesium(rh_udara, 75.0, 85.0, 100.0, 100.0)
    }
    mu_lembap = {
        "Kering": trapesium(soil_moisture, 0.0, 0.0, 35.0, 50.0),
        "Lembap": trapesium(soil_moisture, 35.0, 50.0, 70.0, 80.0),
        "Basah":  trapesium(soil_moisture, 70.0, 80.0, 100.0, 100.0)
    }
    mu_suhu = {
        "Dingin":  trapesium(temperature, 0.0, 0.0, 22.0, 26.0),
        "Optimal": trapesium(temperature, 22.0, 26.0, 30.0, 33.0),
        "Panas":   trapesium(temperature, 30.0, 33.0, 50.0, 50.0)
    }
    mu_citra = {
        "Ringan": trapesium(skor_cnn, 0.0, 0.0, 20.0, 40.0),
        "Sedang": trapesium(skor_cnn, 20.0, 40.0, 60.0, 75.0),
        "Berat":  trapesium(skor_cnn, 60.0, 75.0, 100.0, 100.0)
    }

    r_kritis = max(
        min(mu_rh["Tinggi"], mu_lembap["Basah"], mu_suhu["Optimal"], mu_citra["Berat"]),
        min(mu_rh["Tinggi"], mu_lembap["Basah"], mu_suhu["Optimal"], mu_citra["Sedang"]),
        min(mu_rh["Tinggi"], mu_citra["Berat"]),
        min(mu_lembap["Basah"], mu_citra["Berat"])
    )
    r_waspada = max(
        min(mu_rh["Tinggi"], mu_lembap["Lembap"], mu_suhu["Optimal"], mu_citra["Sedang"]),
        min(mu_rh["Sedang"], mu_lembap["Basah"], mu_suhu["Optimal"], mu_citra["Sedang"]),
        min(mu_rh["Sedang"], mu_lembap["Lembap"], mu_citra["Berat"]),
        min(max(mu_rh["Tinggi"], mu_lembap["Basah"]), mu_citra["Sedang"])
    )
    r_preventif = max(
        min(mu_rh["Sedang"], mu_lembap["Lembap"], mu_suhu["Optimal"], mu_citra["Sedang"]),
        min(mu_rh["Tinggi"], mu_lembap["Basah"], mu_citra["Ringan"]),
        min(mu_rh["Tinggi"], mu_lembap["Lembap"], mu_citra["Ringan"]),
        min(mu_rh["Sedang"], mu_lembap["Basah"], mu_citra["Ringan"])
    )
    r_aman = max(
        min(mu_rh["Sedang"], mu_lembap["Lembap"], mu_suhu["Optimal"], mu_citra["Ringan"]),
        min(mu_rh["Sedang"], mu_lembap["Kering"], mu_citra["Ringan"]),
        min(mu_rh["Rendah"], mu_lembap["Lembap"], mu_citra["Ringan"])
    )

    global _fuzzy_detail
    _fuzzy_detail = {
        "Kelembapan udara": mu_rh, "Kelembapan tanah": mu_lembap,
        "Suhu": mu_suhu, "Gejala citra": mu_citra,
        "aturan": {"Aman": r_aman, "Preventif": r_preventif,
                   "Waspada": r_waspada, "Kritis": r_kritis},
    }

    num = (r_aman * 0.0) + (r_preventif * 20.0) + (r_waspada * 50.0) + (r_kritis * 100.0)
    den = r_aman + r_preventif + r_waspada + r_kritis
    dosis_fuzzy = num / den if den > 0 else 0.0

    if hst == 0:
        fase = "Fase 1: Awal Tanam (0 HST)"
        dosis_akhir = 30.0 if (40.0 <= soil_moisture <= 80.0) else 20.0
        status = "INOKULASI AWAL (Starter Dose)"
    elif 1 <= hst <= 40:
        fase = f"Fase 2: Jeda Vegetatif ({hst} HST)"
        dosis_akhir = dosis_fuzzy if dosis_fuzzy >= 10.0 else 0.0
        status = "PREVENTIF DINI (Early Warning Tanah)" if dosis_akhir > 0 else "AMAN (Standby)"
    elif 41 <= hst <= 44:
        fase = f"Fase 3: Titik Kritis ({hst} HST - Trigger P1)"
        dosis_akhir = dosis_fuzzy if dosis_fuzzy >= 10.0 else 0.0
        status = "TRIGGER KURATIF 41 HST (100cc/15L)" if dosis_akhir > 0 else "AMAN"
    else:
        minggu = min(4, ((hst - 45) // 7) + 1)
        fase = f"Fase 4: Evaluasi Minggu ke-{minggu} ({hst} HST)"
        dosis_akhir = dosis_fuzzy if dosis_fuzzy >= 10.0 else 0.0
        if dosis_akhir >= 70.0:
            status = "KRITIS / BOOSTER (Dosis Intensif)"
        elif dosis_akhir >= 35.0:
            status = "WASPADA (Dosis Kuratif Sedang)"
        elif dosis_akhir >= 10.0:
            status = "PREVENTIF PEMULIHAN (Dosis Rendah)"
        else:
            status = "AMAN / PULIH (Pompa OFF)"

    durasi_detik = round(dosis_akhir / 5.0, 2)
    pump_cmd = "ON" if dosis_akhir > 0 else "OFF"

    return round(dosis_akhir, 2), durasi_detik, pump_cmd, status, fase


# ==========================================================
# 5. VALIDASI KELENGKAPAN IOT (100% DATA RIIL, TANPA DEMO)
# ==========================================================
def baca_sensor(data_iot):
    """Ambil suhu, kelembapan tanah, dan RH udara dari data Firebase."""
    hasil, hilang = {}, []
    for nama, alias in KUNCI_SENSOR.items():
        nilai = None
        for k in alias:
            if k in data_iot and data_iot[k] is not None:
                try:
                    nilai = float(data_iot[k])
                    break
                except (TypeError, ValueError):
                    pass
        if nilai is None:
            hilang.append(nama)
        hasil[nama] = nilai
    return hasil, hilang


def perbarui_cache_cnn():
    """Cek foto terbaru di Supabase. Return True jika citra tersedia & valid."""
    global _cache_cnn, _foto_terakhir_diproses

    path_foto, info = cari_file_terbaru()
    if path_foto is None:
        return _cache_cnn is not None

    id_foto = f"{path_foto}|{info.get('created_at')}|{info.get('updated_at')}"
    if id_foto == _foto_terakhir_diproses:
        return True  # foto sama, gunakan cache

    print(f"       -> Foto daun baru: '{path_foto}' (waktu: {info.get('created_at')})")
    pil_img = download_gambar(path_foto)
    if pil_img is None:
        return _cache_cnn is not None

    label, conf, skor = analisis_citra_pil(pil_img)
    _cache_cnn = {"label": label, "conf": conf, "skor": skor, "path": path_foto,
                  "ts": waktu_foto_epoch(info), "baru": True}
    _foto_terakhir_diproses = id_foto
    print(f"       -> CNN MobileNetV2: {label} ({conf}%) -> Skor Gejala: {skor}%")
    return True


def evaluasi_citra():
    """Memeriksa apakah citra daun di Supabase tersedia DAN berumur <= 24 jam terakhir."""
    ada = perbarui_cache_cnn()
    if not ada or _cache_cnn is None:
        return {
            "valid": False,
            "umur_jam": None,
            "foto": None,
            "alasan": "Belum ada file citra di bucket fusashield-image/cabai_01"
        }

    sekarang = time.time()
    ts_foto = _cache_cnn.get("ts")
    umur_jam = ((sekarang - ts_foto) / 3600.0) if ts_foto else None

    # Syarat mutlak: Foto harus berasal dari 24 jam terakhir
    if umur_jam is not None and umur_jam > FOTO_MAKS_UMUR_JAM:
        return {
            "valid": False,
            "umur_jam": round(umur_jam, 2),
            "foto": _cache_cnn["path"],
            "alasan": f"Citra '{_cache_cnn['path']}' KADALUARSA (umur {umur_jam:.1f} jam > batas {FOTO_MAKS_UMUR_JAM:.0f} jam). Harap unggah foto daun cabai terbaru!"
        }

    return {
        "valid": True,
        "umur_jam": None if umur_jam is None else round(umur_jam, 2),
        "foto": _cache_cnn["path"],
        "alasan": f"Citra valid dari 24 jam terakhir (umur {umur_jam:.1f} jam)" if umur_jam is not None else "Citra valid"
    }


def evaluasi_iot(data_iot):
    """Mengevaluasi kelengkapan dan keaktifan telemetri IoT ESP32.
    TIDAK MENGGUNAKAN NILAI DEMO/SIMULASI. HANYA MENERIMA DATA RIIL.
    """
    global _iot_uptime_terakhir, _iot_waktu_berubah, _iot_pernah_berubah
    sekarang = time.time()

    if not data_iot:
        return {
            "valid": False,
            "status": "TIDAK_ADA_DATA",
            "nilai": None,
            "alasan": ["Firebase node /monitoring/cabai_01 masih kosong"]
        }

    alasan = []
    sensor, hilang = baca_sensor(data_iot)

    # Sensor tanah harus valid (jika sensor melaporkan probe tidak valid)
    if data_iot.get("soil_valid") is False:
        if "soil_moisture" not in hilang:
            hilang.append("soil_moisture")
        alasan.append("sensor tanah ditandai TIDAK VALID oleh ESP32 (cek probe/kabel)")

    # Periksa uptime ESP32 untuk membuktikan ESP32 hidup dan aktif
    uptime = data_iot.get("uptime_ms")
    hidup = "AKTIF"
    if uptime is not None:
        if _iot_uptime_terakhir is None:
            _iot_uptime_terakhir, _iot_waktu_berubah, _iot_pernah_berubah = uptime, sekarang, False
        elif uptime != _iot_uptime_terakhir:
            _iot_uptime_terakhir, _iot_waktu_berubah, _iot_pernah_berubah = uptime, sekarang, True
        
        diam = sekarang - _iot_waktu_berubah
        if diam > IOT_TIMEOUT_DETIK:
            hidup = "TIDAK_AKTIF"
        elif not _iot_pernah_berubah:
            hidup = "MENUNGGU"
    else:
        hidup = "TIDAK_AKTIF"
        alasan.append("field uptime_ms belum dikirim oleh ESP32")

    if hidup == "TIDAK_AKTIF":
        alasan.append(f"ESP32 tidak mengirim data baru > {IOT_TIMEOUT_DETIK} detik (ESP32 offline)")
        return {
            "valid": False,
            "status": "TIDAK_AKTIF",
            "nilai": None,
            "alasan": alasan
        }

    if hilang:
        alasan.append(f"Sensor belum lengkap: {hilang}. Kunci di Firebase: {sorted(data_iot.keys())}")
        return {
            "valid": False,
            "status": "SENSOR_TIDAK_LENGKAP",
            "nilai": None,
            "alasan": alasan
        }

    if hidup == "MENUNGGU":
        alasan.append("menunggu uptime_ms ESP32 berubah (memastikan koneksi aktif)")
        return {
            "valid": False,
            "status": "MENUNGGU",
            "nilai": None,
            "alasan": alasan
        }

    # Data IoT 100% lengkap dan aktif
    return {
        "valid": True,
        "status": "AKTIF",
        "nilai": {
            "temperature": float(sensor["temperature"]),
            "soil_moisture": float(sensor["soil_moisture"]),
            "humidity": float(sensor["humidity"])
        },
        "alasan": ["Seluruh sensor IoT (suhu, kelembapan tanah, RH udara) aktif dan terbaca normal"]
    }


# ==========================================================
# 6. REKOMENDASI AGRONOMI GROQ LLM (XAI)
# ==========================================================
SARAN_PER_KELAS = {
    "Healthy Leaf": "Daun terlihat sehat. Lanjutkan pemantauan rutin dan jaga drainase agar tanah tidak tergenang.",
    "Nutrition Deficiency": "Gejala menguning/klorosis mengarah ke kekurangan hara atau gangguan serapan akar. "
                            "Periksa jadwal pemupukan (N, Mg, Fe), pastikan drainase baik, dan amati apakah gejala menyebar dari daun bawah.",
    "Bacterial Spot": "Buang daun bergejala berat, hindari percikan air ke daun, dan sanitasi alat. Hindari menyiram pada sore hari.",
    "Cercospora Leaf Spot": "Kurangi kelembapan di sekitar tajuk (jarak tanam, pemangkasan), buang daun terinfeksi, dan hindari penyiraman dari atas.",
    "Curl Virus": "Kendalikan serangga vektor (kutu kebul/thrips), cabut tanaman yang terinfeksi berat, dan pisahkan dari tanaman sehat.",
    "White spot": "Buang daun bergejala, jaga sirkulasi udara, dan pantau penyebaran ke tanaman sekitar.",
}


def rekomendasi_template(k):
    """Cadangan tanpa LLM: teks tetap dari kelas CNN + keputusan fuzzy."""
    saran = SARAN_PER_KELAS.get(k["cnn_label"], "Pantau kondisi tanaman dan lakukan pemeriksaan langsung di lahan.")
    return (f"Indikasi: {k['cnn_label']} (skor gejala {k['skor_gejala']}%, tingkat {k['tingkat']}). {saran} "
            f"Pencegahan: hindari genangan, jaga aerasi tanah, dan pantau penyebaran gejala visual di lahan.")


PROMPT_SISTEM = (
    "Anda pakar patologi tanaman cabai yang memberi SARAN AGRONOMI TAMBAHAN dalam Bahasa Indonesia ilmiah. "
    "Sistem FusaShield memakai CNN MobileNetV2 (mengenali gejala daun, bukan diagnosis Fusarium pasti) "
    "dan fuzzy logic Mamdani (singleton defuzzification) untuk kendali aktuator dosis Trichoderma harzianum cair. "
    "ATURAN KETAT: "
    "(1) JANGAN menghitung, mengubah, atau mengusulkan angka dosis/durasi pompa; dosis dari fuzzy sudah final, "
    "dan Anda tidak boleh menyebut angka ml/cc selain yang diberikan. "
    "(2) Nyatakan temuan sebagai INDIKASI, bukan konfirmasi mutlak patogen Fusarium di laboratorium. "
    "(3) Analisis mikroklimat tanah dan udara secara ilmiah berdasarkan data riil telemetri sensor. "
    "(4) Format Output: Maksimal 3 kalimat padat dalam Bahasa Indonesia ilmiah tanpa bullet points yang memuat: "
    "(1) korelasi mikroklimat & gejala visual CNN terhadap risiko patogen Fusarium oxysporum, "
    "(2) evaluasi kesesuaian dosis agen hayati Trichoderma, dan "
    "(3) satu tindakan perawatan fisik atau budidaya spesifik di lahan."
)


def _dosis_dalam_teks_valid(teks, dosis_ml):
    """Tolak teks LLM yang menyebut angka ml/cc selain dosis dari fuzzy."""
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*(?:ml|mL|ML|cc)\b", teks):
        v = float(m.group(1).replace(",", "."))
        if abs(v - dosis_ml) > 0.5:
            return False
    return True


def panggil_llm(k):
    """Panggil Groq LLM dengan auto-fallback tier 1 (70b) -> tier 2 (8b) -> template offline."""
    if not GROQ_API_KEY:
        return rekomendasi_template(k), "template (GROQ_API_KEY belum diset di .env)"

    model_kandidat = [GROQ_MODEL]
    if GROQ_MODEL_FALLBACK and GROQ_MODEL_FALLBACK != GROQ_MODEL:
        model_kandidat.append(GROQ_MODEL_FALLBACK)

    riwayat_error = []
    for model_name in model_kandidat:
        payload = {
            "model": model_name,
            "temperature": 0.3,
            "max_tokens": 400,
            "messages": [
                {"role": "system", "content": PROMPT_SISTEM},
                {"role": "user", "content": json.dumps(k, ensure_ascii=False)},
            ],
        }
        try:
            r = requests.post(
                GROQ_URL,
                json=payload,
                timeout=LLM_TIMEOUT_DETIK,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
            )
            r.raise_for_status()
            teks = (r.json()["choices"][0]["message"]["content"] or "").strip()
            if not teks:
                riwayat_error.append(f"{model_name}: teks kosong")
                continue
            if not _dosis_dalam_teks_valid(teks, k["dosis_fuzzy_ml"]):
                riwayat_error.append(f"{model_name}: dosis berbeda")
                continue
            return teks[:1200], f"groq:{model_name}"
        except Exception as e:
            riwayat_error.append(f"{model_name}: {type(e).__name__}")

    alasan = ", ".join(riwayat_error) if riwayat_error else "tidak diketahui"
    return rekomendasi_template(k), f"template (fallback offline: {alasan})"


def rekomendasi_untuk(k, foto_baru):
    global _llm_cache
    kunci = (k["cnn_label"], k["tingkat"], k["dosis_fuzzy_ml"], k["foto"])
    sekarang = time.time()
    c = _llm_cache
    perlu = (c["key"] is None or foto_baru or
             (kunci != c["key"] and sekarang - c["ts"] >= LLM_INTERVAL_MIN_DETIK))
    if perlu:
        teks, sumber = panggil_llm(k)
        _llm_cache = {"key": kunci, "teks": teks, "sumber": sumber, "ts": sekarang}
    return _llm_cache["teks"], _llm_cache["sumber"]


LABEL_STATUS_IOT = {
    "AKTIF": "AKTIF - Telemetri nyata dari sensor ESP32",
    "MENUNGGU": "MENUNGGU konfirmasi ESP32 (uptime belum berubah)",
    "SENSOR_TIDAK_LENGKAP": "TIDAK LENGKAP - Menunggu seluruh sensor terhubung",
    "TIDAK_AKTIF": "OFFLINE - ESP32 tidak mengirim data / mati",
    "TIDAK_ADA_DATA": "KOSONG - Belum ada data di Firebase",
}


def cetak_ringkasan(c):
    g = "+" + "=" * 70
    print(g)
    print(f"| HASIL ANALISIS FUSASHIELD (100% DATA RIIL)   {c['waktu']}")
    print(g)
    umur = "umur tidak diketahui" if c["umur_jam"] is None else f"umur {c['umur_jam']:.1f} jam"
    print(f"| Foto        : {c['foto']}  ({umur})")
    print(f"| CNN         : {c['label']} ({c['conf']}%)  ->  Skor Gejala {c['skor']}%")
    print(f"| Status IoT  : {LABEL_STATUS_IOT[c['status_iot']]}")
    print(f"|   Suhu              : {c['nilai']['temperature']:>6.1f} C   [IoT]")
    print(f"|   Kelembapan tanah  : {c['nilai']['soil_moisture']:>6.1f} %   [IoT]")
    print(f"|   Kelembapan udara  : {c['nilai']['humidity']:>6.1f} %   [IoT]")
    print(f"|   {'HST':<18}: {c['hst']:>6}    [{c['sumber_hst']}]  {c['fase']}")
    print("| --- Perhitungan fuzzy " + "-" * 47)
    for k, mu in c["detail"].items():
        if k == "aturan":
            continue
        print(f"|   {k:<17}: " + " | ".join(f"{nm} {v:.2f}" for nm, v in mu.items()))
    print("|   Kekuatan aturan : " + " | ".join(f"{nm} {v:.2f}" for nm, v in c["detail"]["aturan"].items()))
    print(f"|   Dosis akhir     : {c['dosis']} ml  ({c['tingkat']})  -> {c['durasi']} detik pompa")
    print(f"|   Status mitigasi : {c['status_mitigasi']}")
    print(f"| POMPA       : {c['ket_pompa']}")
    print(f"| REKOMENDASI AGRONOMI [{c['sumber_llm']}]")
    for baris in textwrap.wrap(c["rekomendasi_llm"], 66):
        print(f"|   {baris}")
    print(g)


# ==========================================================
# 7. EKSEKUSI UTAMA PIPELINE
# ==========================================================
def jalankan_pipeline_cloud(default_hst=41):
    global _cache_cnn, _waktu_pompa_terakhir, _ack_terakhir

    print("\n[SCAN] Memeriksa kelengkapan data IoT Firebase & citra Supabase...")

    # --- (1) EVALUASI DATA IOT ---
    data_iot = ambil_data_iot_firebase()
    iot = evaluasi_iot(data_iot)

    ack = data_iot.get("ack") if data_iot else None
    if isinstance(ack, dict) and (ack.get("id"), ack.get("status")) != _ack_terakhir:
        _ack_terakhir = (ack.get("id"), ack.get("status"))
        print(f"       -> [ACK ESP32] Perintah {ack.get('id')}: {ack.get('status')} "
              f"(aktual {ack.get('durasi_aktual_ms')} ms)")

    # --- (2) EVALUASI CITRA DAUN DARI SUPABASE (HANYA 24 JAM TERAKHIR) ---
    citra = evaluasi_citra()

    # --- (3) GERBANG KELENGKAPAN: HARUS LENGKAP KEDUANYA BARU BEKERJA ---
    if not iot["valid"] or not citra["valid"]:
        print("+" + "-" * 70 + "+")
        print("| [STANDBY] MENUNGGU KELENGKAPAN DATA RIIL (HANYA 24 JAM TERAKHIR)")
        print("+" + "-" * 70 + "+")
        status_iot_txt = "LENGKAP & AKTIF" if iot["valid"] else f"BELUM LENGKAP ({LABEL_STATUS_IOT.get(iot['status'], iot['status'])})"
        print(f"| Telemetri IoT : {status_iot_txt}")
        for al in iot["alasan"]:
            print(f"|                 - {al}")
        
        status_foto_txt = "VALID (<= 24 Jam Terakhir)" if citra["valid"] else "TIDAK VALID / KADALUARSA"
        print(f"| Citra Daun    : {status_foto_txt}")
        print(f"|                 - {citra['alasan']}")
        print("+" + "-" * 70 + "+")
        print("[INFO] Analisis dan pompa ditangguhkan sampai data IoT lengkap dan foto baru (<= 24 jam) tersedia. Pindai ulang 10s...\n")
        return False

    # JIKA KEDUANYA LENGKAP: EKSEKUSI PIPELINE DENGAN DATA 100% NYATA
    cnn = _cache_cnn
    foto_baru = cnn.pop("baru", False)

    nilai = iot["nilai"]
    temperature = nilai["temperature"]
    soil_moisture = nilai["soil_moisture"]
    rh_udara = nilai["humidity"]

    if data_iot and "hst" in data_iot:
        hst_tanaman, sumber_hst = int(float(data_iot["hst"])), "Firebase"
    else:
        hst_tanaman, sumber_hst = default_hst, "default"

    if _waktu_pompa_terakhir == 0 and data_iot:
        _waktu_pompa_terakhir = float(data_iot.get("last_pump_epoch", 0) or 0)

    # --- (4) MESIN FUZZY DENGAN DATA RIIL ---
    sekarang = time.time()
    umur_jam = ((sekarang - cnn["ts"]) / 3600.0) if cnn.get("ts") else None

    dosis_ml, durasi_detik, rekomendasi, status_mitigasi, fase_hst = hitung_fuzzy_dan_workflow(
        hst_tanaman, rh_udara, soil_moisture, temperature, cnn["skor"]
    )
    tingkat = tingkat_dosis(dosis_ml)

    # --- (5) KEPUTUSAN POMPA & COOLDOWN ---
    cooldown_detik = COOLDOWN_PER_TINGKAT.get(tingkat, 0) * SKALA_COOLDOWN
    sisa_cooldown = cooldown_detik - (sekarang - _waktu_pompa_terakhir)
    menyalakan = (rekomendasi == "ON") and (sisa_cooldown <= 0)
    pump_cmd = "ON" if menyalakan else "OFF"

    if menyalakan:
        ket_pompa = f"PERINTAH DIKIRIM ke ESP32: {dosis_ml} ml / {durasi_detik} detik"
    elif rekomendasi == "ON":
        ket_pompa = f"DIKUNCI cooldown {tingkat} {cooldown_detik/3600:.1f} jam, sisa {int(sisa_cooldown)} detik"
    else:
        ket_pompa = "Tidak perlu menyala (dosis < 10 ml / kondisi aman)"

    waktu_skrg = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- (6) REKOMENDASI AGRONOMI GROQ LLM (DATA RIIL) ---
    rek_llm, sumber_llm = rekomendasi_untuk({
        "cnn_label": cnn["label"], "cnn_confidence": cnn["conf"], "skor_gejala": cnn["skor"],
        "tingkat": tingkat, "dosis_fuzzy_ml": dosis_ml, "durasi_detik": durasi_detik,
        "status_mitigasi": status_mitigasi, "fase": fase_hst, "hst": hst_tanaman,
        "suhu_c": temperature, "kelembapan_tanah_persen": soil_moisture,
        "kelembapan_udara_persen": rh_udara, "status_iot": "AKTIF",
        "mode_keputusan": "DATA_RIIL_IOT+CNN", "foto": cnn["path"],
    }, foto_baru)

    cetak_ringkasan({
        "waktu": waktu_skrg, "foto": cnn["path"], "umur_jam": umur_jam,
        "label": cnn["label"], "conf": cnn["conf"], "skor": cnn["skor"],
        "status_iot": iot["status"], "alasan_iot": iot["alasan"],
        "nilai": nilai, "hst": hst_tanaman, "sumber_hst": sumber_hst,
        "fase": fase_hst, "detail": _fuzzy_detail, "dosis": dosis_ml, "tingkat": tingkat,
        "durasi": durasi_detik, "status_mitigasi": status_mitigasi, "ket_pompa": ket_pompa,
        "rekomendasi_llm": rek_llm, "sumber_llm": sumber_llm,
    })

    main_payload = {}
    if menyalakan:
        main_payload["last_pump_epoch"] = int(sekarang)

    fuzzy_payload = {
        "cnn_label": cnn["label"],
        "cnn_confidence": cnn["conf"],
        "cnn_symptom_score": cnn["skor"],
        "status_iot": "AKTIF",
        "input_suhu": temperature,
        "input_kelembapan": soil_moisture,
        "input_kelembapan_udara": rh_udara,
        "dosis_ml": dosis_ml if menyalakan else 0.0,
        "dosis_rekomendasi_ml": dosis_ml,
        "durasi_pompa_detik": durasi_detik if menyalakan else 0.0,
        "agitator_detik": 15 if menyalakan else 0,
        "status_mitigasi": status_mitigasi,
        "fase_hst": fase_hst,
        "pump_command": pump_cmd,
        "foto_sumber": cnn["path"],
        "foto_umur_jam": None if umur_jam is None else round(umur_jam, 2),
        "mode_keputusan": "DATA_RIIL_IOT+CNN",
        "tingkat_dosis": tingkat,
        "cooldown_jam": round(cooldown_detik / 3600.0, 2),
        "sisa_cooldown_detik": max(0, int(sisa_cooldown)) if rekomendasi == "ON" else 0,
        "rekomendasi_llm": rek_llm,
        "rekomendasi_llm_sumber": sumber_llm,
        "last_updated": waktu_skrg
    }

    url_main = f"{FIREBASE_DB_URL}{NODE_PATH}.json?auth={FIREBASE_API_KEY}"
    url_fuzzy = f"{FIREBASE_DB_URL}{NODE_PATH}/fuzzy_output.json?auth={FIREBASE_API_KEY}"
    url_cmd = f"{FIREBASE_DB_URL}{NODE_PATH}/command.json?auth={FIREBASE_API_KEY}"
    url_log = f"{FIREBASE_DB_URL}/history_log/cabai_01.json?auth={FIREBASE_API_KEY}"

    try:
        if main_payload:
            requests.patch(url_main, json=main_payload, timeout=8)
        requests.patch(url_fuzzy, json=fuzzy_payload, timeout=8)
        if foto_baru or menyalakan:
            requests.post(url_log, json=fuzzy_payload, timeout=8)
        if menyalakan and durasi_detik > 0:
            perintah = {
                "id": int(sekarang),
                "dosis_ml": dosis_ml,
                "durasi_detik": durasi_detik,
                "issued_at": int(sekarang),
                "status_mitigasi": status_mitigasi,
            }
            requests.put(url_cmd, json=perintah, timeout=8)
            _waktu_pompa_terakhir = sekarang
            print(f"       -> [PERINTAH] id {perintah['id']} dikirim ke ESP32. Menunggu ack...")
        print("       -> [FIREBASE] fuzzy_output diperbarui dengan data nyata.")
    except Exception as e:
        print(f"       -> [PERINGATAN] Gagal menulis ke Firebase ({e}). Hasil di atas tetap valid.")
    return True


if __name__ == "__main__":
    import sys
    print("[INFO] Program FusaShield Cloud Controller Berjalan (Mode 100% Data Riil).")
    if "--once" in sys.argv:
        print("[INFO] Mode single execution (--once) aktif untuk cloud automation.")
        jalankan_pipeline_cloud(default_hst=41)
        sys.exit(0)
    while True:
        try:
            jalankan_pipeline_cloud(default_hst=41)
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n[INFO] Program dihentikan secara manual oleh pengguna (Ctrl+C).")
            break
        except Exception as e:
            print(f"[ERROR TERJADI] {e}. Melanjutkan scanning dalam 10 detik...")
            time.sleep(10)
