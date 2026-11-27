"""
=============================================================================
FUSASHIELD CLOUD CONTROLLER (100% DATA RIIL: IOT & CITRA LENGKAP)
Sistem Mitigasi Presisi Layu Fusarium Berbasis IoT dan Agen Hayati Otomatis
=============================================================================
Sistem HANYA bekerja jika KEDUA syarat terpenuhi secara lengkap:
  1. Data Telemetri IoT dari ESP32 LENGKAP & AKTIF (Suhu, Tanah, RH Udara).
  2. Citra Daun Cabai TERSEDIA & VALID di Supabase Storage.
TANPA DATA DEMO / NILAI SIMULASI.

LLM: Cloudflare Workers AI (hanya memberi saran agronomi; dosis & pompa
      sepenuhnya ditentukan oleh mesin fuzzy).
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
from label_translations import label_to_indonesian

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
# Cooldown dinonaktifkan sementara untuk menguji aktuasi pompa.
COOLDOWN_STANDAR_DETIK = 0
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

# ---------------- LLM (Cloudflare Workers AI) : Narasi rekomendasi agronomi XAI ----------------
# CF_ACCOUNT_ID : Account ID Cloudflare (dasbor -> Workers & Pages -> Overview, sisi kanan)
# CF_API_TOKEN  : API Token dengan izin "Workers AI" (My Profile -> API Tokens)
CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "")
CF_MODEL = os.environ.get("CF_MODEL", "@cf/meta/llama-3.1-8b-instruct")
CF_MODEL_FALLBACK = os.environ.get("CF_MODEL_FALLBACK", "@cf/meta/llama-3.2-3b-instruct")
CF_API_BASE = "https://api.cloudflare.com/client/v4"

LLM_TIMEOUT_DETIK = 15
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

try:
    import keras.layers
    _orig_dense = keras.layers.Dense.__init__
    def _patched_dense(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        return _orig_dense(self, *args, **kwargs)
    keras.layers.Dense.__init__ = _patched_dense
except Exception:
    pass

print("[INFO] Memuat model CNN MobileNetV2 (.keras)...")
model = keras.models.load_model(MODEL_PATH, compile=False)
with open(LABELS_PATH, "r") as f:
    class_names = json.load(f)
print(f"[INFO] Model MobileNetV2 siap digunakan! ({len(class_names)} kelas: {class_names})")

# Path state lokal persisten untuk deduplikasi gambar
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PROCESSED_IMAGES_FILE = os.path.join(OUTPUT_DIR, ".processed_images.json")
LEGACY_PROCESSED_IMAGE_FILE = os.path.join(BASE_DIR, ".last_processed_photo.json")
_cache_cnn = None

def _muat_processed_images_lokal():
    """Membaca daftar processing_key gambar yang sudah diproses dari file lokal."""
    processed_keys = set()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(PROCESSED_IMAGES_FILE):
        try:
            with open(PROCESSED_IMAGES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    processed_keys.update(data)
                elif isinstance(data, dict):
                    processed_keys.update(data.keys())
        except Exception as e:
            print(f"[WARN] Gagal membaca state lokal {PROCESSED_IMAGES_FILE}: {e}")
    if os.path.exists(LEGACY_PROCESSED_IMAGE_FILE):
        try:
            with open(LEGACY_PROCESSED_IMAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            id_foto = data.get("id_foto") if isinstance(data, dict) else None
            if isinstance(id_foto, str):
                processed_keys.add(id_foto)
        except Exception as e:
            print(f"[WARN] Gagal membaca state lama {LEGACY_PROCESSED_IMAGE_FILE}: {e}")
    return processed_keys

def _simpan_processed_key_lokal_atomic(processing_key, metadata_tambahan=None):
    """Menyimpan processing_key baru ke state lokal secara atomic."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = {}
    if os.path.exists(PROCESSED_IMAGES_FILE):
        try:
            with open(PROCESSED_IMAGES_FILE, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    existing = {k: {} for k in content}
                elif isinstance(content, dict):
                    existing = content
        except Exception:
            existing = {}

    existing[processing_key] = metadata_tambahan or {"processed_at": datetime.now().isoformat()}

    tmp_file = f"{PROCESSED_IMAGES_FILE}.tmp_{os.getpid()}_{int(time.time()*1000)}"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        os.replace(tmp_file, PROCESSED_IMAGES_FILE)
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan {processing_key} secara atomic: {e}")
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass

def cek_key_di_firebase_history(processing_keys, max_retries=3):
    """Cek apakah salah satu kunci foto sudah tercatat di Firebase history."""
    processing_keys = set(processing_keys)
    url = f"{FIREBASE_DB_URL}/history_log/cabai_01.json?auth={FIREBASE_API_KEY}"
    for attempt in range(max_retries):
        try:
            res = requests.get(url, timeout=6)
            res.raise_for_status()
            data = res.json()
            if not data or not isinstance(data, dict):
                return False
            for record in data.values():
                if isinstance(record, dict):
                    stored_keys = (record.get("processing_key"), record.get("id_foto"))
                    if any(isinstance(key, str) and key in processing_keys for key in stored_keys):
                        return True
            return False
        except Exception as e:
            delay = 2 ** attempt
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print(f"[WARN] Gagal verifikasi history Firebase ({e}). Mengandalkan state lokal.")
    return False

# Inisialisasi set processing_keys dari file state lokal saat startup
_processed_keys = _muat_processed_images_lokal()
if _processed_keys:
    print(f"[INFO] Persistent image deduplication aktif: {len(_processed_keys)} foto tercatat di state lokal.")

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


def periksa_metadata_dan_analisis_citra():
    """
    Persistent Image Deduplication:
    1. Ambil metadata file saja dari Supabase Storage.
    2. Bentuk processing_key stabil: image_path + "|" + image_updated_at.
    3. Cek apakah processing_key sudah ada di state lokal persisten (output/.processed_images.json)
       atau di database history Firebase.
    4. Jika sudah ditemukan:
       - Jangan download gambar.
       - Jangan jalankan preprocessing / model.
       - Jangan buat record history baru.
    5. Jika belum ditemukan:
       - Download dan jalankan model CNN.
       - Kembalikan status siap diproses ke database.
    """
    global _cache_cnn, _processed_keys

    # 1. Ambil metadata file saja
    path_foto, info = cari_file_terbaru()
    if path_foto is None or info is None:
        return _cache_cnn is not None, False, None

    created_time = info.get("created_at") or ""
    updated_time = info.get("updated_at") or created_time
    processing_key = f"{path_foto}|{updated_time}"
    legacy_processing_key = f"{path_foto}|{info.get('created_at')}|{info.get('updated_at')}"
    candidate_keys = {processing_key, legacy_processing_key}

    # 2 & 3. Cek di state lokal persisten
    if candidate_keys.intersection(_processed_keys):
        if processing_key not in _processed_keys:
            _processed_keys.add(processing_key)
            _simpan_processed_key_lokal_atomic(
                processing_key,
                {"detected": "legacy_local_state", "legacy_processing_key": legacy_processing_key}
            )
        if _cache_cnn is None or _cache_cnn.get("processing_key") != processing_key:
            _cache_cnn = {
                "path": path_foto,
                "ts": waktu_foto_epoch(info),
                "processing_key": processing_key
            }
        return True, False, processing_key

    # Reuse CNN cache if the sensors or a Firebase write were not ready last scan.
    if _cache_cnn is not None and _cache_cnn.get("processing_key") == processing_key:
        return True, True, processing_key

    # Cek di database history Firebase
    if cek_key_di_firebase_history(candidate_keys):
        _processed_keys.add(processing_key)
        _simpan_processed_key_lokal_atomic(processing_key, {"detected": "firebase_history"})
        _cache_cnn = {
            "path": path_foto,
            "ts": waktu_foto_epoch(info),
            "processing_key": processing_key
        }
        return True, False, processing_key

    ts_foto = waktu_foto_epoch(info)
    if ts_foto is not None:
        umur_jam = (time.time() - ts_foto) / 3600.0
        if umur_jam > FOTO_MAKS_UMUR_JAM:
            _processed_keys.add(processing_key)
            _simpan_processed_key_lokal_atomic(
                processing_key,
                {"status": "expired_before_analysis", "age_hours": round(umur_jam, 2)}
            )
            _cache_cnn = {
                "path": path_foto,
                "ts": ts_foto,
                "processing_key": processing_key
            }
            return True, False, processing_key

    # 5. Jika belum pernah diproses sama sekali:
    print(f"\n[DEDUPLICATION] Citra baru ditemukan! processing_key: '{processing_key}'")
    print(f"       -> Mengunduh citra dari Supabase Storage...")
    pil_img = download_gambar(path_foto)
    if pil_img is None:
        return _cache_cnn is not None, False, None

    print(f"       -> Menjalankan inferensi model MobileNetV2...")
    label, conf, skor = analisis_citra_pil(pil_img)
    _cache_cnn = {
        "label": label,
        "conf": conf,
        "skor": skor,
        "path": path_foto,
        "ts": waktu_foto_epoch(info),
        "baru": True,
        "processing_key": processing_key
    }
    print(f"       -> CNN MobileNetV2: {label} ({conf}%) -> Skor Gejala: {skor}%")
    return True, True, processing_key


def evaluasi_citra():
    """Memeriksa apakah citra daun di Supabase tersedia, belum pernah diproses, DAN berumur <= 24 jam."""
    ada, adalah_foto_baru, processing_key = periksa_metadata_dan_analisis_citra()
    if not ada or _cache_cnn is None:
        return {
            "valid": False,
            "baru": False,
            "umur_jam": None,
            "foto": None,
            "alasan": "Belum ada file citra di bucket fusashield-image/cabai_01"
        }

    sekarang = time.time()
    ts_foto = _cache_cnn.get("ts")
    umur_jam = ((sekarang - ts_foto) / 3600.0) if ts_foto else None

    # Syarat mutlak 1: Foto harus berasal dari 24 jam terakhir
    if umur_jam is not None and umur_jam > FOTO_MAKS_UMUR_JAM:
        return {
            "valid": False,
            "baru": False,
            "umur_jam": round(umur_jam, 2),
            "foto": _cache_cnn["path"],
            "alasan": f"Citra '{_cache_cnn['path']}' KADALUARSA (umur {umur_jam:.1f} jam > batas {FOTO_MAKS_UMUR_JAM:.0f} jam). Harap unggah foto daun cabai terbaru!"
        }

    # Syarat mutlak 2: Foto tidak boleh foto yang sudah pernah diproses sebelumnya
    if not adalah_foto_baru:
        return {
            "valid": False,
            "baru": False,
            "umur_jam": round(umur_jam, 2) if umur_jam else None,
            "foto": _cache_cnn["path"],
            "alasan": f"Citra '{_cache_cnn['path']}' sudah pernah diproses (processing_key terdaftar). Menunggu unggahan foto daun baru..."
        }

    return {
        "valid": True,
        "baru": True,
        "umur_jam": None if umur_jam is None else round(umur_jam, 2),
        "foto": _cache_cnn["path"],
        "processing_key": processing_key,
        "alasan": f"Citra baru valid dari 24 jam terakhir (umur {umur_jam:.1f} jam)" if umur_jam is not None else "Citra baru valid"
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
# 6. REKOMENDASI AGRONOMI CLOUDFLARE WORKERS AI (XAI)
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
    label_tampilan = label_to_indonesian(k["cnn_label"])
    return (f"Indikasi: {label_tampilan} (skor gejala {k['skor_gejala']}%, tingkat {k['tingkat']}). {saran} "
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


def panggil_cloudflare_api(model_name, payload_data):
    """Panggil Cloudflare Workers AI lewat REST API (endpoint /ai/run/{model})."""
    url = f"{CF_API_BASE}/accounts/{CF_ACCOUNT_ID}/ai/run/{model_name}"
    body = {
        "messages": [
            {"role": "system", "content": PROMPT_SISTEM},
            {"role": "user", "content": json.dumps(payload_data, ensure_ascii=False)},
        ],
        "temperature": 0.3,
        "max_tokens": 400,
    }
    r = requests.post(
        url,
        json=body,
        timeout=LLM_TIMEOUT_DETIK,
        headers={
            "Authorization": f"Bearer {CF_API_TOKEN}",
            "Content-Type": "application/json",
        },
    )
    r.raise_for_status()
    res_json = r.json()

    if res_json.get("success") is False:
        raise RuntimeError(f"Cloudflare error: {res_json.get('errors')}")

    hasil = res_json.get("result") or {}
    teks = hasil.get("response")

    # Beberapa model memakai format OpenAI-compatible (choices[...])
    if not teks and hasil.get("choices"):
        teks = hasil["choices"][0].get("message", {}).get("content", "")

    if not isinstance(teks, str):
        return ""
    return teks.strip()


def panggil_llm(k):
    """Panggil LLM: Cloudflare Workers AI (model utama -> model cadangan) -> Template Offline."""
    riwayat_error = []

    if CF_ACCOUNT_ID and CF_API_TOKEN:
        cf_models = [CF_MODEL]
        if CF_MODEL_FALLBACK and CF_MODEL_FALLBACK != CF_MODEL:
            cf_models.append(CF_MODEL_FALLBACK)

        for m_name in cf_models:
            try:
                teks = panggil_cloudflare_api(m_name, k)
                if not teks:
                    riwayat_error.append(f"cloudflare:{m_name} kosong")
                    continue
                if not _dosis_dalam_teks_valid(teks, k["dosis_fuzzy_ml"]):
                    riwayat_error.append(f"cloudflare:{m_name} dosis berbeda")
                    continue
                return teks[:1200], f"cloudflare:{m_name}"
            except Exception as e:
                riwayat_error.append(f"cloudflare:{m_name} ({type(e).__name__})")

    # Fallback terakhir ke template jika Cloudflare gagal / belum dikonfigurasi
    alasan = ", ".join(riwayat_error) if riwayat_error else "CF_ACCOUNT_ID / CF_API_TOKEN belum diset"
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

        status_foto_txt = "VALID (Foto Baru <= 24 Jam)" if citra["valid"] else ("SUDAH DIPROSES / MENUNGGU FOTO BARU" if not citra.get("baru", True) else "TIDAK VALID / KADALUARSA")
        print(f"| Citra Daun    : {status_foto_txt}")
        print(f"|                 - {citra['alasan']}")
        print("+" + "-" * 70 + "+")
        print("[INFO] Analisis dan pompa ditangguhkan sampai data IoT lengkap dan ada foto baru yang diunggah. Pindai ulang 10s...\n")
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

    # --- (6) REKOMENDASI AGRONOMI CLOUDFLARE WORKERS AI (DATA RIIL) ---
    # LLM hanya memberi saran; dosis & pompa tetap dari fuzzy di atas.
    rek_llm, sumber_llm = rekomendasi_untuk({
        "cnn_label": cnn["label"], "cnn_label_display": label_to_indonesian(cnn["label"]),
        "cnn_confidence": cnn["conf"], "skor_gejala": cnn["skor"],
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
        "cnn_label": label_to_indonesian(cnn["label"]),
        "cnn_label_original": cnn["label"],
        "cnn_confidence": cnn["conf"],
        "cnn_symptom_score": cnn["skor"],
        "status_iot": "AKTIF",
        "input_suhu": temperature,
        "input_kelembapan": soil_moisture,
        "input_kelembapan_udara": rh_udara,
        "dosis_ml": dosis_ml if menyalakan else 0.0,
        "dosis_rekomendasi_ml": dosis_ml,
        "durasi_pompa_detik": durasi_detik if menyalakan else 0.0,
        "agitator_detik": 5 if menyalakan else 0,
        "status_mitigasi": status_mitigasi,
        "fase_hst": fase_hst,
        "pump_command": pump_cmd,
        "foto_sumber": cnn["path"],
        "processing_key": cnn.get("processing_key"),
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

    # Pengiriman database dengan retry backoff agar tidak menandai sukses jika koneksi gagal
    max_db_retries = 3
    db_sukses = False

    for attempt in range(max_db_retries):
        try:
            # 1. Update node latest / monitoring
            if main_payload:
                r_main = requests.patch(url_main, json=main_payload, timeout=8)
                r_main.raise_for_status()

            # Node latest fuzzy_output (PUT/PATCH tanpa merusak history)
            r_fuzzy = requests.put(url_fuzzy, json=fuzzy_payload, timeout=8)
            r_fuzzy.raise_for_status()

            # 2. Append history baru dengan POST (auto-generated ID, append-only, history lama tidak ditimpa)
            if foto_baru or menyalakan:
                r_log = requests.post(url_log, json=fuzzy_payload, timeout=8)
                r_log.raise_for_status()

            # 3. Perintah aktuasi pompa jika kondisi terpenuhi
            if menyalakan and durasi_detik > 0:
                perintah = {
                    "id": int(sekarang),
                    "dosis_ml": dosis_ml,
                    "durasi_detik": durasi_detik,
                    "issued_at": int(sekarang),
                    "status_mitigasi": status_mitigasi,
                }
                r_cmd = requests.put(url_cmd, json=perintah, timeout=8)
                r_cmd.raise_for_status()
                _waktu_pompa_terakhir = sekarang
                print(f"       -> [PERINTAH] id {perintah['id']} dikirim ke ESP32. Menunggu ack...")

            db_sukses = True
            print("       -> [FIREBASE] fuzzy_output diperbarui & history dicatat.")
            break
        except Exception as e:
            delay = 2 ** attempt
            print(f"       -> [RETRY {attempt+1}/{max_db_retries}] Gagal menulis ke Firebase ({e}). Tunggu {delay}s...")
            time.sleep(delay)

    # 4. Simpan processing_key ke state lokal SECARA ATOMIC HANYA JIKA DATABASE SUKSES
    current_key = cnn.get("processing_key")
    if db_sukses and current_key:
        _processed_keys.add(current_key)
        _simpan_processed_key_lokal_atomic(current_key, {
            "path": cnn["path"],
            "label": cnn["label"],
            "skor": cnn["skor"],
            "dosis_ml": dosis_ml,
            "processed_at": waktu_skrg
        })
        print(f"       -> [PERSISTENCE] processing_key '{current_key}' berhasil disimpan ke state lokal.")
    elif not db_sukses:
        print("       -> [GAGAL] Database tidak merespons, gambar TIDAK ditandai selesai agar dapat diproses ulang.")

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