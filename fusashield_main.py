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
# 1. KONFIGURASI KREDENSIAL SUPABASE & FIREBASE
# ==========================================================
def _muat_env(path=".env"):
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
        except Exception:
            pass

_muat_env()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "fusashield-image")
SUPABASE_FOLDER = os.environ.get("SUPABASE_FOLDER", "cabai_01")

FIREBASE_DB_URL = os.environ.get("FIREBASE_DB_URL", "")
FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY", "")
NODE_PATH = os.environ.get("NODE_PATH", "/monitoring/cabai_01")

MODEL_PATH = r"C:\project\Fusashield\fusashield_mobilenetv2.keras"
LABELS_PATH = r"C:\project\Fusashield\class_names.json"

EXT_GAMBAR = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Cooldown penguncian pompa mengikuti TINGKAT KEPARAHAN dosis yang direkomendasikan fuzzy.
# Dihitung sejak pompa terakhir menyala (sesuai jeda inkubasi Trichoderma).
# Cooldown Protection: Penguncian pompa otomatis selama 12 jam (43.200 detik)
# setelah penyemprotan untuk mencegah kejenuhan air tanah (over-watering).
COOLDOWN_STANDAR_DETIK = 12 * 3600  # 12 jam (43.200 detik)
COOLDOWN_PER_TINGKAT = {
    "KRITIS":    COOLDOWN_STANDAR_DETIK,   # dosis >= 70 ml (12 jam / 43.200 detik)
    "WASPADA":   COOLDOWN_STANDAR_DETIK,   # 35 ml <= dosis < 70 ml (12 jam / 43.200 detik)
    "PREVENTIF": COOLDOWN_STANDAR_DETIK,   # 10 ml <= dosis < 35 ml (12 jam / 43.200 detik)
}
# Pengali untuk uji di meja: 1.0 = normal, 0.002 = Kritis jadi +-86 detik.
SKALA_COOLDOWN = 1.0

# Foto lebih tua dari ini TIDAK dipakai untuk dosis kuratif. Fuzzy lalu hanya
# memakai sensor tanah/udara (skor gejala dianggap 0), sehingga dosis paling besar preventif.
FOTO_MAKS_UMUR_JAM = 24.0

# ESP32 mengirim data tiap 10 detik. Jika uptime_ms tidak berubah selama ini,
# data di Firebase dianggap LAMA (ESP32 mati / belum menyala).
IOT_TIMEOUT_DETIK = 45

# Nilai netral untuk MODE PERHITUNGAN TANPA IoT. Dipakai hanya untuk sensor yang
# belum masuk, supaya fuzzy tetap berjalan. Dengan nilai ini dosis hanya ditentukan
# oleh skor CNN (maksimum Waspada). Pompa TIDAK PERNAH menyala pada mode ini.
NILAI_DEMO_SENSOR = {"temperature": 28.0, "soil_moisture": 60.0, "humidity": 65.0}

# ---------------- LLM (Groq) : HANYA narasi rekomendasi agronomi ----------------
# Kunci dibaca dari environment variable, JANGAN ditulis di kode:
#   Kunci API Groq dibaca otomatis dari file .env lokal
#   Kunci API Groq dibaca otomatis dari file .env lokal
# LLM dipanggil SETELAH fuzzy & keputusan pompa selesai. Keluarannya tidak pernah
# memengaruhi dosis/pompa; hanya teks untuk dashboard.
# Pengaturan Mode Uji Meja / Demo Sidang:
# Jika True: Sensor tanah yang belum ditancapkan ke tanah (soil_valid: false dari ESP32)
# tidak akan memblokir pompa. Nilai soil_moisture tetap dibaca dari Firebase (atau nilai netral jika null),
# dan pompa tetap diizinkan menyala jika dosis fuzzy menghendaki.
ABAIKAN_SOIL_INVALID_UNTUK_DEMO = True

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
    # pH tanah tidak dipakai (tidak ada sensor pH); diganti kelembapan udara DHT22
    "humidity":      ["humidity", "kelembapan_udara", "air_humidity", "rh"],
}

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("[INFO] Memuat model CNN MobileNetV2 (.keras)...")
model = keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    class_names = json.load(f)
print("[INFO] Model MobileNetV2 siap digunakan!")

# Menyimpan ID foto terakhir yang sudah diproses, agar foto yang sama
# tidak dianalisis dan pompa tidak menyala berulang setiap 10 detik.
_foto_terakhir_diproses = None
_cache_cnn = None          # hasil CNN foto terakhir: dict(label, conf, skor, path)
_waktu_pompa_terakhir = 0  # epoch detik saat pompa terakhir menyala
_ack_terakhir = None       # id ack terakhir dari ESP32 yang sudah dicetak
_iot_uptime_terakhir = None   # uptime_ms ESP32 terakhir yang terlihat
_iot_waktu_berubah = 0.0      # kapan uptime_ms terakhir berubah (epoch)
_iot_pernah_berubah = False   # sudah pernah melihat uptime berubah (bukti ESP32 hidup)
_llm_cache = {"key": None, "teks": "", "sumber": "", "ts": 0.0}
_fuzzy_detail = {}            # rincian derajat keanggotaan & kekuatan aturan (untuk ditampilkan)


# ==========================================================
# 2. AMBIL GAMBAR TERBARU DARI SUPABASE (TANPA NAMA FILE TETAP)
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
    """Waktu unggah foto (epoch detik): dari created_at, atau dari angka
    timestamp di awal nama file. Return None jika tidak bisa ditentukan."""
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
    """Pilih file gambar terbaru dari hasil list() Supabase.

    Urutan penentu: created_at / updated_at, lalu angka timestamp di awal
    nama file (mis. '1793541712542-daun.jpg'). Folder dan file placeholder
    (.emptyFolderPlaceholder) otomatis terabaikan.
    """
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

    # Diagnosa: tampilkan apa yang sebenarnya terlihat oleh script
    nama_terlihat = [f.get("name") for f in (files or [])]
    print(f"[DEBUG] Isi '{SUPABASE_FOLDER}' yang terlihat oleh script: {nama_terlihat}")
    if not files:
        print("[DEBUG] List kosong. Penyebab umum: tidak ada policy SELECT untuk role 'anon' "
              "pada storage.objects (lihat catatan di bawah script), atau nama folder salah.")
    else:
        print(f"[DEBUG] Ada file tapi tidak berekstensi {EXT_GAMBAR}.")
    return None, None


def download_gambar(path_lengkap):
    """Download via API; jika gagal, coba lewat public URL (bucket PUBLIC)."""
    data = None
    try:
        data = supabase.storage.from_(BUCKET_NAME).download(path_lengkap)
    except Exception as e:
        print(f"[WARN] Download via API gagal ({e}). Mencoba public URL...")
        try:
            url = supabase.storage.from_(BUCKET_NAME).get_public_url(path_lengkap)
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.content
        except Exception as e2:
            print(f"[ERROR] Download via public URL juga gagal: {e2}")
            return None

    try:
        img = Image.open(io.BytesIO(data))
        return ImageOps.exif_transpose(img)  # koreksi rotasi foto HP
    except Exception as e:
        print(f"[ERROR] File bukan gambar yang valid: {e}")
        return None


# ==========================================================
# 3. KOMUNIKASI FIREBASE (GET & PATCH)
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


def update_keputusan_ke_firebase(main_payload, fuzzy_payload):
    url_main = f"{FIREBASE_DB_URL}{NODE_PATH}.json?auth={FIREBASE_API_KEY}"
    url_fuzzy = f"{FIREBASE_DB_URL}{NODE_PATH}/fuzzy_output.json?auth={FIREBASE_API_KEY}"
    url_log = f"{FIREBASE_DB_URL}/history_log/cabai_01.json?auth={FIREBASE_API_KEY}"
    try:
        requests.patch(url_main, json=main_payload, timeout=8)
        requests.patch(url_fuzzy, json=fuzzy_payload, timeout=8)
        requests.post(url_log, json=fuzzy_payload, timeout=8)
        return True
    except Exception as e:
        print(f"[ERROR] Gagal memperbarui data ke Firebase: {e}")
        return False


# ==========================================================
# 4. ANALISIS CNN & MESIN FUZZY LOGIC
# ==========================================================
def analisis_citra_pil(pil_img):
    # Model sudah punya layer Rescaling [-1,1] di dalamnya,
    # jadi cukup kirim piksel mentah 0-255 (jangan dinormalisasi lagi).
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
    """Fungsi keanggotaan trapesium [a, b, c, d].
    Plateau diperiksa lebih dulu agar bahu kiri (a=b) dan bahu kanan (c=d)
    bernilai 1 di titik batas (mis. kelembapan 0% = Kering, RH 100% = Tinggi)."""
    if b <= x <= c:
        return 1.0
    if x <= a or x >= d:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def hitung_fuzzy_dan_workflow(hst, rh_udara, soil_moisture, temperature, skor_cnn):
    # Kelembapan udara (%RH, DHT22) menggantikan pH tanah. Pemetaan aturan:
    # pH Asam -> RH Tinggi (kondisi pemicu penyakit), Netral -> Sedang, Basa -> Rendah.
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
# 5. EKSEKUSI UTAMA PIPELINE CLOUD (CONTINUOUS SCANNING)
# ==========================================================
def baca_sensor(data_iot):
    """Ambil suhu, kelembapan, pH, HST dari data Firebase (dengan alias nama).
    Return (dict_sensor, daftar_kunci_hilang)."""
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
    """Cek foto terbaru; CNN hanya dijalankan jika fotonya BARU.
    Return True jika ada hasil CNN yang bisa dipakai (baru atau dari cache)."""
    global _cache_cnn, _foto_terakhir_diproses

    path_foto, info = cari_file_terbaru()
    if path_foto is None:
        return _cache_cnn is not None

    id_foto = f"{path_foto}|{info.get('created_at')}|{info.get('updated_at')}"
    if id_foto == _foto_terakhir_diproses:
        return True  # foto sama, pakai hasil CNN yang sudah ada

    print(f"       -> Foto baru: '{path_foto}' (created_at: {info.get('created_at')})")
    pil_img = download_gambar(path_foto)
    if pil_img is None:
        return _cache_cnn is not None

    label, conf, skor = analisis_citra_pil(pil_img)
    _cache_cnn = {"label": label, "conf": conf, "skor": skor, "path": path_foto,
                  "ts": waktu_foto_epoch(info), "baru": True}
    _foto_terakhir_diproses = id_foto
    print(f"       -> CNN MobileNetV2: {label} ({conf}%) -> Skor Gejala: {skor}%")
    return True


def evaluasi_iot(data_iot):
    """Tentukan status IoT dan nilai sensor yang dipakai fuzzy.

    status: AKTIF | MENUNGGU | SENSOR_TIDAK_LENGKAP | TIDAK_AKTIF | TIDAK_ADA_DATA
    Sensor yang belum masuk diganti NILAI_DEMO_SENSOR (ditandai sumber 'DEMO').
    siap_pompa hanya True jika ESP32 terbukti hidup DAN ketiga sensor nyata.
    """
    global _iot_uptime_terakhir, _iot_waktu_berubah, _iot_pernah_berubah
    nama = ["temperature", "soil_moisture", "humidity"]
    sekarang = time.time()
    semua_demo = {"nilai": dict(NILAI_DEMO_SENSOR), "sumber": {n: "DEMO" for n in nama},
                  "siap_pompa": False}

    if not data_iot:
        return dict(semua_demo, status="TIDAK_ADA_DATA",
                    alasan=["Firebase kosong atau tidak terjangkau"])

    alasan = []
    sensor, hilang = baca_sensor(data_iot)
    if data_iot.get("soil_valid") is False:
        if ABAIKAN_SOIL_INVALID_UNTUK_DEMO:
            alasan.append("sensor tanah soil_valid=False dari ESP32 (DIABAIKAN karena mode demo aktif)")
            if sensor.get("soil_moisture") is None:
                sensor["soil_moisture"] = NILAI_DEMO_SENSOR["soil_moisture"]
        else:
            sensor["soil_moisture"] = None
            if "soil_moisture" not in hilang:
                hilang.append("soil_moisture")
            alasan.append("sensor tanah ditandai TIDAK VALID oleh ESP32 (cek kabel/kalibrasi)")

    # Apakah ESP32 benar-benar hidup? (uptime_ms harus terus berubah)
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

    if hidup == "TIDAK_AKTIF":
        return dict(semua_demo, status="TIDAK_AKTIF",
                    alasan=[f"ESP32 tidak mengirim data baru > {IOT_TIMEOUT_DETIK} detik "
                            f"(data di Firebase sudah lama, diabaikan)"])

    nilai, sumber = {}, {}
    for n in nama:
        if sensor.get(n) is None:
            nilai[n], sumber[n] = NILAI_DEMO_SENSOR[n], "DEMO"
        else:
            nilai[n], sumber[n] = sensor[n], "IoT"

    if hilang:
        status = "SENSOR_TIDAK_LENGKAP"
        alasan.append(f"sensor belum masuk: {hilang}. Kunci di Firebase: {sorted(data_iot.keys())}")
    elif hidup == "MENUNGGU":
        status = "MENUNGGU"
        alasan.append("menunggu bukti ESP32 aktif (uptime_ms belum terlihat berubah)")
    else:
        status = "AKTIF"

    siap_pompa = (status == "AKTIF")
    if ABAIKAN_SOIL_INVALID_UNTUK_DEMO and hidup == "AKTIF" and (not hilang or hilang == ["soil_moisture"]):
        siap_pompa = True

    return {"status": status, "alasan": alasan, "nilai": nilai, "sumber": sumber,
            "siap_pompa": siap_pompa}


SARAN_PER_KELAS = {
    "Healthy Leaf": "Daun terlihat sehat. Lanjutkan pemantauan rutin dan jaga drainase agar tanah tidak tergenang.",
    "Nutrition Deficiency": "Gejala menguning/klorosis mengarah ke kekurangan hara atau gangguan serapan akar. "
                            "Periksa jadwal pemupukan (N, Mg, Fe), pastikan drainase baik, dan amati apakah gejala menyebar dari tanaman bawah.",
    "Bacterial Spot": "Buang daun bergejala berat, hindari percikan air ke daun, dan sanitasi alat. Hindari menyiram pada sore hari.",
    "Cercospora Leaf Spot": "Kurangi kelembapan di sekitar tajuk (jarak tanam, pemangkasan), buang daun terinfeksi, dan hindari penyiraman dari atas.",
    "Curl Virus": "Kendalikan serangga vektor (kutu kebul/thrips), cabut tanaman yang terinfeksi berat, dan pisahkan dari tanaman sehat.",
    "White spot": "Buang daun bergejala, jaga sirkulasi udara, dan pantau penyebaran ke tanaman sekitar.",
}


def rekomendasi_template(k):
    """Cadangan tanpa LLM: teks tetap dari kelas CNN + keputusan fuzzy. Selalu tersedia."""
    saran = SARAN_PER_KELAS.get(k["cnn_label"], "Pantau kondisi tanaman dan lakukan pemeriksaan langsung di lahan.")
    pompa = ("Pompa otomatis hanya menjalankan dosis dari fuzzy; lakukan tindakan fisik di atas sebagai pelengkap."
             if k["pompa_diizinkan"] else
             "Saat ini hanya perhitungan (data IoT belum lengkap/aktif), pompa tidak menyala.")
    return (f"Indikasi: {k['cnn_label']} (skor gejala {k['skor_gejala']}%, tingkat {k['tingkat']}). {saran} "
            f"Pencegahan: hindari genangan, jaga kelembapan tanah sedang, dan pantau tanaman sekitar. {pompa} "
            f"(Gejala visual belum berarti Fusarium; konfirmasi di lapangan.)")


PROMPT_SISTEM = (
    "Anda pakar patologi tanaman cabai yang memberi SARAN AGRONOMI TAMBAHAN dalam Bahasa Indonesia ilmiah. "
    "Sistem FusaShield memakai CNN MobileNetV2 (mengenali gejala daun, bukan diagnosis Fusarium pasti) "
    "dan fuzzy logic Mamdani (singleton defuzzification) untuk kendali aktuator dosis Trichoderma harzianum cair. "
    "ATURAN KETAT: "
    "(1) JANGAN menghitung, mengubah, atau mengusulkan angka dosis/durasi pompa; dosis dari fuzzy sudah final, "
    "dan Anda tidak boleh menyebut angka ml/cc selain yang diberikan. "
    "(2) Nyatakan temuan sebagai INDIKASI, bukan konfirmasi mutlak patogen Fusarium di laboratorium. "
    "(3) Jika sumber sensor bernilai DEMO, katakan itu nilai simulasi dan jangan menyimpulkan mikroklimat darinya. "
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
    """Return (teks, sumber). Menjalankan model utama (llama-3.3-70b-versatile),
    auto-fallback ke model kedua (llama-3.1-8b-instant), dan
    terakhir rule-based fallback bila offline/kedua API gagal."""
    if not GROQ_API_KEY:
        return rekomendasi_template(k), "template (GROQ_API_KEY belum diset)"

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

    # Fallback terakhir ke template jika semua model Groq gagal/offline
    alasan = ", ".join(riwayat_error) if riwayat_error else "tidak diketahui"
    return rekomendasi_template(k), f"template (fallback offline: {alasan})"


def rekomendasi_untuk(k, foto_baru):
    """Panggil LLM hanya bila perlu (foto baru, atau kondisi berubah dan jeda cukup)."""
    global _llm_cache
    kunci = (k["cnn_label"], k["tingkat"], k["status_iot"], k["mode_keputusan"], k["foto"])
    sekarang = time.time()
    c = _llm_cache
    perlu = (c["key"] is None or foto_baru or
             (kunci != c["key"] and sekarang - c["ts"] >= LLM_INTERVAL_MIN_DETIK))
    if perlu:
        teks, sumber = panggil_llm(k)
        _llm_cache = {"key": kunci, "teks": teks, "sumber": sumber, "ts": sekarang}
    return _llm_cache["teks"], _llm_cache["sumber"]


LABEL_STATUS_IOT = {
    "AKTIF": "AKTIF - sensor nyata dari ESP32",
    "MENUNGGU": "MENUNGGU konfirmasi ESP32 (data nyata dipakai, pompa belum diizinkan)",
    "SENSOR_TIDAK_LENGKAP": "SENSOR TIDAK LENGKAP - sebagian memakai nilai DEMO",
    "TIDAK_AKTIF": "TIDAK AKTIF - ESP32 tidak mengirim data baru, nilai DEMO dipakai",
    "TIDAK_ADA_DATA": "BELUM ADA DATA IoT - nilai DEMO dipakai",
}


def cetak_ringkasan(c):
    """Tampilkan hasil analisis lengkap di terminal (untuk laporan progres)."""
    g = "+" + "=" * 70
    print(g)
    print(f"| HASIL ANALISIS FUSASHIELD   {c['waktu']}")
    print(g)
    umur = "umur tidak diketahui" if c["umur_jam"] is None else f"umur {c['umur_jam']:.1f} jam"
    print(f"| Foto        : {c['foto']}  ({umur})")
    print(f"| CNN         : {c['label']} ({c['conf']}%)  ->  Skor Gejala {c['skor']}%")
    if c["catatan_foto"]:
        print(f"|               {c['catatan_foto']}")
    print(f"| Status IoT  : {LABEL_STATUS_IOT[c['status_iot']]}")
    for a in c["alasan_iot"]:
        print(f"|               - {a}")
    unit = {"temperature": "C", "soil_moisture": "%", "humidity": "%"}
    nama = {"temperature": "Suhu", "soil_moisture": "Kelembapan tanah", "humidity": "Kelembapan udara"}
    for n in ("temperature", "soil_moisture", "humidity"):
        tag = "[IoT ]" if c["sumber"][n] == "IoT" else "[DEMO]"
        info = "" if c["sumber"][n] == "IoT" else "  <- belum masuk dari IoT"
        print(f"|   {nama[n]:<18}: {c['nilai'][n]:>6} {unit[n]:<2} {tag}{info}")
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


def jalankan_pipeline_cloud(default_hst=41, auto_off=True):
    global _cache_cnn, _waktu_pompa_terakhir, _ack_terakhir

    print("[SCAN] Membaca sensor Firebase & memeriksa foto Supabase...")

    # --- (1) IoT: dibaca setiap scan. TIDAK menghentikan proses bila belum masuk ---
    data_iot = ambil_data_iot_firebase()
    iot = evaluasi_iot(data_iot)
    print(f"       -> [IoT] {iot['status']}")

    ack = data_iot.get("ack") if data_iot else None
    if isinstance(ack, dict) and (ack.get("id"), ack.get("status")) != _ack_terakhir:
        _ack_terakhir = (ack.get("id"), ack.get("status"))
        print(f"       -> [ACK ESP32] perintah {ack.get('id')}: {ack.get('status')} "
              f"(aktual {ack.get('durasi_aktual_ms')} ms)")

    if data_iot and "hst" in data_iot:
        hst_tanaman, sumber_hst = int(float(data_iot["hst"])), "Firebase"
    else:
        hst_tanaman, sumber_hst = default_hst, "default"

    if _waktu_pompa_terakhir == 0 and data_iot:
        _waktu_pompa_terakhir = float(data_iot.get("last_pump_epoch", 0) or 0)

    # --- (2) FOTO: CNN hanya jalan jika ada foto baru ---
    if not perbarui_cache_cnn():
        print("       -> Belum ada foto yang bisa dianalisis di Supabase. Memindai ulang...")
        return False
    cnn = _cache_cnn
    foto_baru = cnn.pop("baru", False)

    nilai, sumber = iot["nilai"], iot["sumber"]
    temperature, soil_moisture, rh_udara = nilai["temperature"], nilai["soil_moisture"], nilai["humidity"]

    # --- (3) FUZZY: selalu dihitung (sensor nyata, atau nilai DEMO jika IoT belum masuk) ---
    sekarang = time.time()
    umur_jam = ((sekarang - cnn["ts"]) / 3600.0) if cnn.get("ts") else None
    foto_basi = (umur_jam is None) or (umur_jam > FOTO_MAKS_UMUR_JAM)
    # Aturan keselamatan foto basi hanya relevan bila pompa BISA menyala.
    # Pada mode perhitungan (tanpa IoT) foto lama tetap dipakai agar demo berjalan.
    blok_foto = foto_basi and iot["siap_pompa"]
    skor_dipakai = 0.0 if blok_foto else cnn["skor"]
    catatan_foto = ""
    if blok_foto:
        catatan_foto = (f"FOTO BASI (> {FOTO_MAKS_UMUR_JAM:.0f} jam): skor CNN tidak dipakai, "
                        f"keputusan dari sensor saja (maks. preventif)")
    elif foto_basi:
        catatan_foto = "Foto lama tetap dipakai karena mode perhitungan (pompa tidak aktif)"

    dosis_ml, durasi_detik, rekomendasi, status_mitigasi, fase_hst = hitung_fuzzy_dan_workflow(
        hst_tanaman, rh_udara, soil_moisture, temperature, skor_dipakai
    )
    if blok_foto:
        status_mitigasi += " [SENSOR SAJA: foto basi]"
    tingkat = tingkat_dosis(dosis_ml)

    # --- (4) KEPUTUSAN POMPA: butuh IoT AKTIF + cooldown sesuai tingkat ---
    cooldown_detik = COOLDOWN_PER_TINGKAT.get(tingkat, 0) * SKALA_COOLDOWN
    sisa_cooldown = cooldown_detik - (sekarang - _waktu_pompa_terakhir)
    menyalakan = iot["siap_pompa"] and (rekomendasi == "ON") and (sisa_cooldown <= 0)
    pump_cmd = "ON" if menyalakan else "OFF"

    if menyalakan:
        ket_pompa = f"PERINTAH DIKIRIM ke ESP32: {dosis_ml} ml / {durasi_detik} detik"
    elif not iot["siap_pompa"]:
        ket_pompa = "TIDAK DINYALAKAN - hanya perhitungan (data IoT belum lengkap/aktif)"
    elif rekomendasi == "ON":
        ket_pompa = f"DIKUNCI cooldown {tingkat} {cooldown_detik/3600:.1f} jam, sisa {int(sisa_cooldown)} detik"
    else:
        ket_pompa = "Tidak perlu menyala (dosis < 10 ml)"

    waktu_skrg = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if any(v == "DEMO" for v in sumber.values()):
        mode_keputusan = "DEMO_TANPA_IOT"
    elif blok_foto:
        mode_keputusan = "SENSOR_SAJA"
    else:
        mode_keputusan = "CNN+SENSOR"

    # --- (5) LLM: narasi saja, SETELAH dosis & keputusan pompa final ---
    rek_llm, sumber_llm = rekomendasi_untuk({
        "cnn_label": cnn["label"], "cnn_confidence": cnn["conf"], "skor_gejala": cnn["skor"],
        "tingkat": tingkat, "dosis_fuzzy_ml": dosis_ml, "durasi_detik": durasi_detik,
        "status_mitigasi": status_mitigasi, "fase": fase_hst, "hst": hst_tanaman,
        "suhu_c": temperature, "kelembapan_tanah_persen": soil_moisture,
        "kelembapan_udara_persen": rh_udara, "sumber_sensor": sumber,
        "status_iot": iot["status"], "mode_keputusan": mode_keputusan,
        "foto_basi": foto_basi, "pompa_diizinkan": iot["siap_pompa"], "foto": cnn["path"],
    }, foto_baru)

    cetak_ringkasan({
        "waktu": waktu_skrg, "foto": cnn["path"], "umur_jam": umur_jam,
        "label": cnn["label"], "conf": cnn["conf"], "skor": cnn["skor"],
        "catatan_foto": catatan_foto, "status_iot": iot["status"], "alasan_iot": iot["alasan"],
        "nilai": nilai, "sumber": sumber, "hst": hst_tanaman, "sumber_hst": sumber_hst,
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
        "status_iot": iot["status"],
        "sumber_sensor": sumber,
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
        "foto_basi": foto_basi,
        "skor_dipakai_fuzzy": skor_dipakai,
        "mode_keputusan": mode_keputusan,
        "tingkat_dosis": tingkat,
        "cooldown_jam": round(cooldown_detik / 3600.0, 2),
        "sisa_cooldown_detik": max(0, int(sisa_cooldown)) if rekomendasi == "ON" else 0,
        "rekomendasi_llm": rek_llm,
        "rekomendasi_llm_sumber": sumber_llm,
        "last_updated": waktu_skrg
    }

    # Kegagalan menulis ke Firebase TIDAK menghentikan program (hasil tetap tampil di terminal)
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
            # Perintah sekali pakai: ESP32 yang menghitung durasi & mematikan pompa sendiri
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
        print("       -> [FIREBASE] fuzzy_output diperbarui.")
    except Exception as e:
        print(f"       -> [PERINGATAN] Gagal menulis ke Firebase ({e}). Hasil di atas tetap valid.")
    return True


if __name__ == "__main__":
    print("[INFO] Program FusaShield Cloud Controller Berjalan. Melakukan continuous scanning...")
    while True:
        try:
            jalankan_pipeline_cloud(default_hst=41, auto_off=True)
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n[INFO] Program dihentikan secara manual oleh pengguna (Ctrl+C).")
            break
        except Exception as e:
            print(f"[ERROR TERJADI] {e}. Melanjutkan scanning dalam 10 detik...")
            time.sleep(10)

