import json
import os
import threading
import time
import urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from label_translations import label_to_indonesian

"""
=============================================================================
FUSASHIELD REST API SERVER
Backend API Service untuk Dashboard Website FusaShield
=============================================================================
Endpoint dan alur data tetap sama; transport HTTP menggunakan FastAPI.
=============================================================================
"""

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))


def _muat_env(path=".env"):
    """Muat variabel environment dari file .env lokal secara otomatis."""
    lokasi = os.path.join(os.path.dirname(os.path.abspath(__file__)), path) if "__file__" in globals() else path
    if os.path.exists(lokasi):
        try:
            with open(lokasi, "r", encoding="utf-8") as f:
                for line in f:
                    b = line.strip()
                    if b and not b.startswith("#") and "=" in b:
                        k, v = b.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception as e:
            print(f"[WARN] Gagal membaca {path}: {e}")


_muat_env()

FIREBASE_DB_URL = os.environ.get("FIREBASE_DB_URL", "")
FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY", "")
NODE_PATH = os.environ.get("NODE_PATH", "/monitoring/cabai_01")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "fusashield-image")


def _fetch_firebase(path):
    """Mengambil data dari Firebase Realtime Database via REST API."""
    url = f"{FIREBASE_DB_URL}{path}.json?auth={FIREBASE_API_KEY}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "FusaShield-WebAPI/1.0"}
        )
        with urllib.request.urlopen(req, timeout=6) as response:
            raw_bytes = response.read()
            if not raw_bytes:
                return {}
            return json.loads(raw_bytes.decode("utf-8")) or {}
    except Exception as e:
        print(f"[ERROR Firebase] Gagal membaca {path}: {e}")
        return None


def _put_firebase(path, payload):
    """Mengirimkan perintah ke Firebase via PUT."""
    url = f"{FIREBASE_DB_URL}{path}.json?auth={FIREBASE_API_KEY}"
    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "FusaShield-WebAPI/1.0"
            },
            method="PUT"
        )
        with urllib.request.urlopen(req, timeout=6) as response:
            return response.status in (200, 204)
    except Exception as e:
        print(f"[ERROR Firebase] Gagal put ke {path}: {e}")
        return False


@asynccontextmanager
async def lifespan(app):
    """Jalankan satu worker controller selama aplikasi API hidup."""
    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=_loop_cloud_controller,
        args=(stop_event,),
        daemon=True
    )
    worker_thread.start()
    app.state.worker_stop_event = stop_event
    app.state.worker_thread = worker_thread
    try:
        yield
    finally:
        stop_event.set()
        worker_thread.join(timeout=5)


app = FastAPI(
    title="FusaShield REST API",
    description="Backend API untuk dashboard FusaShield.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/")
def root():
    return {
        "system": "FusaShield Precision Mitigation REST API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "GET /api/monitoring": "Seluruh data lengkap (Sensor IoT + CNN + Fuzzy + HST + Rekomendasi LLM)",
            "GET /api/iot": "Data telemetri sensor IoT real-time (suhu, kelembapan tanah, RH udara)",
            "GET /api/fuzzy": "Hasil inferensi Fuzzy Logic (dosis ml, durasi pompa, tingkat keparahan, status)",
            "GET /api/cnn": "Hasil klasifikasi gejala visual daun CNN MobileNetV2",
            "GET /api/llm": "Narasi rekomendasi agronomi XAI terkini",
            "GET /api/status": "Status kesiapan sistem (apakah IoT lengkap & foto <= 24 jam)",
            "GET /api/history?limit=20": "Riwayat log aksi mitigasi dan penyemprotan",
            "POST /api/pump/trigger": "Kirim perintah manual dari website untuk menyalakan pompa"
        }
    }


@app.get("/api/monitoring")
def get_monitoring():
    iot_data = _fetch_firebase(NODE_PATH)
    fuzzy_data = _fetch_firebase(f"{NODE_PATH}/fuzzy_output")

    if iot_data is None and fuzzy_data is None:
        return JSONResponse(
            {"error": "Gagal terhubung ke database cloud Firebase"},
            status_code=503
        )

    return {
        "success": True,
        "timestamp": int(time.time()),
        "plant_id": "cabai_01",
        "iot_sensors": {
            "temperature": iot_data.get("temperature") if iot_data else None,
            "soil_moisture": iot_data.get("soil_moisture") if iot_data else None,
            "humidity_air": iot_data.get("humidity") if iot_data else None,
            "hst": iot_data.get("hst", 41) if iot_data else 41,
            "uptime_ms": iot_data.get("uptime_ms") if iot_data else None,
            "soil_valid": iot_data.get("soil_valid", True) if iot_data else True,
            "pump_status": iot_data.get("pump_status", "OFF") if iot_data else "OFF",
            "mixer_status": iot_data.get("mixer_status", "OFF") if iot_data else "OFF"
        },
        "system_readiness": {
            "is_ready": (
                iot_data is not None
                and all(iot_data.get(k) is not None for k in ["temperature", "humidity", "soil_moisture"])
            ) and (
                fuzzy_data is not None
                and fuzzy_data.get("foto_umur_jam") is not None
                and fuzzy_data.get("foto_umur_jam") <= 24.0
            ),
            "iot_complete": iot_data is not None and all(
                iot_data.get(k) is not None for k in ["temperature", "humidity", "soil_moisture"]
            ),
            "image_valid_24h": (
                fuzzy_data is not None
                and fuzzy_data.get("foto_umur_jam") is not None
                and fuzzy_data.get("foto_umur_jam") <= 24.0
            ),
            "mode": "100% DATA RIIL (TANPA DEMO)"
        },
        "cnn_vision": {
            "label": label_to_indonesian(fuzzy_data.get("cnn_label")) if fuzzy_data else "Unknown",
            "label_original": (
                fuzzy_data.get("cnn_label_original") or fuzzy_data.get("cnn_label")
                if fuzzy_data else "Unknown"
            ),
            "confidence_pct": fuzzy_data.get("cnn_confidence") if fuzzy_data else 0.0,
            "symptom_score_pct": fuzzy_data.get("cnn_symptom_score") if fuzzy_data else 0.0,
            "image_source": fuzzy_data.get("foto_sumber") if fuzzy_data else None,
            "image_age_hours": fuzzy_data.get("foto_umur_jam") if fuzzy_data else None,
            "image_valid_24h": (
                fuzzy_data.get("foto_umur_jam") is not None
                and fuzzy_data.get("foto_umur_jam") <= 24.0
            ) if fuzzy_data else False,
            "image_public_url": (
                f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{fuzzy_data.get('foto_sumber')}"
                if fuzzy_data and fuzzy_data.get("foto_sumber") and SUPABASE_URL else None
            )
        },
        "fuzzy_decision": {
            "dosis_ml": fuzzy_data.get("dosis_rekomendasi_ml", 0.0) if fuzzy_data else 0.0,
            "durasi_detik": fuzzy_data.get("durasi_pompa_detik", 0.0) if fuzzy_data else 0.0,
            "pump_command": fuzzy_data.get("pump_command", "OFF") if fuzzy_data else "OFF",
            "status_mitigasi": fuzzy_data.get("status_mitigasi", "STANDBY") if fuzzy_data else "STANDBY",
            "fase_hst": fuzzy_data.get("fase_hst") if fuzzy_data else None,
            "tingkat_dosis": fuzzy_data.get("tingkat_dosis", "AMAN") if fuzzy_data else "AMAN",
            "cooldown_jam": fuzzy_data.get("cooldown_jam", 12.0) if fuzzy_data else 12.0,
            "sisa_cooldown_detik": fuzzy_data.get("sisa_cooldown_detik", 0) if fuzzy_data else 0
        },
        "llm_advisory": {
            "text": fuzzy_data.get("rekomendasi_llm", "") if fuzzy_data else "",
            "source": fuzzy_data.get("rekomendasi_llm_sumber", "template") if fuzzy_data else "template"
        },
        "last_updated": fuzzy_data.get("last_updated") if fuzzy_data else None
    }


@app.get("/api/iot")
def get_iot():
    iot_data = _fetch_firebase(NODE_PATH)
    if iot_data is None:
        return JSONResponse({"error": "Gagal membaca sensor IoT"}, status_code=503)
    return {
        "success": True,
        "plant_id": "cabai_01",
        "temperature": iot_data.get("temperature"),
        "soil_moisture": iot_data.get("soil_moisture"),
        "humidity": iot_data.get("humidity"),
        "hst": iot_data.get("hst", 41),
        "soil_valid": iot_data.get("soil_valid", True),
        "uptime_ms": iot_data.get("uptime_ms"),
        "pump_status": iot_data.get("pump_status", "OFF"),
        "mixer_status": iot_data.get("mixer_status", "OFF"),
        "ack": iot_data.get("ack")
    }


@app.get("/api/fuzzy")
def get_fuzzy():
    fuzzy_data = _fetch_firebase(f"{NODE_PATH}/fuzzy_output")
    if fuzzy_data is None:
        return JSONResponse({"error": "Gagal membaca output fuzzy"}, status_code=503)
    return {
        "success": True,
        "plant_id": "cabai_01",
        "dosis_rekomendasi_ml": fuzzy_data.get("dosis_rekomendasi_ml", 0.0),
        "durasi_pompa_detik": fuzzy_data.get("durasi_pompa_detik", 0.0),
        "pump_command": fuzzy_data.get("pump_command", "OFF"),
        "status_mitigasi": fuzzy_data.get("status_mitigasi"),
        "fase_hst": fuzzy_data.get("fase_hst"),
        "tingkat_dosis": fuzzy_data.get("tingkat_dosis"),
        "cooldown_jam": fuzzy_data.get("cooldown_jam"),
        "sisa_cooldown_detik": fuzzy_data.get("sisa_cooldown_detik"),
        "last_updated": fuzzy_data.get("last_updated")
    }


@app.get("/api/cnn")
def get_cnn():
    fuzzy_data = _fetch_firebase(f"{NODE_PATH}/fuzzy_output")
    if fuzzy_data is None:
        return JSONResponse({"error": "Gagal membaca data CNN"}, status_code=503)
    foto_src = fuzzy_data.get("foto_sumber")
    umur_jam = fuzzy_data.get("foto_umur_jam")
    valid_24h = umur_jam is not None and umur_jam <= 24.0
    pub_url = (
        f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{foto_src}"
        if foto_src and SUPABASE_URL else None
    )
    return {
        "success": True,
        "plant_id": "cabai_01",
        "label": label_to_indonesian(fuzzy_data.get("cnn_label")),
        "label_original": fuzzy_data.get("cnn_label_original") or fuzzy_data.get("cnn_label"),
        "confidence_pct": fuzzy_data.get("cnn_confidence"),
        "symptom_score_pct": fuzzy_data.get("cnn_symptom_score"),
        "foto_sumber": foto_src,
        "foto_public_url": pub_url,
        "foto_umur_jam": umur_jam,
        "foto_valid_24h": valid_24h
    }


@app.get("/api/llm")
def get_llm():
    fuzzy_data = _fetch_firebase(f"{NODE_PATH}/fuzzy_output")
    if fuzzy_data is None:
        return JSONResponse({"error": "Gagal membaca data LLM"}, status_code=503)
    return {
        "success": True,
        "plant_id": "cabai_01",
        "advisory": fuzzy_data.get("rekomendasi_llm", ""),
        "source": fuzzy_data.get("rekomendasi_llm_sumber", "template"),
        "last_updated": fuzzy_data.get("last_updated")
    }


@app.get("/api/history")
def get_history(request: Request):
    limit = 20
    if "limit" in request.query_params:
        try:
            limit = int(request.query_params["limit"])
        except ValueError:
            pass

    history_raw = _fetch_firebase("/history_log/cabai_01")
    logs = []
    if isinstance(history_raw, dict):
        for key, val in history_raw.items():
            if isinstance(val, dict):
                item = dict(val)
                if item.get("cnn_label") is not None:
                    item["cnn_label_original"] = item.get("cnn_label_original") or item["cnn_label"]
                    item["cnn_label"] = label_to_indonesian(item["cnn_label"])
                item["log_id"] = key
                logs.append(item)
    elif isinstance(history_raw, list):
        logs = [x for x in history_raw if x]

    logs.reverse()
    logs = logs[:limit]
    return {
        "success": True,
        "total_fetched": len(logs),
        "limit": limit,
        "logs": logs
    }


@app.post("/api/pump/trigger")
async def pump_trigger(request: Request):
    body = await request.body()
    try:
        payload_in = json.loads(body.decode("utf-8")) if body else {}
    except json.JSONDecodeError:
        payload_in = {}

    dosis = float(payload_in.get("dosis_ml", 25.0))
    durasi = float(payload_in.get("durasi_detik", dosis / 5.0))
    sekarang = int(time.time())
    perintah = {
        "id": sekarang,
        "dosis_ml": dosis,
        "durasi_detik": durasi,
        "issued_at": sekarang,
        "status_mitigasi": "MANUAL_TRIGGER_VIA_WEBSITE"
    }

    sukses = await run_in_threadpool(_put_firebase, f"{NODE_PATH}/command", perintah)
    if sukses:
        return {
            "success": True,
            "message": "Perintah pompa berhasil dikirim ke ESP32",
            "command": perintah
        }
    return JSONResponse({
        "success": False,
        "error": "Gagal mengirim perintah ke Firebase"
    }, status_code=500)


@app.api_route("/{path:path}", methods=["GET", "POST"])
def endpoint_tidak_ditemukan(path: str):
    normalized_path = f"/{path}".rstrip("/")
    return JSONResponse(
        {"error": "Endpoint tidak ditemukan", "path": normalized_path},
        status_code=404
    )


@app.options("/{path:path}")
def options_preflight():
    return Response(status_code=204)


def _loop_cloud_controller(stop_event):
    print("[BACKGROUND WORKER] Menjalankan pemindaian otomatis FusaShield di latar belakang...")
    try:
        import fusashield_main
    except Exception as e:
        print(f"[WORKER INIT ERROR] Gagal mengimpor fusashield_main: {e}")
        return

    while not stop_event.is_set():
        try:
            fusashield_main.jalankan_pipeline_cloud()
        except Exception as err:
            print(f"[WORKER ERROR] {err}. Mengulang dalam 10 detik...")
        if stop_event.wait(10):
            break


def run_api_server(host=HOST, port=PORT):
    import uvicorn

    print("=" * 65)
    print(" [FUSASHIELD FASTAPI SERVER BERJALAN]")
    print(f" URL API Lokal: http://localhost:{port}")
    print(f" URL Jaringan : http://127.0.0.1:{port}")
    print(f" Dokumentasi  : http://localhost:{port}/docs")
    print("=" * 65)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_api_server()
